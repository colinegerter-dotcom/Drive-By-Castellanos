"""
pitches table -- raw Statcast data via pybaseball, ~700k rows/season.

Pulled in weekly chunks, not one full-season call: pybaseball's statcast()
wraps Baseball Savant's CSV search, and very large single-range pulls are a
known source of timeouts / silently-truncated results in that library.
Chunking also means a failed/interrupted backfill only has to re-pull the
one broken week, not the whole season.

at_bat_id: Statcast's raw export doesn't have a single "at_bat_id" field.
We use `at_bat_number` (the at-bat's sequence number within that game),
which is exactly what the schema doc's natural key needs -- it only has to
be unique within a game, not globally, since the natural key is
(game_id, at_bat_id, pitch_number).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from pipelines.savant_client import pull_statcast_range

log = logging.getLogger(__name__)

# Must stay in sync with the same list in pipelines/games/games.py -- these
# are the game types we do NOT create rows for in mlb.games, so pitch rows
# pointing at them have nothing to reference.
SKIP_GAME_TYPES = {"S", "E", "A"}  # spring training, exhibition, all-star

# Added 23 Sep 2026, agreed with Colin: our column name -> Savant CSV column.
# Stored as-is, one value per pitch; nothing here is aggregated.
#   Movement and release: pfx_x / pfx_z are horizontal / vertical break in
#     feet (catcher's view), release_extension is feet in front of the rubber,
#     release_pos_x / _z is where the ball left the hand.
#   Expected stats: Savant's estimate for this batted ball from exit velocity
#     and launch angle (xba / xwoba), plus woba_value / woba_denom so a full
#     expected wOBA, including strikeouts and walks, can be computed.
#   Game situation: outs, runners (player ids, empty if the base is empty),
#     score before the pitch, which half-inning, and the pitch's run value
#     (change in expected runs).
EXTRA_FIELD_SOURCES = {
    "pfx_x": "pfx_x",
    "pfx_z": "pfx_z",
    "release_extension": "release_extension",
    "release_pos_x": "release_pos_x",
    "release_pos_z": "release_pos_z",
    "xba": "estimated_ba_using_speedangle",
    "xwoba": "estimated_woba_using_speedangle",
    "woba_value": "woba_value",
    "woba_denom": "woba_denom",
    "outs": "outs_when_up",
    "on_1b": "on_1b",
    "on_2b": "on_2b",
    "on_3b": "on_3b",
    "inning_topbot": "inning_topbot",
    "home_score": "home_score",
    "away_score": "away_score",
    "bat_score": "bat_score",
    "fld_score": "fld_score",
    "delta_run_exp": "delta_run_exp",
}

# Every Savant column this module reads. If Savant ever stops sending one
# (renamed or dropped), r.get() would quietly return None for every pitch and
# a whole season would be written with an empty column and a clean log. This
# repo has been bitten by that shape of failure before, so a missing column
# stops the pull instead. A column that is present but empty for some
# pitches is normal (e.g. exit velocity on a called strike) and is fine.
REQUIRED_SAVANT_COLUMNS = {
    "game_pk", "at_bat_number", "pitch_number", "pitcher", "batter", "inning",
    "balls", "strikes", "pitch_type", "release_speed", "release_spin_rate",
    "plate_x", "plate_z", "sz_top", "sz_bot", "description", "launch_speed",
    "launch_angle", "events", "bb_type", "hit_location",
    *EXTRA_FIELD_SOURCES.values(),
}


def _clean(value):
    """Any pandas/numpy scalar -> a plain Python value psycopg2 can bind.

    Does two jobs:

    1. Missing -> None. Pandas has THREE ways to say "missing" and the old
       version of this function only caught one of them:
         - None
         - float('nan')          -- classic float64 columns
         - pd.NA (NAType)        -- pandas' nullable Int64/boolean/string
                                    dtypes, which Savant's CSV now produces
       pd.NA is NOT a float, so `isinstance(value, float) and isnan(value)`
       sailed straight past it and handed psycopg2 an object it can't adapt.
       Confirmed live 21 Sep 2026: every pitch batch died with
       "ProgrammingError: can't adapt type 'NAType'", which then dragged the
       whole batch into the row-by-row fallback and wrote ~1% of the data.
       pd.isna() covers all three and is the only correct test here.

    2. numpy scalar -> Python scalar. np.float64 subclasses float so
       psycopg2 adapts it by luck, but np.int64 does NOT subclass int and
       would hit the same "can't adapt" wall the moment a column comes back
       as a nullable integer dtype. .item() unwraps both.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # pd.isna returns an array for list-likes rather than a bool. Nothing
        # we store is list-like, so treat this as a genuine value.
        pass
    unwrap = getattr(value, "item", None)
    if callable(unwrap):
        try:
            return unwrap()
        except (AttributeError, ValueError):
            pass
    return value


def date_chunks(start_date: str, end_date: str, chunk_days: int = 7):
    """Yield (chunk_start, chunk_end) date strings covering [start_date, end_date]."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur.isoformat(), chunk_end.isoformat()
        cur = chunk_end + timedelta(days=1)


def build_pitch_rows_for_range(start_date: str, end_date: str) -> list[dict]:
    """Pull and transform one (small) date range into row dicts ready for upsert.

    Callers doing a multi-week/season backfill should loop date_chunks()
    and call this once per chunk, upserting after each chunk -- that way a
    crash mid-backfill loses at most one chunk's progress, not the whole run.
    """
    df: pd.DataFrame = pull_statcast_range(start_date, end_date)
    if df is None or df.empty:
        return []

    # GAME-TYPE FILTER (21 Sep 2026). Savant returns pitches for spring
    # training, exhibitions and the All-Star game; pipelines/games/games.py
    # deliberately does NOT write rows for those types, so every such pitch
    # row violates pitches_game_id_fkey. Left unfiltered this produced ~800
    # FK failures in the first few chunks of a 2025 run (March is entirely
    # spring training), which then triggered db.upsert_rows' row-by-row
    # savepoint fallback thousands of times and exhausted Postgres' shared
    # lock table ("out of shared memory"), taking the whole database offline.
    # Dropping them here is the cheap fix; backfill_pitches ALSO filters
    # against the games table, which is the belt-and-braces guarantee.
    if "game_type" in df.columns:
        before = len(df)
        df = df[~df["game_type"].isin(SKIP_GAME_TYPES)]
        if len(df) != before:
            log.info(
                "dropped %d non-regular/postseason pitch rows (%s) for %s..%s",
                before - len(df), "/".join(sorted(SKIP_GAME_TYPES)), start_date, end_date,
            )
        if df.empty:
            return []

    missing = sorted(REQUIRED_SAVANT_COLUMNS - set(df.columns))
    if missing:
        raise RuntimeError(
            f"Savant data for {start_date}..{end_date} is missing expected columns: "
            f"{', '.join(missing)}. Refusing to write pitches with those fields silently "
            f"empty. Check whether Savant renamed them."
        )

    rows = []
    for _, r in df.iterrows():
        game_id = _clean(r.get("game_pk"))
        at_bat_id = _clean(r.get("at_bat_number"))
        pitch_number = _clean(r.get("pitch_number"))
        if game_id is None or at_bat_id is None or pitch_number is None:
            # Can't form the natural key -- skip rather than write a row
            # that'll collide with every other row missing the same fields.
            continue
        row = {
            "at_bat_id": int(at_bat_id),
            "game_id": int(game_id),
            "pitcher_id": _clean(r.get("pitcher")),
            "batter_id": _clean(r.get("batter")),
            "inning": _clean(r.get("inning")),
            "balls": _clean(r.get("balls")),
            "strikes": _clean(r.get("strikes")),
            "pitch_type": _clean(r.get("pitch_type")),
            "release_speed": _clean(r.get("release_speed")),
            "spin_rate": _clean(r.get("release_spin_rate")),
            "plate_x": _clean(r.get("plate_x")),
            "plate_z": _clean(r.get("plate_z")),
            "sz_top": _clean(r.get("sz_top")),
            "sz_bot": _clean(r.get("sz_bot")),
            "pitch_result": _clean(r.get("description")),
            "exit_velocity": _clean(r.get("launch_speed")),
            "launch_angle": _clean(r.get("launch_angle")),
            "events": _clean(r.get("events")),
            "bb_type": _clean(r.get("bb_type")),
            "hit_location": _clean(r.get("hit_location")),
            "pitch_number": int(pitch_number),
        }
        for ours, savant in EXTRA_FIELD_SOURCES.items():
            row[ours] = _clean(r.get(savant))
        rows.append(row)
    log.info("built %d pitch rows for %s..%s", len(rows), start_date, end_date)
    return rows
