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
import math
from datetime import date, timedelta

import pandas as pd

from pipelines.savant_client import pull_statcast_range

log = logging.getLogger(__name__)


def _clean(value):
    """NaN -> None. pybaseball/pandas fills missing numeric fields with NaN,
    which psycopg2 will happily insert as the string 'NaN' unless we convert
    it -- that's silent data corruption, not a null.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
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

    rows = []
    for _, r in df.iterrows():
        game_id = _clean(r.get("game_pk"))
        at_bat_id = _clean(r.get("at_bat_number"))
        pitch_number = _clean(r.get("pitch_number"))
        if game_id is None or at_bat_id is None or pitch_number is None:
            # Can't form the natural key -- skip rather than write a row
            # that'll collide with every other row missing the same fields.
            continue
        rows.append(
            {
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
        )
    log.info("built %d pitch rows for %s..%s", len(rows), start_date, end_date)
    return rows
