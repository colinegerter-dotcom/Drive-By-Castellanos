#!/usr/bin/env python3
"""
Historical backfill for one or more seasons.

Usage:
    python scripts/backfill.py --seasons 2021 2022 2023 2024 2025
    python scripts/backfill.py --seasons 2025 --skip-pitches   # faster iteration while debugging

Why "bulk-load raw data first, compute derived form-tables second" is safe:
every derived-table query in pipelines/player_form and pipelines/games
(team_form, bullpen_status, starting_pitcher_form, starting_batter_form,
umpire_stats) filters strictly on `g.date < as_of_date` in SQL. That WHERE
clause is airtight regardless of what else is sitting in the table or what
order it was inserted in -- so it's fine, and much simpler, to load an
entire season's games/pitches/game_results up front and compute the
rolling/season stats in a second pass afterward, rather than interleaving
raw ingestion and derived-stat computation day by day. The no-lookahead
guarantee lives in the SQL filters, not in the order this script runs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Make the repo root importable regardless of how this script is invoked.
# `python scripts/backfill.py` (the way this script's own docstring, the
# README, and every command Colin has actually run tell you to run it) puts
# scripts/ -- not the repo root -- on sys.path[0], so a bare `import
# pipelines...` fails with ModuleNotFoundError even though pipelines/ is
# sitting right there one level up. Inserting the repo root here fixes that
# for this exact invocation style, without requiring `python -m
# scripts.backfill` or a manually-set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from pipelines.config import HISTORICAL_SEASONS, SEASON_START_MD, SEASON_END_MD
from pipelines.db import get_conn, upsert_rows
from pipelines.reference.teams import build_team_rows
from pipelines.reference.players import (
    collect_player_ids_for_season,
    build_player_rows,
    ensure_players_exist,
)
from pipelines.reference.park_factors import build_park_factor_rows
from pipelines.games.games import build_game_rows
from pipelines.mlb_stats_client import get_stats_by_date_range_bulk, get_career_totals_before_season
from pipelines.games.game_results import build_game_result_row, update_game_umpire
from pipelines.games.lineup import build_lineup_rows
from pipelines.games.game_conditions import build_game_condition_row, _venue_coords_by_name
from pipelines.games.team_form import build_team_form_row
from pipelines.games import bullpen_status
from pipelines.games.bullpen_status import build_bullpen_status_row
from pipelines import savant_client
from pipelines.pitches.pitches import build_pitch_rows_for_range, date_chunks
from pipelines import pitch_store
from pipelines.player_form.starting_pitcher_form import build_starting_pitcher_form_row
from pipelines.player_form.starting_batter_form import build_starting_batter_form_row, SEASON_START
from pipelines.player_form.umpire_stats import build_umpire_stats_row, umpire_k_bb_rate, league_k_bb_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backfill")

# 23 Sep 2026: sourced from pipelines/config.py instead of hardcoded here, so
# daily_pull.py's season guard and this range can't drift apart. Same values
# as before (03-01 / 11-15), same .format(season=...) call sites below.
SEASON_DATE_RANGE = (f"{{season}}-{SEASON_START_MD}", f"{{season}}-{SEASON_END_MD}")

# How many games' worth of postgame/form-table work to accumulate in memory
# before writing it as one batch and committing. Confirmed live (17 Sep
# 2026): the original per-row, never-commits-until-the-whole-season-is-done
# version of backfill_postgame/backfill_form_tables took 4+ hours and was
# still under half done on a single season, because every team/pitcher/
# batter/game wrote as its own network round trip inside one long-lived
# transaction -- and a killed run (GitHub's 6-hour job cap, a dropped
# connection, anything) would have lost ALL of it, not just the tail.
# Batching cuts round trips drastically; committing every chunk means an
# interruption only costs a re-run of the last partial chunk, which is
# cheap and safe given every write here is an idempotent upsert.
COMMIT_EVERY_N_GAMES = 100


def backfill_reference(conn, season: int, extra_player_ids: set[int] | None = None):
    log.info("[%s] teams + players", season)
    team_rows = build_team_rows(season)
    upsert_rows(conn, "teams", team_rows, conflict_cols=["team_id"])

    team_ids = [r["team_id"] for r in team_rows]
    player_ids = collect_player_ids_for_season(team_ids, season, extra_ids=extra_player_ids)
    # current_team_by_player: best-effort, last roster call wins for a
    # traded player -- good enough for "current team," which is refreshed
    # every run anyway (see players.py docstring).
    current_team_by_player: dict[int, int] = {}
    from pipelines.mlb_stats_client import get_roster

    for tid in team_ids:
        for entry in get_roster(tid, season):
            pid = (entry.get("person") or {}).get("id")
            if pid:
                current_team_by_player[pid] = tid

    player_rows = build_player_rows(player_ids, current_team_by_player)
    upsert_rows(conn, "players", player_rows, conflict_cols=["player_id"])
    log.info("[%s] %d teams, %d players", season, len(team_rows), len(player_rows))

    log.info("[%s] park_factors", season)
    # Signature changed 22 Sep 2026: park factors are now computed from our
    # own games/game_results (Savant's leaderboard is confirmed dead -- see
    # savant_client.get_park_factors), so this needs a connection.
    #
    # Ordering note: this runs BEFORE backfill_postgame for the current
    # season, which is fine and intentional. A park factor for year Y uses
    # ONLY seasons before Y, so it never wants this season's results and
    # cannot be affected by them not being loaded yet. That is also what
    # makes it safe under --skip-postgame.
    park_rows = build_park_factor_rows(
        conn, season, available_seasons=_seasons_with_results(conn)
    )
    upsert_rows(conn, "park_factors", park_rows, conflict_cols=["park_id", "year"])


def _seasons_with_results(conn) -> set[int]:
    """Seasons that actually have completed results in the database.

    Passed to build_park_factor_rows so its lookback window is intersected
    with what exists, rather than querying for seasons that were never
    backfilled. Cheap -- one grouped scan over a column we index on anyway.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            select distinct g.season
            from mlb.games g
            join mlb.game_results gr on gr.game_id = g.game_id
            where gr.game_status = 'completed'
              and gr.home_score_final is not null
            """
        )
        return {int(r[0]) for r in cur.fetchall()}


def pull_season_games(season: int) -> list[dict]:
    """Just the schedule pull (no DB writes) -- split out from backfill_games
    so main() can harvest probable-starter player_ids from the result and
    seed the players table with them BEFORE anything tries to insert a game
    row that references one via foreign key. See
    collect_player_ids_for_season's docstring for why that's necessary."""
    start, end = SEASON_DATE_RANGE[0].format(season=season), SEASON_DATE_RANGE[1].format(season=season)
    log.info("[%s] games %s..%s", season, start, end)
    return build_game_rows(start, end, season=season)


def backfill_games(conn, season: int, game_rows: list[dict]) -> list[dict]:
    upsert_rows(conn, "games", game_rows, conflict_cols=["game_id"])
    return game_rows


def _existing_game_ids(conn, game_ids: set[int]) -> set[int]:
    """Which of these game_ids actually have a row in mlb.games?

    One cheap query per chunk. Exists because Savant's date-range export
    covers game types we intentionally don't store (see SKIP_GAME_TYPES in
    pipelines/pitches/pitches.py) and, more generally, because ANY pitch row
    whose game is missing is an FK violation -- and an FK violation inside a
    large batch is what triggers the row-by-row savepoint fallback that took
    the database down on 21 Sep 2026. Cheaper to ask first than to fail.
    """
    if not game_ids:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT game_id FROM mlb.games WHERE game_id = ANY(%s)",
            (list(game_ids),),
        )
        return {r[0] for r in cur.fetchall()}


def games_for_pitch_source(conn, season: int) -> list[dict]:
    """game_id / date / season / umpire_id, read from Postgres.

    Deliberately NOT the schedule dicts that pull_season_games() returns.
    Those never carry umpire_id: MLB doesn't publish the plate umpire before
    a game, so games.py leaves it unset and update_game_umpire fills it in
    from the box score during backfill_postgame.

    Building DuckDB's games table from the schedule dicts therefore left
    umpire_id NULL for all 2,477 games. Every umpire_stats query joined on
    `g.umpire_id = $umpire_id`, matched nothing, and returned None -- so the
    table came out EMPTY with no error anywhere. Confirmed live on the
    21 Sep 2026 run. Reading from Postgres after the postgame phase is the
    only source that actually has the umpire.
    """
    with conn.cursor() as cur:
        cur.execute(
            "select game_id, date, season, umpire_id from mlb.games where season = %s",
            (season,),
        )
        return [
            {"game_id": r[0], "date": r[1], "season": r[2], "umpire_id": r[3]}
            for r in cur.fetchall()
        ]


def backfill_pitches_to_parquet(season: int, known_game_ids: set[int]):
    """Pull a season of Statcast pitches and write them to Parquet.

    Replaces the Postgres path (21 Sep 2026). Pitches are the one table whose
    shape suits columnar files rather than a row store -- see
    pipelines/pitch_store.py for the full reasoning. Nothing is aggregated or
    dropped: every pitch keeps its own row, which is what makes pitch-type,
    sequencing and platoon features possible later.

    known_game_ids comes from the schedule pull rather than a database query,
    so this function needs no connection at all.
    """
    start, end = SEASON_DATE_RANGE[0].format(season=season), SEASON_DATE_RANGE[1].format(season=season)
    total = 0
    for i, (chunk_start, chunk_end) in enumerate(date_chunks(start, end, chunk_days=7), 1):
        log.info("[%s] pitches %s..%s", season, chunk_start, chunk_end)
        rows = build_pitch_rows_for_range(chunk_start, chunk_end)

        # Same filter as the Postgres path had, for the same reason: Savant
        # returns game types we don't keep. There's no foreign key to violate
        # in Parquet, but a pitch whose game we have no row for can't be
        # joined to a date, so it would silently vanish from every query
        # anyway. Dropping it here makes the loss visible in the log.
        if rows:
            wanted = {r["game_id"] for r in rows}
            unknown = wanted - known_game_ids
            if unknown:
                kept = [r for r in rows if r["game_id"] in known_game_ids]
                log.warning(
                    "[%s] %s..%s: dropped %d pitch rows across %d game_ids not in this season's schedule",
                    season, chunk_start, chunk_end, len(rows) - len(kept), len(unknown),
                )
                rows = kept

        pitch_store.write_chunk(rows, season, i)
        total += len(rows)

    out = pitch_store.consolidate_season(season)
    log.info("[%s] pitches done: %d rows -> %s", season, total, out)
    return out


def backfill_pitches(conn, season: int):
    """Legacy Postgres pitch load. Kept so an existing database can still be
    topped up, but the backfill no longer calls it -- see
    backfill_pitches_to_parquet above.
    """
    start, end = SEASON_DATE_RANGE[0].format(season=season), SEASON_DATE_RANGE[1].format(season=season)
    total = 0
    for chunk_start, chunk_end in date_chunks(start, end, chunk_days=7):
        log.info("[%s] pitches %s..%s", season, chunk_start, chunk_end)
        rows = build_pitch_rows_for_range(chunk_start, chunk_end)

        # Drop rows whose game isn't in mlb.games rather than letting Postgres
        # reject them. See _existing_game_ids above.
        if rows:
            wanted = {r["game_id"] for r in rows}
            known = _existing_game_ids(conn, wanted)
            if len(known) != len(wanted):
                kept = [r for r in rows if r["game_id"] in known]
                log.warning(
                    "[%s] %s..%s: dropped %d pitch rows across %d game_ids with no row in mlb.games",
                    season, chunk_start, chunk_end, len(rows) - len(kept), len(wanted - known),
                )
                rows = kept

        upsert_rows(conn, "pitches", rows, conflict_cols=["game_id", "at_bat_id", "pitch_number"])
        total += len(rows)

        # Commit after every chunk. Without this the entire ~700k-row season
        # load sits in one uncommitted transaction: a single failure loses all
        # of it, and every subtransaction the fallback path opens stays pinned
        # in shared memory until commit. Chunk-sized transactions bound both.
        conn.commit()
        log.info("[%s] committed pitches through %s (%d rows this season so far)", season, chunk_end, total)


def backfill_postgame(conn, season: int, game_rows: list[dict]):
    """game_results, lineup, game_conditions (observed) -- everything that
    depends on a game actually having been played.

    Batches rows per table and commits every COMMIT_EVERY_N_GAMES games
    instead of one upsert (and one long-lived uncommitted transaction) per
    row -- see that constant's comment for why.
    """
    # Pass the season so parks renamed since then still resolve. Without it,
    # a 2021 backfill looks up "Guaranteed Rate Field" against a venue list
    # that only knows "Rate Field" and skips the weather for every White Sox
    # home game without failing. See get_venue_names_by_season.
    coords_cache = _venue_coords_by_name(seasons=[season])
    # Keyed by venue NAME (not park_id) -- matches how park_factors.py itself
    # resolves orientation, and avoids a name->id->name round trip here.
    from pipelines.reference.park_factors import _load_orientation_by_name

    orientation_by_venue_name = _load_orientation_by_name()

    game_result_rows: list[dict] = []
    lineup_rows_all: list[dict] = []
    condition_rows: list[dict] = []

    # Every player already in mlb.players, so each chunk can cheaply spot the
    # ones that appear in a box score but were never on a roster pull. See
    # ensure_players_exist -- without this, those players' lineup rows are
    # dropped by the foreign key and every chunk pays for a row-by-row retry.
    with conn.cursor() as cur:
        cur.execute("select player_id from mlb.players")
        known_player_ids = {r[0] for r in cur.fetchall()}

    def _flush():
        nonlocal known_player_ids
        known_player_ids = ensure_players_exist(
            conn, {r["player_id"] for r in lineup_rows_all}, known_player_ids
        )
        upsert_rows(conn, "game_results", game_result_rows, conflict_cols=["game_id"])
        upsert_rows(conn, "lineup", lineup_rows_all, conflict_cols=["game_id", "team_id", "player_id"])
        upsert_rows(conn, "game_conditions", condition_rows, conflict_cols=["game_id"])
        conn.commit()
        game_result_rows.clear()
        lineup_rows_all.clear()
        condition_rows.clear()

    total_games = len(game_rows)
    for i, g in enumerate(game_rows, 1):
        game_id = g["game_id"]
        result_row, umpire_id = build_game_result_row(game_id)
        update_game_umpire(conn, game_id, umpire_id)
        if result_row is not None:  # postponed/suspended/not yet played -> nothing else to write for this game
            game_result_rows.append(result_row)
            lineup_rows_all.extend(build_lineup_rows(game_id))

            orientation = orientation_by_venue_name.get(g["venue"])
            condition_row = build_game_condition_row(
                game_id=game_id,
                venue_name=g["venue"],
                first_pitch_time_iso=g["first_pitch_time"],
                game_date=g["date"],
                is_forecast=False,
                orientation_deg=orientation,
                coords_cache=coords_cache,
            )
            if condition_row:
                condition_rows.append(condition_row)

        if i % COMMIT_EVERY_N_GAMES == 0 or i == total_games:
            _flush()
            log.info("[%s] postgame: %d/%d games processed", season, i, total_games)

    log.info("[%s] game_results/lineup/game_conditions done for %d games", season, len(game_rows))


def backfill_form_tables(conn, pitches, season: int, game_rows: list[dict], resume: bool = False):
    """team_form, bullpen_status, starting_pitcher_form,
    starting_batter_form, umpire_stats -- all as-of the day before each
    game. Requires games/game_results/pitches already loaded for the season
    (see module docstring for why ordering doesn't matter for correctness,
    only for having the source rows present at all).

    This is the stage that was actually taking 4+ hours and counting on a
    real run -- team/pitcher/batter form for every player in every game. It
    took three passes to make it viable:

    1. (17 Sep) Writes were one row at a time with no commit until the whole
       season finished. Now batched per table and committed every
       COMMIT_EVERY_N_GAMES games.
    2. (18 Sep) bullpen_status refetched every prior game's box score, making
       the cost per game grow as the season went on -- quadratic, ~15 hours
       for one season. Fixed by caching in that module.
    3. (18 Sep) The remaining cost was ~60 MLB API calls per game for
       per-player stats, still 20+ minutes per 100 games. Now every player in
       a game shares one request per window (~4 calls per game), and career
       totals are prefetched once for the whole season instead of per player
       per game.
    """
    # Pull debut dates once so player_form doesn't hit the DB per player per game.
    with conn.cursor() as cur:
        cur.execute("select player_id, debut_date from mlb.players")
        debut_by_player = {r[0]: str(r[1]) if r[1] else None for r in cur.fetchall()}
        cur.execute("select team_id, division from mlb.teams")
        division_by_team = {r[0]: r[1] for r in cur.fetchall()}

    # starter_lookup: {(game_id, team_id): starter_id}, covering the whole
    # season (bullpen_status.py's season-long lookback window can reach back
    # to any game already played this year, not just games in this batch).
    starter_lookup: dict[tuple[int, int], int | None] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            select gr.game_id, g.home_team, g.away_team, gr.actual_home_starter_id, gr.actual_away_starter_id
            from mlb.game_results gr
            join mlb.games g on g.game_id = gr.game_id
            where g.season = %s
            """,
            (season,),
        )
        for gid, home_team, away_team, home_starter, away_starter in cur.fetchall():
            starter_lookup[(gid, home_team)] = home_starter
            starter_lookup[(gid, away_team)] = away_starter

    # --resume: skip games whose form rows are already committed. Because
    # _flush() writes all four tables and commits together, a game present in
    # team_form has its whole chunk done, so this is a clean restart point
    # after a job that hit GitHub's 6-hour cap. NOT the default, deliberately:
    # if the computation has changed since the interrupted run (as it did on
    # 18 Sep 2026, when the lookahead fix changed every wOBA/K%/ERA figure),
    # resuming would leave the table half old-logic and half new-logic. Only
    # pass --resume when continuing an interrupted run of the SAME code.
    already_done: set[int] = set()
    if resume:
        with conn.cursor() as cur:
            cur.execute(
                """
                select distinct tf.game_id
                from mlb.team_form tf
                join mlb.games g on g.game_id = tf.game_id
                where g.season = %s
                """,
                (season,),
            )
            already_done = {r[0] for r in cur.fetchall()}
        log.info(
            "[%s] --resume: %d games already have form rows, skipping them",
            season,
            len(already_done),
        )

    # Career totals for every player who appears this season, fetched ONCE in
    # bulk rather than once per player per game. See
    # mlb_stats_client.get_career_totals_before_season -- career-to-date is
    # rebuilt as (all prior seasons) + (this season so far), so the expensive
    # half is a constant for the whole backfill.
    with conn.cursor() as cur:
        cur.execute(
            """
            select distinct l.player_id
            from mlb.lineup l join mlb.games g on g.game_id = l.game_id
            where g.season = %s
            """,
            (season,),
        )
        season_batter_ids = [r[0] for r in cur.fetchall()]
    season_pitcher_ids = sorted(
        {pid for g in game_rows for pid in (g.get("home_starter_id"), g.get("away_starter_id")) if pid}
    )
    log.info(
        "[%s] prefetching career totals for %d batters and %d starting pitchers",
        season, len(season_batter_ids), len(season_pitcher_ids),
    )
    career_hitting = get_career_totals_before_season(season_batter_ids, "hitting", season)
    career_pitching = get_career_totals_before_season(season_pitcher_ids, "pitching", season)

    team_form_rows: list[dict] = []
    bullpen_rows: list[dict] = []
    pitcher_form_rows: list[dict] = []
    batter_form_rows: list[dict] = []

    def _flush():
        upsert_rows(conn, "team_form", team_form_rows, conflict_cols=["team_id", "game_id"])
        upsert_rows(conn, "bullpen_status", bullpen_rows, conflict_cols=["team_id", "game_id"])
        upsert_rows(conn, "starting_pitcher_form", pitcher_form_rows, conflict_cols=["pitcher_id", "game_id"])
        upsert_rows(conn, "starting_batter_form", batter_form_rows, conflict_cols=["batter_id", "game_id"])
        conn.commit()
        team_form_rows.clear()
        bullpen_rows.clear()
        pitcher_form_rows.clear()
        batter_form_rows.clear()

    def _maybe_flush(i: int):
        """Flush/commit on a chunk boundary. Deliberately a helper rather than
        inline: every `continue` in the loop below has to run this first, or a
        game skipped on the FINAL iteration would swallow the last partial
        chunk and silently lose it."""
        if i % COMMIT_EVERY_N_GAMES == 0 or i == total_games:
            _flush()
            log.info("[%s] form tables: %d/%d games processed", season, i, total_games)

    total_games = len(game_rows)
    for i, g in enumerate(game_rows, 1):
        game_id, game_date = g["game_id"], g["date"]
        home_team, away_team = g["home_team"], g["away_team"]

        if game_id in already_done:
            _maybe_flush(i)
            continue

        if not game_date:
            # The schedule can carry placeholder entries with no official date.
            # Everything below does date arithmetic on it, so skip rather than
            # crash a multi-hour run on one unplayable row.
            log.warning("[%s] game %s has no date -- skipping its form rows", season, game_id)
            _maybe_flush(i)
            continue

        for team_id in (home_team, away_team):
            division = division_by_team.get(team_id)
            if division is None:
                continue
            team_form_rows.append(build_team_form_row(conn, team_id, game_id, game_date, season, division))
            bullpen_rows.append(build_bullpen_status_row(conn, team_id, game_id, game_date, season, starter_lookup))

        with conn.cursor() as cur:
            cur.execute(
                "select player_id from mlb.lineup where game_id = %s",
                (game_id,),
            )
            batter_ids = [r[0] for r in cur.fetchall()]
        pitcher_ids = [p for p in (g.get("home_starter_id"), g.get("away_starter_id")) if p]

        # One request per window for the WHOLE game instead of three per
        # player: ~4 calls here where the old code made ~60. Every player in
        # this game shares the same as-of date, so they share the same windows.
        # These three windows MUST match what build_*_form_row computes for
        # itself in the non-prefetched path, or the batched numbers would
        # quietly differ from the per-call ones. SEASON_START is imported from
        # the form module rather than retyped here so the two can't drift.
        as_of_end = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()
        last30_start = (date.fromisoformat(game_date) - timedelta(days=30)).isoformat()
        season_start = SEASON_START.format(season=season)

        hit_season = get_stats_by_date_range_bulk(batter_ids, "hitting", season_start, as_of_end)
        hit_last30 = get_stats_by_date_range_bulk(batter_ids, "hitting", last30_start, as_of_end)
        pit_season = get_stats_by_date_range_bulk(pitcher_ids, "pitching", season_start, as_of_end)
        pit_last30 = get_stats_by_date_range_bulk(pitcher_ids, "pitching", last30_start, as_of_end)

        for pitcher_id in pitcher_ids:
            pitcher_form_rows.append(
                build_starting_pitcher_form_row(
                    pitches, pitcher_id, game_id, game_date, season, debut_by_player.get(pitcher_id),
                    prefetched={
                        "season": pit_season.get(pitcher_id),
                        "last30": pit_last30.get(pitcher_id),
                        "career_before_season": career_pitching.get(pitcher_id),
                    },
                )
            )

        for batter_id in batter_ids:
            batter_form_rows.append(
                build_starting_batter_form_row(
                    pitches, batter_id, game_id, game_date, season, debut_by_player.get(batter_id),
                    prefetched={
                        "season": hit_season.get(batter_id),
                        "last30": hit_last30.get(batter_id),
                        "career_before_season": career_hitting.get(batter_id),
                    },
                )
            )

        _maybe_flush(i)

    # umpire_stats: one row per (umpire, season), refreshed to as-of "now"
    # (end of backfill range) rather than per-game -- it's a season-level
    # aggregate (dozens of umpires, not thousands of rows), so building the
    # whole list and upserting once is fine as-is.
    with conn.cursor() as cur:
        cur.execute("select distinct umpire_id from mlb.games where season = %s and umpire_id is not null", (season,))
        umpire_ids = [r[0] for r in cur.fetchall()]
    as_of_end = SEASON_DATE_RANGE[1].format(season=season)
    umpire_rows = []
    for ump_id in umpire_ids:
        row = build_umpire_stats_row(pitches, ump_id, season, as_of_end)
        if row is None:
            continue
        u_k, u_bb = umpire_k_bb_rate(pitches, ump_id, season, as_of_end)
        lg_k, lg_bb = league_k_bb_rate(pitches, season, as_of_end)
        row["k_rate_boost"] = round(u_k - lg_k, 1) if u_k is not None and lg_k is not None else None
        row["bb_rate_boost"] = round(u_bb - lg_bb, 1) if u_bb is not None and lg_bb is not None else None
        umpire_rows.append(row)
    upsert_rows(conn, "umpire_stats", umpire_rows, conflict_cols=["umpire_id", "season"])
    conn.commit()

    log.info("[%s] form tables done", season)


def _completed_result_count(conn, season: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            select count(*)
            from mlb.games g
            join mlb.game_results gr on gr.game_id = g.game_id
            where g.season = %(season)s
              and gr.game_status = 'completed'
              and gr.home_score_final is not null
            """,
            {"season": season},
        )
        return int(cur.fetchone()[0])


# How much of a season's schedule must already have results before
# --skip-postgame is allowed to skip. Not 100%: a handful of scheduled games
# never produce a result (postponements that were never made up, ties in the
# schedule feed), so an exact match would never be reachable.
_POSTGAME_COVERAGE_FLOOR = 0.90


def _require_postgame_loaded(conn, season: int, game_rows: list[dict]) -> None:
    """Abort unless this season's postgame data is already in the database.

    --skip-postgame exists to save ~28 minutes on a RE-run. Used on a season
    that was never loaded, it would hand backfill_form_tables an empty
    game_results table, and the form build would cheerfully produce a full
    set of rows computed from nothing at all and log a clean finish. This
    repo has been bitten by exactly that shape of failure three times (see
    the build doc's standing lesson), so this fails loudly and early instead.
    """
    have = _completed_result_count(conn, season)
    want = len(game_rows)
    if want == 0:
        raise SystemExit(f"[{season}] --skip-postgame: schedule pull returned no games; aborting.")

    coverage = have / want
    if coverage < _POSTGAME_COVERAGE_FLOOR:
        raise SystemExit(
            f"[{season}] --skip-postgame refused: only {have} of {want} scheduled games "
            f"({coverage:.1%}) have completed results in the database, below the "
            f"{_POSTGAME_COVERAGE_FLOOR:.0%} floor.\n"
            f"  Skipping postgame here would build the form tables on missing data and "
            f"report success.\n"
            f"  Run this season WITHOUT --skip-postgame first, then use the flag on "
            f"subsequent re-runs.\n"
            f"  (Also expected to trip for a season still in progress, where much of the "
            f"schedule simply hasn't been played yet -- don't use --skip-postgame there.)"
        )


def build_parser() -> argparse.ArgumentParser:
    """Split out from main() so tests can assert on the flags without
    running a backfill (tests/test_backfill_guards.py)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=HISTORICAL_SEASONS)
    parser.add_argument("--skip-pitches", action="store_true", help="skip the slow Statcast pull (debugging)")
    parser.add_argument("--skip-forms", action="store_true", help="skip form-table computation (debugging)")
    parser.add_argument(
        "--skip-postgame",
        action="store_true",
        help=(
            "skip backfill_postgame (game_results, lineup, game_conditions), ~28 min per "
            "season. ONLY for re-running a season whose postgame data is already loaded -- "
            "the run aborts if it isn't, rather than building form tables on missing results."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip games that already have form rows (continue an interrupted run of the SAME code -- see backfill_form_tables)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    load_dotenv()

    for season in args.seasons:
        # Per-season, not per-run: the box-score and Savant caches are keyed
        # by game/date within a season, so there's nothing to gain from
        # carrying one season's entries into the next.
        bullpen_status.clear_cache()
        savant_client.clear_caches()

        game_rows = pull_season_games(season)
        starter_ids = {
            pid
            for g in game_rows
            for pid in (g.get("home_starter_id"), g.get("away_starter_id"))
            if pid
        }
        with get_conn() as conn:
            backfill_reference(conn, season, extra_player_ids=starter_ids)
            backfill_games(conn, season, game_rows)
        if not args.skip_pitches:
            backfill_pitches_to_parquet(season, {g["game_id"] for g in game_rows})

        with get_conn() as conn:
            if args.skip_postgame:
                # Guard, not a courtesy. backfill_form_tables reads
                # game_results and lineup; if postgame never ran for this
                # season, skipping it here would build every form table on
                # absent data and report a clean finish -- the exact failure
                # mode that produced bugs 6 and 7 in the build log. Refuse.
                _require_postgame_loaded(conn, season, game_rows)
                log.info(
                    "[%s] skipping postgame (--skip-postgame); %d games already have results",
                    season,
                    _completed_result_count(conn, season),
                )
            else:
                backfill_postgame(conn, season, game_rows)
            if not args.skip_forms:
                # The pitch source is DuckDB over this season's Parquet file,
                # opened once and reused for all ~101,000 pitch queries the
                # form build makes. Postgres stays the writer; it just isn't
                # the reader for pitch data any more.
                # Read games from Postgres, not from game_rows -- see
                # games_for_pitch_source for why (umpire_id).
                pitches = pitch_store.open_pitch_source(
                    season, games_for_pitch_source(conn, season)
                )
                try:
                    backfill_form_tables(conn, pitches, season, game_rows, resume=args.resume)
                finally:
                    pitches.close()


if __name__ == "__main__":
    main()
