#!/usr/bin/env python3
"""
Daily incremental pull. Meant to run once a day (see .github/workflows/daily-pull.yml)
via GitHub Actions cron, not run ad hoc by hand.

Order matters here, unlike backfill.py, because this script deliberately
processes only a day or two of data each run rather than reloading a whole
season (where the WHERE-clause argument in backfill.py's docstring made
ordering irrelevant):

  1. refresh teams/players (cheap, catches trades/roster moves)
  2. yesterday's Statcast pitches -> merged into the season's Parquet file
     on GitHub Releases, NOT mlb.pitches (23 Sep 2026: that table is
     retired -- see pipelines/pitch_store.py. The workflow downloads the
     current release asset before this script runs and re-uploads it
     after; see .github/workflows/daily-pull.yml).
  3. open a PitchSource over that just-merged file, same DuckDB-backed
     reader scripts/backfill.py uses, so every query below reads the
     record straight after it's written rather than a stale download.
  4. yesterday's game_results / actual lineup / observed weather / actual
     starting_batter_form (needs #2-3 done first for anything that reads
     pitches; game_results/lineup themselves come straight from the box
     score and don't)
  5. today's schedule -> mlb.games (pregame: probable starters, broadcasts)
  6. today's team_form / bullpen_status / starting_pitcher_form -- these
     read mlb.game_results and the pitch source through "yesterday", which
     steps 2-4 just made current. Running this before 2-4 would compute
     today's form stats against stale data, not violate no-lookahead (the
     date filters are still airtight) but would just be WRONG in a boring,
     avoidable way. umpire_stats is NOT refreshed here: it's keyed to the
     PRIOR season and doesn't change during the current one.
  7. today's game_conditions (forecast) + live/probable lineup (probable
     lineup itself still not built -- see README)

Runs every day of the year on the workflow's cron (23 Sep 2026: changed to
0600 Central to widen the buffer after a late West Coast game). Outside the
season window (pipelines/config.py's SEASON_START_MD/SEASON_END_MD, same
bounds scripts/backfill.py uses) run() logs and returns immediately, before
any API calls -- the schedule trigger firing daily was a deliberate choice
to keep the workflow simple, not an assumption that every firing does real
work.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make the repo root importable regardless of how this script is invoked.
# `python scripts/daily_pull.py` (what .github/workflows/daily-pull.yml
# actually runs, and what this script's own docstring tells you to run) puts
# scripts/ -- not the repo root -- on sys.path[0], so a bare `import
# pipelines...` fails with ModuleNotFoundError even though pipelines/ is
# sitting right there one level up. Inserting the repo root here fixes that
# for this exact invocation style. Same fix as scripts/backfill.py, for the
# same reason -- found when Colin hit it running backfill.py locally.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from pipelines.config import CENTRAL, CURRENT_SEASON, SEASON_START_MD, SEASON_END_MD
from pipelines.db import get_conn, upsert_rows
from pipelines.reference.teams import build_team_rows
from pipelines.reference.players import (
    collect_player_ids_for_season,
    build_player_rows,
    ensure_players_exist,
)
from pipelines.games.games import build_game_rows
from pipelines.games.game_results import build_game_result_row, update_game_umpire
from pipelines.games.lineup import build_lineup_rows
from pipelines.games.game_conditions import build_game_condition_row, _venue_coords_by_name
from pipelines.games.team_form import build_team_form_row
from pipelines.games.bullpen_status import build_bullpen_status_row
from pipelines.pitches.pitches import build_pitch_rows_for_range
from pipelines.pitch_store import merge_day, open_pitch_source
from pipelines.player_form.starting_pitcher_form import build_starting_pitcher_form_row
from pipelines.player_form.starting_batter_form import build_starting_batter_form_row
from pipelines.reference.park_factors import _load_orientation_by_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("daily_pull")


def _games_for_pitch_source(conn, season: int) -> list[dict]:
    """game_id / date / season / umpire_id, read from Postgres.

    Same helper and same reasoning as games_for_pitch_source() in
    scripts/backfill.py (duplicated rather than imported: scripts/ isn't a
    package, and this is six lines). The schedule dicts this script builds
    from the MLB API never carry umpire_id -- MLB doesn't publish the plate
    umpire pregame, only update_game_umpire() filled in from the box score
    has it. Building the PitchSource's games table from Postgres instead of
    from those schedule dicts is what keeps this correct.
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


def run(today: str | None = None):
    load_dotenv()
    now_central = datetime.now(CENTRAL)
    today = today or now_central.date().isoformat()
    season = CURRENT_SEASON

    # 23 Sep 2026: the workflow's cron fires 365 days a year (see
    # .github/workflows/daily-pull.yml), so this script -- not the schedule
    # trigger -- is what keeps the off-season quiet. Bounds shared with
    # scripts/backfill.py's SEASON_DATE_RANGE via pipelines/config.py, so
    # there's one definition of "the season" instead of two that can drift.
    # A --date override still respects this: backfilling a specific missed
    # day only makes sense inside the season anyway.
    season_start, season_end = f"{season}-{SEASON_START_MD}", f"{season}-{SEASON_END_MD}"
    if not (season_start <= today <= season_end):
        log.info(
            "%s is outside the %s season window (%s to %s) -- nothing to do",
            today, season, season_start, season_end,
        )
        return

    yesterday = (datetime.fromisoformat(today).date() - timedelta(days=1)).isoformat()

    # Pulled before the DB connection even opens, same reason as
    # backfill.py's pull_season_games: harvest probable-starter player_ids
    # from the schedule up front so the players table can be seeded with
    # them before anything inserts a game row that references one via
    # foreign key. Confirmed live (17 Sep 2026): the roster pull alone isn't
    # reliably complete -- see collect_player_ids_for_season's docstring.
    # Reused below instead of re-pulling the same schedule twice.
    yesterday_games = build_game_rows(yesterday, yesterday, season=season)
    today_games = build_game_rows(today, today, season=season)
    starter_ids = {
        pid
        for g in (yesterday_games + today_games)
        for pid in (g.get("home_starter_id"), g.get("away_starter_id"))
        if pid
    }

    with get_conn() as conn:
        log.info("refreshing teams/players")
        team_rows = build_team_rows(season)
        upsert_rows(conn, "teams", team_rows, conflict_cols=["team_id"])
        team_ids = [r["team_id"] for r in team_rows]

        from pipelines.mlb_stats_client import get_roster

        current_team_by_player = {}
        for tid in team_ids:
            for entry in get_roster(tid, season):
                pid = (entry.get("person") or {}).get("id")
                if pid:
                    current_team_by_player[pid] = tid
        player_ids = collect_player_ids_for_season(team_ids, season, extra_ids=starter_ids)
        player_rows = build_player_rows(player_ids, current_team_by_player)
        upsert_rows(conn, "players", player_rows, conflict_cols=["player_id"])

        # Needed by both the batter-form step below (yesterday's actual
        # lineups) and the pitcher-form step further down (today's probable
        # starters), so fetched once up front rather than twice.
        with conn.cursor() as cur:
            cur.execute("select player_id, debut_date from mlb.players")
            debut_by_player = {r[0]: str(r[1]) if r[1] else None for r in cur.fetchall()}
            cur.execute("select team_id, division from mlb.teams")
            division_by_team = {r[0]: r[1] for r in cur.fetchall()}

        log.info("pulling yesterday's pitches (%s)", yesterday)
        pitch_rows = build_pitch_rows_for_range(yesterday, yesterday)
        # 23 Sep 2026: merged into the season's Parquet file, NOT upserted
        # into Postgres -- mlb.pitches is retired (see pipelines/pitch_store.py).
        # The workflow has already downloaded the current release asset into
        # data/ before this script runs, and uploads the merged result back
        # after it finishes; see .github/workflows/daily-pull.yml.
        merge_day(season, pitch_rows)

        # Every query below that touches pitch-level data reads through this
        # PitchSource, not through `conn` -- see pipelines/pitch_store.py and
        # pipelines/player_form/sql_helpers.py. Passing the Postgres `conn`
        # into build_starting_pitcher_form_row / build_starting_batter_form_row
        # would run DuckDB-dialect SQL ($name parameters, unqualified table
        # names) against a psycopg2 cursor and fail outright -- confirmed live
        # 23 Sep 2026, see the status page's decision log.
        pitches = open_pitch_source(season, _games_for_pitch_source(conn, season))
        try:
            log.info("processing yesterday's game results (%s)", yesterday)
            upsert_rows(conn, "games", yesterday_games, conflict_cols=["game_id"])
            coords_cache = _venue_coords_by_name()
            orientation_by_venue = _load_orientation_by_name()
            for g in yesterday_games:
                result_row, umpire_id = build_game_result_row(g["game_id"])
                update_game_umpire(conn, g["game_id"], umpire_id)
                if result_row is None:
                    continue
                upsert_rows(conn, "game_results", [result_row], conflict_cols=["game_id"])
                lineup_rows = build_lineup_rows(g["game_id"])
                # Same gap the backfill hit: a player can appear in a box score
                # without having been on any roster pull, and his lineup row would
                # otherwise be silently dropped by the foreign key every night.
                ensure_players_exist(conn, {r["player_id"] for r in lineup_rows})
                upsert_rows(conn, "lineup", lineup_rows, conflict_cols=["game_id", "team_id", "player_id"])

                # Actual starting_batter_form for yesterday's real lineups --
                # not wired up before 23 Sep 2026 (build_starting_batter_form_row
                # was imported but never called). Goes on the "yesterday" side,
                # same as game_results/lineup/weather, because there is no
                # confirmed lineup to score pregame -- only probable starters
                # are known that far ahead. Uses the pitch source, which by
                # this point in the run includes yesterday's just-merged pitches.
                for batter_id in {r["player_id"] for r in lineup_rows}:
                    bf_row = build_starting_batter_form_row(
                        pitches, batter_id, g["game_id"], yesterday, season,
                        debut_by_player.get(batter_id),
                    )
                    upsert_rows(conn, "starting_batter_form", [bf_row], conflict_cols=["batter_id", "game_id"])

                cond_row = build_game_condition_row(
                    game_id=g["game_id"],
                    venue_name=g["venue"],
                    first_pitch_time_iso=g["first_pitch_time"],
                    game_date=g["date"],
                    is_forecast=False,
                    orientation_deg=orientation_by_venue.get(g["venue"]),
                    coords_cache=coords_cache,
                )
                if cond_row:
                    upsert_rows(conn, "game_conditions", [cond_row], conflict_cols=["game_id"])

            log.info("upserting today's schedule (%s)", today)
            upsert_rows(conn, "games", today_games, conflict_cols=["game_id"])

            log.info("computing today's pregame form tables")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select gr.game_id, g.home_team, g.away_team, gr.actual_home_starter_id, gr.actual_away_starter_id
                    from mlb.game_results gr join mlb.games g on g.game_id = gr.game_id
                    where g.season = %s
                    """,
                    (season,),
                )
                starter_lookup = {}
                for gid, home_team, away_team, home_starter, away_starter in cur.fetchall():
                    starter_lookup[(gid, home_team)] = home_starter
                    starter_lookup[(gid, away_team)] = away_starter

            for g in today_games:
                game_id, game_date = g["game_id"], g["date"]
                for team_id in (g["home_team"], g["away_team"]):
                    division = division_by_team.get(team_id)
                    if division is None:
                        continue
                    tf_row = build_team_form_row(conn, team_id, game_id, game_date, season, division)
                    upsert_rows(conn, "team_form", [tf_row], conflict_cols=["team_id", "game_id"])
                    bp_row = build_bullpen_status_row(conn, team_id, game_id, game_date, season, starter_lookup)
                    upsert_rows(conn, "bullpen_status", [bp_row], conflict_cols=["team_id", "game_id"])

                for pitcher_id in (g.get("home_starter_id"), g.get("away_starter_id")):
                    if pitcher_id is None:
                        continue
                    row = build_starting_pitcher_form_row(
                        pitches, pitcher_id, game_id, game_date, season, debut_by_player.get(pitcher_id)
                    )
                    upsert_rows(conn, "starting_pitcher_form", [row], conflict_cols=["pitcher_id", "game_id"])

                # Probable (not yet actual) lineup for today's games, and forecast
                # weather -- both get overwritten with the actual/observed version
                # tomorrow once the game is final. NOTE: build_lineup_rows() as
                # currently written reads the box score, which is empty pregame --
                # a live-lineup puller reading the schedule's probable/pregame
                # lineup endpoint instead is flagged as not-yet-built; see README.
                cond_row = build_game_condition_row(
                    game_id=game_id,
                    venue_name=g["venue"],
                    first_pitch_time_iso=g["first_pitch_time"],
                    game_date=game_date,
                    is_forecast=True,
                    orientation_deg=orientation_by_venue.get(g["venue"]),
                    coords_cache=coords_cache,
                )
                if cond_row:
                    upsert_rows(conn, "game_conditions", [cond_row], conflict_cols=["game_id"])
        finally:
            pitches.close()

    log.info("daily pull complete for %s", today)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="override 'today' (YYYY-MM-DD), for backfilling a missed day")
    args = parser.parse_args()
    run(today=args.date)
