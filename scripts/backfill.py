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

from pipelines.config import HISTORICAL_SEASONS
from pipelines.db import get_conn, upsert_rows
from pipelines.reference.teams import build_team_rows
from pipelines.reference.players import collect_player_ids_for_season, build_player_rows
from pipelines.reference.park_factors import build_park_factor_rows
from pipelines.games.games import build_game_rows
from pipelines.games.game_results import build_game_result_row, update_game_umpire
from pipelines.games.lineup import build_lineup_rows
from pipelines.games.game_conditions import build_game_condition_row, _venue_coords_by_name
from pipelines.games.team_form import build_team_form_row
from pipelines.games.bullpen_status import build_bullpen_status_row
from pipelines.pitches.pitches import build_pitch_rows_for_range, date_chunks
from pipelines.player_form.starting_pitcher_form import build_starting_pitcher_form_row
from pipelines.player_form.starting_batter_form import build_starting_batter_form_row
from pipelines.player_form.umpire_stats import build_umpire_stats_row, umpire_k_bb_rate, league_k_bb_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backfill")

SEASON_DATE_RANGE = ("{season}-03-01", "{season}-11-15")  # covers spring training cutoff through World Series


def backfill_reference(conn, season: int):
    log.info("[%s] teams + players", season)
    team_rows = build_team_rows(season)
    upsert_rows(conn, "teams", team_rows, conflict_cols=["team_id"])

    team_ids = [r["team_id"] for r in team_rows]
    player_ids = collect_player_ids_for_season(team_ids, season)
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
    park_rows = build_park_factor_rows(season)
    upsert_rows(conn, "park_factors", park_rows, conflict_cols=["park_id", "year"])


def backfill_games(conn, season: int) -> list[dict]:
    start, end = SEASON_DATE_RANGE[0].format(season=season), SEASON_DATE_RANGE[1].format(season=season)
    log.info("[%s] games %s..%s", season, start, end)
    game_rows = build_game_rows(start, end, season=season)
    upsert_rows(conn, "games", game_rows, conflict_cols=["game_id"])
    return game_rows


def backfill_pitches(conn, season: int):
    start, end = SEASON_DATE_RANGE[0].format(season=season), SEASON_DATE_RANGE[1].format(season=season)
    for chunk_start, chunk_end in date_chunks(start, end, chunk_days=7):
        log.info("[%s] pitches %s..%s", season, chunk_start, chunk_end)
        rows = build_pitch_rows_for_range(chunk_start, chunk_end)
        upsert_rows(conn, "pitches", rows, conflict_cols=["game_id", "at_bat_id", "pitch_number"])


def backfill_postgame(conn, season: int, game_rows: list[dict]):
    """game_results, lineup, game_conditions (observed) -- everything that
    depends on a game actually having been played.
    """
    coords_cache = _venue_coords_by_name()
    # Keyed by venue NAME (not park_id) -- matches how park_factors.py itself
    # resolves orientation, and avoids a name->id->name round trip here.
    from pipelines.reference.park_factors import _load_orientation_by_name

    orientation_by_venue_name = _load_orientation_by_name()

    for g in game_rows:
        game_id = g["game_id"]
        result_row, umpire_id = build_game_result_row(game_id)
        update_game_umpire(conn, game_id, umpire_id)
        if result_row is None:
            continue  # postponed/suspended/not yet played
        upsert_rows(conn, "game_results", [result_row], conflict_cols=["game_id"])

        lineup_rows = build_lineup_rows(game_id)
        upsert_rows(conn, "lineup", lineup_rows, conflict_cols=["game_id", "team_id", "player_id"])

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
            upsert_rows(conn, "game_conditions", [condition_row], conflict_cols=["game_id"])

    log.info("[%s] game_results/lineup/game_conditions done for %d games", season, len(game_rows))


def backfill_form_tables(conn, season: int, game_rows: list[dict]):
    """team_form, bullpen_status, starting_pitcher_form,
    starting_batter_form, umpire_stats -- all as-of the day before each
    game. Requires games/game_results/pitches already loaded for the season
    (see module docstring for why ordering doesn't matter for correctness,
    only for having the source rows present at all).
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

    for g in game_rows:
        game_id, game_date = g["game_id"], g["date"]
        home_team, away_team = g["home_team"], g["away_team"]

        for team_id in (home_team, away_team):
            division = division_by_team.get(team_id)
            if division is None:
                continue
            tf_row = build_team_form_row(conn, team_id, game_id, game_date, season, division)
            upsert_rows(conn, "team_form", [tf_row], conflict_cols=["team_id", "game_id"])

            bp_row = build_bullpen_status_row(conn, team_id, game_id, game_date, season, starter_lookup)
            upsert_rows(conn, "bullpen_status", [bp_row], conflict_cols=["team_id", "game_id"])

        for pitcher_id, side in ((g.get("home_starter_id"), "home"), (g.get("away_starter_id"), "away")):
            if pitcher_id is None:
                continue
            row = build_starting_pitcher_form_row(
                conn, pitcher_id, game_id, game_date, season, debut_by_player.get(pitcher_id)
            )
            upsert_rows(conn, "starting_pitcher_form", [row], conflict_cols=["pitcher_id", "game_id"])

        with conn.cursor() as cur:
            cur.execute(
                "select player_id from mlb.lineup where game_id = %s",
                (game_id,),
            )
            batter_ids = [r[0] for r in cur.fetchall()]
        for batter_id in batter_ids:
            row = build_starting_batter_form_row(
                conn, batter_id, game_id, game_date, season, debut_by_player.get(batter_id)
            )
            upsert_rows(conn, "starting_batter_form", [row], conflict_cols=["batter_id", "game_id"])

    # umpire_stats: one row per (umpire, season), refreshed to as-of "now"
    # (end of backfill range) rather than per-game -- it's a season-level
    # aggregate, not a per-game one.
    with conn.cursor() as cur:
        cur.execute("select distinct umpire_id from mlb.games where season = %s and umpire_id is not null", (season,))
        umpire_ids = [r[0] for r in cur.fetchall()]
    as_of_end = SEASON_DATE_RANGE[1].format(season=season)
    for ump_id in umpire_ids:
        row = build_umpire_stats_row(conn, ump_id, season, as_of_end)
        if row is None:
            continue
        u_k, u_bb = umpire_k_bb_rate(conn, ump_id, season, as_of_end)
        lg_k, lg_bb = league_k_bb_rate(conn, season, as_of_end)
        row["k_rate_boost"] = round(u_k - lg_k, 1) if u_k is not None and lg_k is not None else None
        row["bb_rate_boost"] = round(u_bb - lg_bb, 1) if u_bb is not None and lg_bb is not None else None
        upsert_rows(conn, "umpire_stats", [row], conflict_cols=["umpire_id", "season"])

    log.info("[%s] form tables done", season)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=HISTORICAL_SEASONS)
    parser.add_argument("--skip-pitches", action="store_true", help="skip the slow Statcast pull (debugging)")
    parser.add_argument("--skip-forms", action="store_true", help="skip form-table computation (debugging)")
    args = parser.parse_args()

    load_dotenv()

    for season in args.seasons:
        with get_conn() as conn:
            backfill_reference(conn, season)
            game_rows = backfill_games(conn, season)
        if not args.skip_pitches:
            with get_conn() as conn:
                backfill_pitches(conn, season)
        with get_conn() as conn:
            backfill_postgame(conn, season, game_rows)
            if not args.skip_forms:
                backfill_form_tables(conn, season, game_rows)


if __name__ == "__main__":
    main()
