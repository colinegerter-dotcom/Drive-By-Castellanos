#!/usr/bin/env python3
"""
Export the database tables the feature builder reads into a folder.

    python scripts/export_feature_inputs.py --out inputs/

The pitch files are downloaded separately (the workflow does it with
`gh release download`). Everything is read inside one REPEATABLE READ,
read-only transaction, so all tables come from the same moment even if the
nightly job writes mid-export.

Column names and formats match pipelines/features/inputs.py exactly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from pipelines.config import db_dsn  # noqa: E402

FIRST_SEASON = 2021   # feature rows start here
FIRST_LINES = 2015    # official season lines used as priors go back to here

QUERIES = {
    "games.csv": f"""
        select g.game_id, g.date, g.season, g.home_team, g.away_team,
               coalesce(r.actual_home_starter_id, g.home_starter_id) as home_sp,
               coalesce(r.actual_away_starter_id, g.away_starter_id) as away_sp,
               extract(epoch from g.first_pitch_time)::bigint as fp_sched,
               extract(epoch from g.actual_first_pitch)::bigint as fp_actual,
               g.day_night, case when g.doubleheader_flag then 't' else 'f' end as dh,
               g.game_type, g.venue_id, g.venue, r.game_status as status,
               r.innings_played as innings, r.home_score_final as home_final, r.away_score_final as away_final
        from mlb.games g left join mlb.game_results r using (game_id)
        where g.season >= {FIRST_SEASON} order by g.game_id""",
    "lineup.csv": f"""
        select l.game_id, l.team_id, l.batting_order_slot as slot, l.player_id, l.bats_hand as bats,
               l.defensive_position as pos
        from mlb.lineup l join mlb.games g using (game_id) where g.season >= {FIRST_SEASON}
        order by 1, 2, 3""",
    "players.csv": """
        select player_id, bats, throws, birth_date, debut_date, primary_position as position
        from mlb.players order by 1""",
    "season_lines.csv": f"""
        select player_id, season, stat_group as grp, coalesce(num_teams, 1) as num_teams, age,
               coalesce(games, 0) g, coalesce(games_started, 0) gs, coalesce(plate_appearances, 0) pa,
               coalesce(at_bats, 0) ab, coalesce(batters_faced, 0) bf, coalesce(outs, 0) outs,
               coalesce(hits, 0) h, coalesce(doubles, 0) d2, coalesce(triples, 0) d3,
               coalesce(home_runs, 0) hr, coalesce(walks, 0) bb, coalesce(intentional_walks, 0) ibb,
               coalesce(hit_by_pitch, 0) hbp, coalesce(strikeouts, 0) so, coalesce(sac_flies, 0) sf,
               coalesce(ground_outs, 0) go, coalesce(air_outs, 0) ao, coalesce(runs, 0) r,
               coalesce(earned_runs, 0) er
        from mlb.player_season_stats where season >= {FIRST_LINES} order by 1, 2, 3""",
    "pf.csv": f"""
        select park_id, year, park_factor_runs as pf_runs from mlb.park_factors
        where park_factor_runs is not null and year >= {FIRST_SEASON} order by 1, 2""",
    "gc.csv": f"""
        select c.game_id, c.temp_f, c.wind_speed, c.wind_direction as wind_dir, c.humidity,
               c.precip_flag as precip, c.wind_effect, c.is_forecast
        from mlb.game_conditions c join mlb.games g using (game_id) where g.season >= {FIRST_SEASON}
        order by 1""",
    "tr.csv": """
        select game_id, team_id, is_home, runs_f5, runs_8, runs_total,
               last_inning_batted as last_inning, innings_played as innings
        from mlb.team_game_runs order by 1, 2""",
    "rg.csv": "select game_id, original_date, resume_date from mlb.resumed_games order by 1",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    load_dotenv()
    a.out.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(db_dsn())
    conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
    try:
        for name, sql in QUERIES.items():
            df = pd.read_sql(sql, conn)
            # nullable integers stay integers in the CSV (no "123.0")
            df = df.convert_dtypes()
            df.to_csv(a.out / name, index=False)
            print(f"{name:18s} {len(df):8d} rows")
            if len(df) == 0 and name != "rg.csv":
                raise SystemExit(f"{name} is empty; refusing to build features from it")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
