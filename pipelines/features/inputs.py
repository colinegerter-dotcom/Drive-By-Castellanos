"""
Load everything the feature builder reads into one in-memory DuckDB.

Why a folder of files instead of reading Postgres directly: the design says
training reads a fixed, dated copy of the data (a snapshot), never live
tables that the nightly job keeps changing. So the builder only ever reads a
folder. scripts/export_feature_inputs.py fills that folder from Postgres
(or a db snapshot) plus the pitch files; tests fill it with tiny fake data.

Expected files in the folder:
  pitches_<season>.parquet   one per season (the GitHub release files)
  games.csv                  one row per game, starters are the actual ones
  lineup.csv                 true starting lineups (phase A1)
  players.csv                bats / throws / birth date
  season_lines.csv           official MLB season lines (phase A5)
  pf.csv                     runs park factors by venue id and year
  gc.csv                     game-time weather
  tr.csv                     runs per team-game (the targets, phase A3)
  rg.csv                     resumed games
"""
from __future__ import annotations

from pathlib import Path

import duckdb

# Venues with a roof (retractable or fixed), by MLB venue id. Same list as
# pipelines/config.py ROOFED_VENUE_IDS; repeated here so the feature builder
# has no import-time dependency on database settings.
ROOFED_VENUE_IDS = {32, 15, 2392, 5325, 4169, 14, 680, 12}
# Fixed dome: the roof is always closed, so weather never matters there.
# 12 = Tropicana Field. Every other id above has a retractable roof.
DOME_VENUE_IDS = {12}


def connect(folder: str | Path, seasons: list[int] | None = None) -> duckdb.DuckDBPyConnection:
    """Open DuckDB with views over every input file.

    seasons: which pitch files to load (default: every pitches_*.parquet).
    """
    folder = Path(folder)
    con = duckdb.connect()
    con.execute("set preserve_insertion_order = false")

    files = sorted(folder.glob("pitches_*.parquet"))
    if seasons is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) in seasons]
    if not files:
        raise FileNotFoundError(f"no pitch files in {folder}")
    # union_by_name: older files and newer files may differ slightly in
    # column order; the season comes from the file name, not a column.
    parts = [f"select *, {int(f.stem.split('_')[1])} as season from read_parquet('{f.as_posix()}')" for f in files]
    con.execute("create view pitches_raw as " + " union all by name ".join(parts))

    def csv(name: str) -> str:
        return f"read_csv('{(folder / name).as_posix()}', header=true, auto_detect=true)"

    con.execute(f"create table resumed as select game_id, original_date::date original_date, resume_date::date resume_date from {csv('rg.csv')}")
    con.execute(f"""
        create table games as
        select g.game_id, g.date::date as date, g.season, g.home_team, g.away_team,
               g.home_sp, g.away_sp,
               to_timestamp(g.fp_sched) as fp_sched,
               to_timestamp(coalesce(g.fp_actual, g.fp_sched)) as fp_actual,
               g.day_night, (lower(g.dh::varchar) in ('t', 'true')) as dh, g.game_type, g.venue_id,
               g.status, g.innings,
               -- data_date: the day the game's data can first be used. A
               -- resumed game finished on its resume date, so ALL of its
               -- pitches are filed under that date (design A4).
               coalesce(r.resume_date, g.date::date) as data_date,
               (r.game_id is not null) as resumed
        from {csv('games.csv')} g left join resumed r using (game_id)
    """)
    con.execute(f"create table lineup as select * from {csv('lineup.csv')}")
    con.execute(f"create table players as select * from {csv('players.csv')}")
    con.execute(f"create table season_lines as select * from {csv('season_lines.csv')}")
    con.execute(f"create table park_factors as select park_id::int venue_id, year, pf_runs from {csv('pf.csv')}")
    con.execute(f"create table conditions as select * from {csv('gc.csv')}")
    con.execute(f"create table team_runs as select * from {csv('tr.csv')}")

    # Pitches joined to their game, so every pitch knows its date and which
    # team was batting and which was fielding.
    con.execute("""
        create view pitches as
        select p.*, g.date, g.data_date, g.game_type,
               case when p.inning_topbot = 'Top' then g.away_team else g.home_team end as bat_team,
               case when p.inning_topbot = 'Top' then g.home_team else g.away_team end as fld_team
        from pitches_raw p join games g using (game_id)
    """)
    return con
