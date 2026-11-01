#!/usr/bin/env python3
"""
Export the private odds folder that market_d.py and run_holdout.py read.

    python scripts/export_odds_dir.py --out <private folder outside the repo>

Added in the post-mortem (9 Oct 2026). Until now this folder was built by
ad hoc session scripts that were never committed, so nobody could rebuild
the G4 inputs from the repo. This script writes:

  ml.parquet           moneyline: game_id, season, book, snap, ts, home_odds, away_odds
  rl.parquet           run line: the same plus line (home line, -1.5 or +1.5)
  games_2022_2024.csv  development seasons (market_d.py)
  games_holdout.csv    2025-2026 (run_holdout.py, scoring only)

snap is mlb.odds_*.snapshot_type: open, p1 (10am CT), lineup (first pitch
minus 90 minutes) and close (5 minutes before first pitch). Only the Covers
source is used, as in every G4 run.

HARD RULE: the output holds scraped commercial odds. Write it to a private
folder that is NOT inside the repo, never commit it, never upload it. The
script refuses an output folder inside the repo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from pipelines.config import db_dsn  # noqa: E402

ODDS = """
    select o.game_id, g.season, o.book, o.snapshot_type as snap,
           to_char(o.timestamp at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"+00:00"') as ts,
           {line} o.home_odds, o.away_odds
    from mlb.{table} o join mlb.games g using (game_id)
    where o.source = 'covers'
    order by o.game_id, o.book, o.snapshot_type{order_line}"""

GAMES = """
    select g.game_id, g.season, g.date, g.game_type, r.game_status as status,
           r.innings_played as innings, r.home_score_final as home_final,
           r.away_score_final as away_final, g.venue_id
    from mlb.games g join mlb.game_results r using (game_id)
    where g.season between {lo} and {hi} order by g.game_id"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out = a.out.resolve()
    if REPO in out.parents or out == REPO:
        raise SystemExit("refusing: the odds folder must be outside the repo (scraped odds stay private)")
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(REPO / ".env")
    con = psycopg2.connect(db_dsn())
    con.set_session(readonly=True, isolation_level="REPEATABLE READ")   # one consistent moment
    try:
        ml = pd.read_sql(ODDS.format(line="", table="odds_moneyline", order_line=""), con)
        rl = pd.read_sql(ODDS.format(line="o.line::float8 as line,", table="odds_runline", order_line=", o.line"), con)
        for df in (ml, rl):
            for c in ("game_id", "season", "home_odds", "away_odds"):
                df[c] = df[c].astype("int64")
        ml.to_parquet(out / "ml.parquet", index=False)
        rl.to_parquet(out / "rl.parquet", index=False)
        pd.read_sql(GAMES.format(lo=2022, hi=2024), con).to_csv(out / "games_2022_2024.csv", index=False)
        pd.read_sql(GAMES.format(lo=2025, hi=2026), con).to_csv(out / "games_holdout.csv", index=False)
    finally:
        con.close()
    print(f"ml {len(ml)} rows, rl {len(rl)} rows written to {out}")


if __name__ == "__main__":
    main()
