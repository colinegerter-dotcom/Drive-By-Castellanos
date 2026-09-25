"""
Future-perturbation leak test (design 5.8, test 1).

Idea: scramble everything that happened on or after a date D, rebuild the
features, and check that no game dated on or before D changed. If a feature
moved, it was reading the future.

    python -m pipelines.features.leak_test --inputs <folder> --date 2022-06-15 --point P2

What gets scrambled, for data dated D or later:
  - every pitch outcome: all plate appearances become home runs, fastball
    velocity +5 mph, all balls in play become ground balls, all pitch types
    four-seamers
  - who pitched and who batted, innings, scores and pitch numbers
  - runs per team-game (targets and team strength inputs) x3
  - game results
  - official season lines for D's season and later
  - park factors for years after D's
What gets scrambled only AFTER D (game D's own values are legitimately known
for game D: its starters, venue, game-time weather, and at P2 its confirmed
lineup):
  - listed starters and venues
  - starting lineups (players shuffled, slots shuffled); at P1 the lineups
    of D itself are scrambled too, because P1 must not see them
  - weather
(Extended 25 Sep after the independent review found the first version left
ids, innings, scores, starters, venues, season lines and park factors alone.)

D must be in 2022 or later: 2021 is the warm-up season whose full-season
data is used to fit two small pieces (see pitching.py), by design.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from .build import build_features



def perturb(src: Path, dst: Path, d: date, point: str = "P2") -> None:
    dst.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    games = pd.read_csv(src / "games.csv")
    rg = pd.read_csv(src / "rg.csv")
    data_date = pd.to_datetime(games["date"]).dt.date
    rmap = dict(zip(rg.game_id, pd.to_datetime(rg.resume_date).dt.date))
    games["data_date"] = [rmap.get(g, dd) for g, dd in zip(games.game_id, data_date)]
    late = set(games.loc[games.data_date >= d, "game_id"])
    after = set(games.loc[pd.to_datetime(games["date"]).dt.date > d, "game_id"])
    con.register("late_df", pd.DataFrame({"game_id": sorted(late)}))

    for f in src.glob("pitches_*.parquet"):
        con.execute(f"""
            copy (
              select * replace (
                case when game_id in (select game_id from late_df) and events is not null then 'home_run' else events end as events,
                case when game_id in (select game_id from late_df) and events is not null then 2.0 else woba_value end as woba_value,
                case when game_id in (select game_id from late_df) and events is not null then 1 else woba_denom end as woba_denom,
                case when game_id in (select game_id from late_df) then release_speed + 5 else release_speed end as release_speed,
                case when game_id in (select game_id from late_df) and bb_type is not null then 'ground_ball' else bb_type end as bb_type,
                case when game_id in (select game_id from late_df) then 'FF' else pitch_type end as pitch_type,
                case when game_id in (select game_id from late_df) then pitcher_id % 997 + 1 else pitcher_id end as pitcher_id,
                case when game_id in (select game_id from late_df) then batter_id % 991 + 1 else batter_id end as batter_id,
                case when game_id in (select game_id from late_df) then 0 else bat_score end as bat_score,
                case when game_id in (select game_id from late_df) then 9 else fld_score end as fld_score,
                case when game_id in (select game_id from late_df) then 8 else inning end as inning,
                case when game_id in (select game_id from late_df) then pitch_number * 3 else pitch_number end as pitch_number
              )
              from read_parquet('{f.as_posix()}')
            ) to '{(dst / f.name).as_posix()}' (format parquet)
        """)

    tr = pd.read_csv(src / "tr.csv")
    m = tr.game_id.isin(late)
    for c in ("runs_f5", "runs_8", "runs_total"):
        tr.loc[m, c] = tr.loc[m, c] * 3 + 1
    tr.to_csv(dst / "tr.csv", index=False)

    g2 = games.drop(columns=["data_date"]).copy()
    m = g2.game_id.isin(late)
    g2.loc[m, "home_final"] = 99
    m = g2.game_id.isin(after)
    g2.loc[m, "home_sp"] = g2.loc[m, "home_sp"].sample(frac=1, random_state=3).to_numpy()
    g2.loc[m, "venue_id"] = 1
    g2.to_csv(dst / "games.csv", index=False)

    lu = pd.read_csv(src / "lineup.csv")
    on_or_after = set(games.loc[pd.to_datetime(games["date"]).dt.date >= d, "game_id"])
    m = lu.game_id.isin(on_or_after if point == "P1" else after)
    lu.loc[m, "player_id"] = lu.loc[m, "player_id"].sample(frac=1, random_state=7).to_numpy()
    lu.loc[m, "slot"] = lu.loc[m, "slot"].sample(frac=1, random_state=5).to_numpy()
    lu.to_csv(dst / "lineup.csv", index=False)

    sl = pd.read_csv(src / "season_lines.csv")
    m = sl.season >= d.year
    for c in ("pa", "ab", "bf", "h", "d2", "d3", "hr", "bb", "ibb", "hbp", "so", "sf", "go", "ao", "r", "er"):
        sl.loc[m, c] = sl.loc[m, c] * 2 + 3
    sl.to_csv(dst / "season_lines.csv", index=False)

    pf = pd.read_csv(src / "pf.csv")
    pf.loc[pf.year > d.year, "pf_runs"] = 1.5
    pf.to_csv(dst / "pf.csv", index=False)

    gc = pd.read_csv(src / "gc.csv")
    gc.loc[gc.game_id.isin(after), "temp_f"] = 120
    gc.to_csv(dst / "gc.csv", index=False)

    for name in ("players.csv", "rg.csv"):
        shutil.copy(src / name, dst / name)


def run(inputs: Path, d: date, point: str, seasons: list[int]) -> int:
    from .build import TARGET_COLUMNS
    tmp = Path(tempfile.mkdtemp())
    try:
        perturb(inputs, tmp, d, point)
        a, _ = build_features(inputs, seasons, point)
        b, _ = build_features(tmp, seasons, point)
        fa = a.execute(f"select * from features where date <= '{d}'").fetchdf()
        fb = b.execute(f"select * from features where date <= '{d}'").fetchdf()
    finally:
        shutil.rmtree(tmp)
    skip = set(TARGET_COLUMNS) | {"game_id", "bat_team"}
    fa = fa.set_index(["game_id", "bat_team"]).sort_index()
    fb = fb.set_index(["game_id", "bat_team"]).sort_index()
    if len(fa) != len(fb):
        print(f"FAIL: {len(fa)} rows before, {len(fb)} after")
        return 1
    bad = []
    for c in fa.columns:
        if c in skip:
            continue
        x, y = fa[c], fb[c]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            diff = ~((x - y).abs().fillna(0) < 1e-9) | (x.isna() != y.isna())
        else:
            diff = ~((x == y) | (x.isna() & y.isna()))
        if diff.any():
            bad.append((c, int(diff.sum())))
    print(f"leak test, point {point}, D = {d}: {len(fa)} team-games dated on or before D compared")
    if bad:
        print("FAIL: features changed when only the future changed:")
        for c, n in bad:
            print(f"  {c}: {n} rows")
        return 1
    print("PASS: no feature of a game on or before D changed")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--point", default="P2")
    ap.add_argument("--seasons", default="2021,2022")
    a = ap.parse_args()
    d = date.fromisoformat(a.date)
    if d.year < 2022:
        sys.exit("D must be 2022 or later (2021 is the warm-up season used for fitting)")
    sys.exit(run(Path(a.inputs), d, a.point, [int(s) for s in a.seasons.split(",")]))


if __name__ == "__main__":
    main()
