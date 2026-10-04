"""
Future-perturbation leak test (design 5.8, test 1).

Idea: scramble everything that happened on or after a date D, rebuild the
features, and check that no game dated on or before D changed. If a feature
moved, it was reading the future.

    python -m pipelines.features.leak_test --inputs <folder> --date 2022-06-15 --point P2

What gets scrambled, for data dated D or later:
  - every pitch outcome: all plate appearances become home runs, fastball
    velocity +5 mph, all balls in play become ground balls, all pitch types
    four-seamers, every batted ball 115 mph at 28 degrees (contact quality,
    added 26 Sep with round 2 group 1)
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
  - weather (observed and forecast)
(Extended 25 Sep after the independent review found the first version left
ids, innings, scores, starters, venues, season lines and park factors alone.)

(Wind speed and direction are scrambled too, from 1 Oct 2026.)
(Umpire calls, pitch locations, batter zones and counts of D or later, and
plate umpires of later games, from 2 Oct 2026, design E14.)
(Pitch movement, spin, extension and release point of D or later, from
2 Oct 2026, design E16.)
(Catcher ids of later games move to other real catchers, and a variant with
lineups left unchanged, from 3 Oct 2026, design E18.)
(Minor-league lines and team totals of D's season and later, from 4 Oct 2026,
design E19; a variant keeps only players who debuted by D.)
(Dates of games after D move 2 days later, from 3 Oct 2026, design E17; their
venues were already scrambled. The travel measure reads only dates and venues.)

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

LINEUP_MODE = "scrambled"   # "unchanged" = variant A of the E18 leak test
MINOR_DEBUT_CUT = False     # True = the E19 variant: minor lines only for players who debuted by D



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

    # Pitcher ids of later games move to OTHER REAL pitchers (a fixed shift
    # through the sorted list of ids), so their pitches stay in every
    # calculation that joins to players and the other scrambles below are
    # actually scored. (Fixed 3 Oct 2026 after the E16 review found that fake
    # ids under 1000 made the stuff features drop those pitches.)
    pid = sorted(set().union(*[set(con.execute(f"select distinct pitcher_id from read_parquet('{f.as_posix()}') where pitcher_id is not null").fetchdf().pitcher_id)
                               for f in src.glob("pitches_*.parquet")]))
    con.register("pid_map_df", pd.DataFrame({"pitcher_id": pid, "new_pid": pid[7:] + pid[:7]}))
    for f in src.glob("pitches_*.parquet"):
        con.execute(f"""
            copy (
              select * exclude (new_pid) replace (
                case when game_id in (select game_id from late_df) and events is not null then 'home_run' else events end as events,
                case when game_id in (select game_id from late_df) and events is not null then 2.0 else woba_value end as woba_value,
                case when game_id in (select game_id from late_df) and events is not null then 1 else woba_denom end as woba_denom,
                case when game_id in (select game_id from late_df) then release_speed + 5 else release_speed end as release_speed,
                case when game_id in (select game_id from late_df) and bb_type is not null then 'ground_ball' else bb_type end as bb_type,
                case when game_id in (select game_id from late_df) then 'FF' else pitch_type end as pitch_type,
                case when game_id in (select game_id from late_df) then 115.0 else exit_velocity end as exit_velocity,
                case when game_id in (select game_id from late_df) then 28 else launch_angle end as launch_angle,
                case when game_id in (select game_id from late_df) then coalesce(new_pid, pitcher_id) else pitcher_id end as pitcher_id,
                case when game_id in (select game_id from late_df) then batter_id % 991 + 1 else batter_id end as batter_id,
                case when game_id in (select game_id from late_df) then 0 else bat_score end as bat_score,
                case when game_id in (select game_id from late_df) then 9 else fld_score end as fld_score,
                case when game_id in (select game_id from late_df) then 8 else inning end as inning,
                case when game_id in (select game_id from late_df) then pitch_number * 3 else pitch_number end as pitch_number,
                -- umpire calls (design E14): balls become called strikes, swinging
                -- strikes become balls (so the set of taken pitches changes too),
                -- locations and the batter's zone move, every count is 3-2
                case when game_id in (select game_id from late_df) then
                     case when pitch_result in ('ball', 'blocked_ball') then 'called_strike'
                          when pitch_result in ('swinging_strike', 'swinging_strike_blocked') then 'ball'
                          else pitch_result end
                     else pitch_result end as pitch_result,
                case when game_id in (select game_id from late_df) then plate_z + 0.5 else plate_z end as plate_z,
                case when game_id in (select game_id from late_df) then sz_top - 0.1 else sz_top end as sz_top,
                case when game_id in (select game_id from late_df) then 3 else balls end as balls,
                case when game_id in (select game_id from late_df) then 2 else strikes end as strikes,
                -- pitch traits (design E16): movement, spin, extension, release
                case when game_id in (select game_id from late_df) then pfx_x + 0.5 else pfx_x end as pfx_x,
                case when game_id in (select game_id from late_df) then pfx_z + 0.5 else pfx_z end as pfx_z,
                case when game_id in (select game_id from late_df) then spin_rate + 500 else spin_rate end as spin_rate,
                case when game_id in (select game_id from late_df) then release_extension + 1 else release_extension end as release_extension,
                case when game_id in (select game_id from late_df) then release_pos_z + 1 else release_pos_z end as release_pos_z,
                case when game_id in (select game_id from late_df) then release_pos_x + 1 else release_pos_x end as release_pos_x
              )
              from read_parquet('{f.as_posix()}') left join pid_map_df using (pitcher_id)
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
    # travel (design E17): later games also move 2 days later
    g2.loc[m, "date"] = (pd.to_datetime(g2.loc[m, "date"]) + pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")
    # plate umpires (design E14): later games get fake ids; at P1 game D's own
    # umpire too (P1 doesn't know it; at P2 it is a legitimate input)
    if "umpire_id" in g2.columns:
        on_or_after_d = set(games.loc[pd.to_datetime(games["date"]).dt.date >= d, "game_id"])
        mu = g2.game_id.isin(on_or_after_d if point == "P1" else after)
        g2.loc[mu, "umpire_id"] = g2.loc[mu, "umpire_id"] % 997 + 900000
    g2.to_csv(dst / "games.csv", index=False)

    lu = pd.read_csv(src / "lineup.csv")
    on_or_after = set(games.loc[pd.to_datetime(games["date"]).dt.date >= d, "game_id"])
    m = lu.game_id.isin(on_or_after if point == "P1" else after)
    lu.loc[m, "player_id"] = lu.loc[m, "player_id"].sample(frac=1, random_state=7).to_numpy()
    lu.loc[m, "slot"] = lu.loc[m, "slot"].sample(frac=1, random_state=5).to_numpy()
    # catchers (design E18): later games' C rows get OTHER REAL catchers (the
    # plain shuffle above lands mostly on non-catchers with no history, which
    # is a weak test for a catcher-keyed feature). LINEUP_MODE "unchanged" keeps
    # every lineup as it was, so only pitch outcomes move (most sensitive for
    # a pitch-outcome leak)
    mc = m & (lu["pos"] == "C")
    real_c = sorted(lu.loc[lu["pos"] == "C", "player_id"].unique())
    if mc.any():
        idx = {p_: i for i, p_ in enumerate(real_c)}
        lu.loc[mc, "player_id"] = [real_c[(idx[p_] + 17) % len(real_c)] for p_ in lu.loc[mc, "player_id"]]
    if LINEUP_MODE == "unchanged":
        lu = pd.read_csv(src / "lineup.csv")
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
    # wind too (added 1 Oct 2026 with round 2 group 3, design E12)
    gc.loc[gc.game_id.isin(after), "wind_speed"] = 40.0
    gc.loc[gc.game_id.isin(after), "wind_dir"] = (gc.loc[gc.game_id.isin(after), "wind_dir"] + 180) % 360
    gc.to_csv(dst / "gc.csv", index=False)

    fc = pd.read_csv(src / "fc.csv")
    fc.loc[fc.game_id.isin(after), "fc_temp_f"] = 120
    fc.loc[fc.game_id.isin(after), "fc_wind_mph"] = 40.0
    fc.loc[fc.game_id.isin(after), "fc_wind_dir"] = (fc.loc[fc.game_id.isin(after), "fc_wind_dir"] + 180) % 360
    fc.to_csv(dst / "fc.csv", index=False)

    # minor-league lines and team totals (design E19): seasons on or after D's
    # season scrambled (only seasons before a game's season may be used).
    # MINOR_DEBUT_CUT drops the lines of every player who hadn't debuted in
    # MLB by D, so nothing a later call-up did can reach earlier features.
    if (src / "minor_lines.csv").exists():
        ml = pd.read_csv(src / "minor_lines.csv")
        m = ml.season >= d.year
        for c in ("pa", "ab", "bf", "h", "d2", "d3", "hr", "bb", "ibb", "hbp", "so", "sf", "go", "ao"):
            ml.loc[m, c] = ml.loc[m, c] * 2 + 3
        if MINOR_DEBUT_CUT:
            pl = pd.read_csv(src / "players.csv")
            deb = pd.to_datetime(pl.set_index("player_id").debut_date, errors="coerce")
            keep = {p for p, v in deb.items() if pd.notna(v) and v.date() <= d}
            ml = ml[ml.player_id.isin(keep)]
        ml.to_csv(dst / "minor_lines.csv", index=False)
        mt = pd.read_csv(src / "minor_totals.csv")
        m = mt.season >= d.year
        for c in ("pa", "ab", "bf", "h", "d2", "d3", "hr", "bb", "ibb", "hbp", "so", "sf", "go", "ao"):
            mt.loc[m, c] = mt.loc[m, c] * 2 + 3
        mt.to_csv(dst / "minor_totals.csv", index=False)
    for name in ("players.csv", "rg.csv"):
        shutil.copy(src / name, dst / name)
    # 2018-2020 wind (design E13): all before any test date, copied unchanged
    if (src / "wind_hist.csv").exists():
        shutil.copy(src / "wind_hist.csv", dst / "wind_hist.csv")


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
