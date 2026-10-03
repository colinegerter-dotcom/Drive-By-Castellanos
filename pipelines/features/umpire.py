"""
Round 2 group 4: plate umpire (design 5.7 item 4, as built: design E14,
drafted 2 Oct 2026 and revised after an independent review, before the build).

The plate umpire's strike zone changes how many strikeouts and walks happen,
for both teams. An umpire's called zone is far more consistent from one
stretch of games to the next than his strikeout or walk rates (design check
on 2021-2022 in E14), so the feature measures the zone directly:

  ump_cs   extra called strikes per 100 taken pitches, compared with what
           a typical umpire calls at the same location, batter side and
           count; shrunk toward 0 (a typical zone)

Steps:
1. Expected call. Every taken pitch (called strike, ball, blocked ball) with a
   location and the batter's measured zone gets an expected called-strike
   chance from a smoothed grid fitted on FIT_SEASONS (2021) regular-season
   taken pitches and then frozen. One table per batter side (listed bats;
   switch hitters bat opposite the pitcher's arm, as in park_hand.py) and
   count group (two strikes, three balls, full count, other). Cells are
   0.05 ft across the plate by 0.025 of the batter's zone height, over x
   -2.5..2.5 ft and height -1.5..2.5 zones (0 = bottom, 1 = top); pitches
   outside are clipped to the edge cells. Fine layer: Gaussian smoothing over
   1 cell; wide layer: 6 cells, pulled toward the side-and-count mean by
   0.01 pseudo-pitches; the fine layer is pulled toward the wide one by 2
   pseudo-pitches. Kernels are cut at 4 sigma, with zero padding.
2. Residual. Called strike (1 or 0) minus the expected chance, minus the
   league's average residual over the same span: each prior season's own
   league average; for the current season, the league average over days
   before the game's date. A league-wide zone change isn't credited to the
   umpire.
3. Pooling. Over the regular-season games he worked behind the plate: the
   current season's days before the game's date (weight 1) plus up to 3
   prior seasons (weights 0.8, 0.64, 0.48, the hitter weights in
   priors.py), shrunk toward 0: 100 x sum / (weighted taken pitches + K).

Columns written (table ump_feat, one row per key row):
  ump_cs        the measure; 0 when missing
  ump_missing   no plate umpire id, or P1 (the models use ump_cs only)

Prediction points: P2 only (design 5.7); at P1 every row is 0 and missing.
The plate umpire of game D itself is a legitimate P2 input; his and
everyone's calls from game D's own day and later are never used.

The module-level switches below exist for the leak test's mutation checks
(design E14) and must stay at their defaults.
"""
from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd

from .priors import HIT_WEIGHTS

FIT_SEASONS: tuple[int, ...] | None = (2021,)   # None = every loaded season (mutation only)
K_SHRINK = 1000.0          # taken pitches (design E14)
SAME_DAY = False           # True counts the game's own day (mutation only)
LEAGUE_FULL_SEASON = False # True uses the current season's full-season league average (mutation only)

X_LO, X_PER_FT, NX = -2.5, 20, 100       # 0.05 ft cells
H_LO, H_PER_ZONE, NH = -1.5, 40, 160     # 0.025 zone-height cells
SIGMA_FINE, SIGMA_WIDE = 1.0, 6.0
PULL_WIDE, PULL_FINE = 0.01, 2.0


def _kernel(sigma: float) -> np.ndarray:
    r = int(np.ceil(4 * sigma))
    t = np.arange(-r, r + 1)
    w = np.exp(-0.5 * (t / sigma) ** 2)
    return w / w.sum()


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing, zero padding at the edges."""
    k = _kernel(sigma)
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def build_taken(con: duckdb.DuckDBPyConnection) -> None:
    """Table ump_taken: every regular-season taken pitch with its cell, side,
    count group and plate umpire (null if unknown)."""
    con.execute(f"""
        create or replace table ump_taken as
        with t as (
            select p.game_id, p.season, p.data_date, g.umpire_id,
                   -- sd = 1 for a left-handed batter. A switch hitter bats
                   -- opposite the pitcher's arm: right-handed against a lefty
                   case when coalesce(b.bats, 'R') = 'S'
                        then case when coalesce(pt.throws, 'R') = 'L' then 0 else 1 end
                        else case when coalesce(b.bats, 'R') = 'L' then 1 else 0 end end as sd,
                   case when p.balls = 3 and p.strikes = 2 then 3
                        when p.strikes = 2 then 1
                        when p.balls = 3 then 2 else 0 end as cg,
                   p.plate_x, (p.plate_z - p.sz_bot) / (p.sz_top - p.sz_bot) as h,
                   (p.pitch_result = 'called_strike')::int as cs
            from pitches p
            join games g using (game_id)
            left join players b on b.player_id = p.batter_id
            left join players pt on pt.player_id = p.pitcher_id
            where p.game_type = 'R'
              and p.pitch_result in ('called_strike', 'ball', 'blocked_ball')
              and p.plate_x is not null and p.plate_z is not null
              and p.sz_top is not null and p.sz_bot is not null and p.sz_top > p.sz_bot
        )
        select game_id, season, data_date, umpire_id, sd, cg, cs,
               least(greatest(floor((plate_x - ({X_LO})) * {X_PER_FT})::int, 0), {NX - 1}) as ix,
               least(greatest(floor((h - ({H_LO})) * {H_PER_ZONE})::int, 0), {NH - 1}) as iz
        from t
    """)


def fit_grid(con: duckdb.DuckDBPyConnection) -> dict:
    """Table ump_grid (sd, cg, ix, iz, p_cs): the frozen expected-call grid."""
    where = "" if FIT_SEASONS is None else f"where season in ({','.join(map(str, FIT_SEASONS))})"
    cells = con.execute(f"""
        select sd, cg, ix, iz, count(*) as n, sum(cs) as s
        from ump_taken {where} group by all
    """).fetchdf()
    if cells.empty:
        raise RuntimeError("no taken pitches to fit the umpire grid on")
    out = []
    for (sd, cg), q in cells.groupby(["sd", "cg"]):
        n = np.zeros((NX, NH))
        s = np.zeros((NX, NH))
        n[q.ix.to_numpy(), q.iz.to_numpy()] = q.n.to_numpy()
        s[q.ix.to_numpy(), q.iz.to_numpy()] = q.s.to_numpy()
        m = s.sum() / n.sum()
        wn, ws = _blur(n, SIGMA_WIDE), _blur(s, SIGMA_WIDE)
        p_wide = (ws + PULL_WIDE * m) / (wn + PULL_WIDE)
        fn, fs = _blur(n, SIGMA_FINE), _blur(s, SIGMA_FINE)
        p = (fs + PULL_FINE * p_wide) / (fn + PULL_FINE)
        ix, iz = np.meshgrid(np.arange(NX), np.arange(NH), indexing="ij")
        out.append(pd.DataFrame({"sd": int(sd), "cg": int(cg), "ix": ix.ravel(), "iz": iz.ravel(), "p_cs": p.ravel()}))
    grid = pd.concat(out, ignore_index=True)
    con.register("ump_grid_df", grid)
    con.execute("create or replace table ump_grid as select * from ump_grid_df")
    con.unregister("ump_grid_df")
    return {"fit_seasons": list(FIT_SEASONS) if FIT_SEASONS else "all", "fit_pitches": int(cells.n.sum()),
            "groups": int(grid[["sd", "cg"]].drop_duplicates().shape[0])}


def umpire_features(con: duckdb.DuckDBPyConnection, keys: str = "tg", point: str = "P2") -> dict:
    """Writes table ump_feat (game_id, bat_team, ump_cs, ump_missing)."""
    if point != "P2":
        con.execute(f"""
            create or replace table ump_feat as
            select game_id, bat_team, 0.0::double as ump_cs, true as ump_missing from {keys}
        """)
        return {"point": point, "note": "P1: umpire features are 0 by design (E14)"}

    build_taken(con)
    notes = fit_grid(con)
    # residual per pitch, totals per game and per league-day
    con.execute("""
        create or replace table ump_game as
        select t.game_id, t.season, t.data_date, t.umpire_id,
               sum(t.cs - gr.p_cs) as r, count(*) as n
        from ump_taken t
        join ump_grid gr using (sd, cg, ix, iz)
        group by all
    """)
    con.execute("""
        create or replace table ump_lg_day as
        select season, data_date, sum(r) as r, sum(n) as n from ump_game group by all
    """)
    con.execute("""
        create or replace table ump_lg_season as
        select season, sum(r) / sum(n) as l from ump_game group by all
    """)
    # each umpire's prior full seasons, centred on that season's league average
    con.execute("""
        create or replace table ump_season as
        select u.umpire_id, u.season, sum(u.r) - sum(u.n) * any_value(l.l) as rc, sum(u.n) as n
        from ump_game u join ump_lg_season l using (season)
        where u.umpire_id is not null
        group by all
    """)
    day_cmp = "<=" if SAME_DAY else "<"
    w1, w2, w3 = HIT_WEIGHTS[1], HIT_WEIGHTS[2], HIT_WEIGHTS[3]
    lg_cur = ("(select l from ump_lg_season s where s.season = k.season)" if LEAGUE_FULL_SEASON else
              "(select sum(d.r) / nullif(sum(d.n), 0) from ump_lg_day d "
              f"where d.season = k.season and d.data_date {day_cmp} k.date)")
    con.execute(f"""
        create or replace table ump_feat_game as
        with k as (
            select distinct q.game_id, q.season, q.date, g.umpire_id
            from {keys} q join games g using (game_id)
        ),
        cur as (
            select k.game_id, sum(u.r) as r, sum(u.n) as n
            from k join ump_game u
              on u.umpire_id = k.umpire_id and u.season = k.season and u.data_date {day_cmp} k.date
            group by k.game_id
        ),
        pri as (
            select k.game_id,
                   sum(case k.season - s.season when 1 then {w1} when 2 then {w2} else {w3} end * s.rc) as rc,
                   sum(case k.season - s.season when 1 then {w1} when 2 then {w2} else {w3} end * s.n) as n
            from k join ump_season s
              on s.umpire_id = k.umpire_id and s.season between k.season - 3 and k.season - 1
            group by k.game_id
        ),
        j as (
            select k.game_id, k.umpire_id,
                   coalesce(cur.r, 0) - coalesce(cur.n, 0) * coalesce({lg_cur}, 0) as cur_rc,
                   coalesce(cur.n, 0) as cur_n,
                   coalesce(pri.rc, 0) as pri_rc, coalesce(pri.n, 0) as pri_n
            from k left join cur using (game_id) left join pri using (game_id)
        )
        select game_id, umpire_id, cur_n, pri_n,
               (case when umpire_id is null then 0.0
                     else 100.0 * (cur_rc + pri_rc) / (cur_n + pri_n + {K_SHRINK}) end)::double as ump_cs,
               (umpire_id is null) as ump_missing
        from j
    """)
    con.execute(f"""
        create or replace table ump_feat as
        select q.game_id, q.bat_team, coalesce(f.ump_cs, 0.0)::double as ump_cs, coalesce(f.ump_missing, true) as ump_missing
        from {keys} q left join ump_feat_game f using (game_id)
    """)
    notes.update(con.execute("""
        select count(*) as games, avg(ump_cs) as mean, stddev(ump_cs) as sd,
               sum(ump_missing::int) as missing
        from ump_feat_game
    """).fetchdf().iloc[0].to_dict())
    # loud, not silent: a game with no plate umpire gets ump_cs = 0. In
    # 2021-2024 only the 2 cancelled games have none; live, it means the
    # announced umpire wasn't captured (design E14, before live use)
    if notes["missing"]:
        ids = con.execute("select game_id from ump_feat_game where ump_missing order by 1 limit 10").fetchdf().game_id.tolist()
        print(f"warning: {int(notes['missing'])} game(s) have no plate umpire; ump_cs set to 0 (first ids: {ids})",
              file=sys.stderr)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in notes.items()}
