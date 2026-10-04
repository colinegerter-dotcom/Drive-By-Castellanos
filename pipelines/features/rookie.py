"""
Round 2 group 8: rookie translations (design E19, drafted 4 Oct 2026 and
revised after an independent pre-build review, before the build).

A player with no MLB season before the current one starts the prior layer
from the league average (hitters' wOBA minus 0.020; pitchers carry a separate
rookie penalty in pitching.py and the sp_rookie flag in M1). This module gives
such players a starting value from their Triple-A and Double-A numbers,
WITHOUT moving the rookie average: it only ranks rookies against each other.

For each rookie (player p, season S) and rate (hitters: strikeout, walk,
wOBA; pitchers: strikeout, walk, ground-ball share):

  1. Prior minor-league seasons only (a season line includes games after the
     game being predicted): up to 3 seasons before S, skipping 2020 (no
     minors), weights 0.8, 0.64, 0.48.
  2. Per level L (11 Triple-A, 12 Double-A): the player's rate minus the
     level-season whole-league rate (team totals, Mexican League teams
     excluded), weighted over the seasons: x_L, with chances n_L.
  3. Shrunk: xs_L = x_L * n_L / (n_L + k), k from the prior layer.
  4. adj = sum_L n_L * (b_L * xs_L - c_L) / sum_L n_L, where b_L is fitted
     on the FIT_SEASONS rookies and c_L is the mean of b_L * xs_L over them
     (so the rookie average doesn't move).
  5. Scales: wOBA terms times woba_scale and ground-ball terms times gb_b
     (priors.lines_map), since the fit is on official-line scales and the
     prior layer is on the Statcast scale.

The prior mean for a rookie becomes today's default + adj. Players with
MLB history, and rookies with no minor line, are unchanged.

2019 minor-league ground outs: MLB's own data has about half the usual ground
outs per batted-ball out at both levels in 2019 (team totals 0.25 and 0.27
against about 0.50 in other seasons). As written in E19 before fitting, 2019
relative ground-ball rates are divided by the ratio of their spread to the
2018 spread.

Module switches for the leak test's mutation checks (E19) must stay at their
defaults: CURRENT_SEASON_LINE, POOL_LEAGUE, REFIT_ALL.
"""
from __future__ import annotations

import sys

import duckdb
import numpy as np

FIT_SEASONS = (2018, 2019, 2021)
MIN_MLB_CHANCES = 30
WEIGHTS = (0.8, 0.64, 0.48)
LEVELS = (11, 12)
# Mexican League teams, which MLB's API files under Triple-A in 2017 and 2019
MEXICAN_LEAGUE_TEAM_IDS = (434, 442, 447, 496, 502, 520, 523, 528, 532, 536, 560, 562, 569, 579, 4444, 5010)
# stat -> (group, k for shrinkage, chances column)
STATS = {
    "hk": ("h", 60.0, "n_pa"), "hbb": ("h", 120.0, "n_pa"), "hwoba": ("h", 300.0, "n_wd"),
    "pk": ("p", 70.0, "n_bf"), "pbb": ("p", 170.0, "n_bf"), "pgb": ("p", 70.0, "n_gb"),
}

# Frozen b and c per (stat, level), from fit_translation() on FIT_SEASONS
# (4 Oct 2026). build_rookie_adj() refits and stops if they no longer match.
FROZEN: dict = {
    ('hbb', 11): {"b": 0.7539660452152455, "c": -0.0003810119790388108, "n_rookies": 124},
    ('hbb', 12): {"b": 0.792788454808803, "c": 0.0028066928969708413, "n_rookies": 195},
    ('hk', 11): {"b": 0.9230597076520914, "c": -0.004445903536490132, "n_rookies": 124},
    ('hk', 12): {"b": 0.7466871321764198, "c": -0.010833882056301038, "n_rookies": 195},
    ('hwoba', 11): {"b": 1.0628454940336742, "c": 0.006339183003835359, "n_rookies": 124},
    ('hwoba', 12): {"b": 0.8413475022214546, "c": 0.010846663360996305, "n_rookies": 195},
    ('pbb', 11): {"b": 1.0334413809322194, "c": -0.003960743628766128, "n_rookies": 146},
    ('pbb', 12): {"b": 0.8622382334340644, "c": -0.0033886404595833805, "n_rookies": 285},
    ('pgb', 11): {"b": 1.258532852294149, "c": 0.010589015539755772, "n_rookies": 145},
    ('pgb', 12): {"b": 1.054143503188947, "c": 0.0025786858247591987, "n_rookies": 285},
    ('pk', 11): {"b": 0.7672657868540801, "c": 0.0033447353664958638, "n_rookies": 146},
    ('pk', 12): {"b": 0.517652321932553, "c": 0.007444141418917052, "n_rookies": 285},
}

# NOT ADOPTED (4 Oct 2026): the pre-registered rule failed its second gate,
# so the translation is off unless switched on (to reproduce E19).
ENABLED = False

GB_2019_RATIO = {11: 0.9148507104302912, 12: 0.8545415873072335}

CURRENT_SEASON_LINE = False   # mutation only: also use season S's own minor line
POOL_LEAGUE = False           # mutation only: level averages pooled over every season
REFIT_ALL = False             # mutation only: fit b and c on every season with MLB lines


def _prior_seasons(S: int) -> list[tuple[int, float]]:
    seasons = [s for s in range(S - 1, S - 6, -1) if s != 2020][:3]
    out = list(zip(seasons, WEIGHTS))
    if CURRENT_SEASON_LINE:
        out = [(S, 1.0)] + out
    return out


def _have_inputs(con) -> bool:
    names = {r[0] for r in con.execute("select table_name from information_schema.tables").fetchall()}
    return {"minor_lines", "minor_totals"} <= names


def build_minor_rel(con: duckdb.DuckDBPyConnection) -> None:
    """Table minor_rel: per player, season, level and stat, the relative rate
    and its chances."""
    excl = ",".join(map(str, MEXICAN_LEAGUE_TEAM_IDS))
    lg_keys = "s, g" if POOL_LEAGUE else "season, s, g"
    con.execute(f"""
        create or replace table minor_lg as
        select {lg_keys},
               sum(so)::double / nullif(sum(pa), 0) as hk, sum(bb - ibb)::double / nullif(sum(pa), 0) as hbb,
               sum(0.69 * (bb - ibb) + 0.72 * hbp + 0.88 * (h - d2 - d3 - hr) + 1.25 * d2 + 1.58 * d3 + 2.03 * hr)
                 / nullif(sum(ab + bb - ibb + sf + hbp), 0) as hwoba,
               sum(so)::double / nullif(sum(bf), 0) as pk, sum(bb - ibb)::double / nullif(sum(bf), 0) as pbb,
               sum(go)::double / nullif(sum(go + ao), 0) as pgb
        from minor_totals where team_id not in ({excl})
        group by {lg_keys}
    """)
    join = "using (s, g)" if POOL_LEAGUE else "using (season, s, g)"
    con.execute(f"""
        create or replace table minor_rel0 as
        with r as (
            select player_id, season, s, g,
                   pa as n_pa, ab + bb - ibb + sf + hbp as n_wd, bf as n_bf, go + ao as n_gb,
                   so::double / nullif(pa, 0) as hk, (bb - ibb)::double / nullif(pa, 0) as hbb,
                   (0.69 * (bb - ibb) + 0.72 * hbp + 0.88 * (h - d2 - d3 - hr) + 1.25 * d2 + 1.58 * d3 + 2.03 * hr)
                     / nullif(ab + bb - ibb + sf + hbp, 0) as hwoba,
                   so::double / nullif(bf, 0) as pk, (bb - ibb)::double / nullif(bf, 0) as pbb,
                   go::double / nullif(go + ao, 0) as pgb
            from minor_lines
        )
        select r.player_id, r.season, r.s, r.g, r.n_pa, r.n_wd, r.n_bf, r.n_gb,
               r.hk - l.hk as hk, r.hbb - l.hbb as hbb, r.hwoba - l.hwoba as hwoba,
               r.pk - l.pk as pk, r.pbb - l.pbb as pbb, r.pgb - l.pgb as pgb
        from r join (select * from minor_lg) l {join}
    """)
    # 2019 ground-out rescale, frozen: the spread of relative ground-out share
    # among pitchers with 100+ batted-ball outs, 2018 over 2019, per level, as
    # computed on 4 Oct 2026 by gb_2019_ratio(). Frozen because the spread
    # depends on which players are in the table, and the table holds players
    # who reached MLB later (the debut-cut leak test found this).
    ratio = dict(GB_2019_RATIO)
    con.execute(f"""
        create or replace table minor_rel as
        select * replace (case when season = 2019 then pgb * (case s when 11 then {ratio[11]} else {ratio[12]} end)
                               else pgb end as pgb)
        from minor_rel0
    """)
    return ratio


def gb_2019_ratio(con: duckdb.DuckDBPyConnection) -> dict:
    """How GB_2019_RATIO was computed (needs table minor_rel0)."""
    ratio = {}
    for lev in LEVELS:
        sd = dict(con.execute(f"""
            select season, stddev_samp(pgb) from minor_rel0
            where g = 'p' and s = {lev} and n_gb >= 100 and season in (2018, 2019) group by 1
        """).fetchall())
        ratio[lev] = sd[2018] / sd[2019]
    return ratio


def _xs_table(con, targets: str) -> None:
    """Table rk_xs (player_id, ts, stat, s, n, xs) for every (player_id, ts) in
    `targets`: weighted relative rate per level, shrunk."""
    pairs = sorted({S for (S,) in con.execute(f"select distinct ts from {targets}").fetchall()})
    rows = [(S, s, w) for S in pairs for s, w in _prior_seasons(S)]
    con.execute("create or replace temp table rk_w (ts integer, season integer, w double)")
    if rows:
        con.executemany("insert into rk_w values (?, ?, ?)", rows)
    parts = []
    for stat, (g, k, ncol) in STATS.items():
        parts.append(f"""
            select t.player_id, t.ts, '{stat}' as stat, m.s,
                   sum(w.w * m.{ncol}) as n,
                   sum(w.w * m.{ncol} * m.{stat}) / nullif(sum(w.w * m.{ncol}), 0) as x
            from {targets} t
            join rk_w w on w.ts = t.ts
            join minor_rel m on m.player_id = t.player_id and m.season = w.season and m.g = '{g}'
            where m.{ncol} > 0 and m.{stat} is not null and t.g = '{g}'
            group by 1, 2, 3, 4""")
    k_case = " ".join(f"when '{st}' then {v[1]}" for st, v in STATS.items())
    con.execute(f"""
        create or replace table rk_xs as
        select player_id, ts, stat, s, n, x * n / (n + case stat {k_case} end) as xs
        from ({' union all '.join(parts)}) q
        where n > 0
    """)


def fit_translation(con: duckdb.DuckDBPyConnection, seasons) -> dict:
    """b and c per (stat, level) from rookie-seasons in `seasons`: WLS of the
    first MLB season's rate (minus the previous season's MLB league rate) on
    xs, weighted by MLB chances; c = mean of b * xs over the same rookies."""
    seasons = tuple(seasons)
    con.execute(f"""
        create or replace table rk_targets as
        with sl as (
            select player_id, season, grp,
                   case grp when 'hitting' then pa else bf end as n,
                   case grp when 'hitting' then so::double / nullif(pa, 0) else so::double / nullif(bf, 0) end as k,
                   case grp when 'hitting' then (bb - ibb)::double / nullif(pa, 0) else (bb - ibb)::double / nullif(bf, 0) end as bb,
                   (0.69 * (bb - ibb) + 0.72 * hbp + 0.88 * (h - d2 - d3 - hr) + 1.25 * d2 + 1.58 * d3 + 2.03 * hr)
                     / nullif(ab + bb - ibb + sf + hbp, 0) as woba,
                   go::double / nullif(go + ao, 0) as gb
            from season_lines
        ),
        lg as (
            select l.season, l.grp,
                   sum(so)::double / nullif(sum(case grp when 'hitting' then pa else bf end), 0) as k,
                   sum(bb - ibb)::double / nullif(sum(case grp when 'hitting' then pa else bf end), 0) as bb,
                   sum(0.69 * (bb - ibb) + 0.72 * hbp + 0.88 * (h - d2 - d3 - hr) + 1.25 * d2 + 1.58 * d3 + 2.03 * hr)
                     / nullif(sum(ab + bb - ibb + sf + hbp), 0) as woba,
                   sum(go)::double / nullif(sum(go + ao), 0) as gb
            from season_lines l group by 1, 2
        )
        select sl.player_id, sl.season as ts, case sl.grp when 'hitting' then 'h' else 'p' end as g, sl.n,
               sl.k - lg.k as y_k, sl.bb - lg.bb as y_bb, sl.woba - lg.woba as y_woba, sl.gb - lg.gb as y_gb
        from sl join lg on lg.season = sl.season - 1 and lg.grp = sl.grp
        where sl.season in ({','.join(map(str, seasons))}) and sl.n >= {MIN_MLB_CHANCES}
          and not exists (select 1 from season_lines e where e.player_id = sl.player_id and e.grp = sl.grp and e.season < sl.season)
    """)
    _xs_table(con, "rk_targets")
    df = con.execute("""
        select x.stat, x.s, x.xs, t.n,
               case x.stat when 'hk' then y_k when 'pk' then y_k when 'hbb' then y_bb when 'pbb' then y_bb
                           when 'hwoba' then y_woba else y_gb end as y
        from rk_xs x join rk_targets t on t.player_id = x.player_id and t.ts = x.ts
    """).fetchdf().dropna()
    out = {}
    for (stat, lev), q in df.groupby(["stat", "s"]):
        X = np.column_stack([np.ones(len(q)), q["xs"].to_numpy()])
        w = q["n"].to_numpy(dtype=float)
        beta = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (q["y"].to_numpy() * w))
        b = float(beta[1])
        c = float(np.average(b * q["xs"].to_numpy(), weights=w))
        out[(stat, int(lev))] = {"b": b, "c": c, "n_rookies": int(len(q))}
    return out


def build_rookie_adj(con: duckdb.DuckDBPyConnection) -> dict:
    """Table rookie_adj (player_id, season, g, adj_k, adj_bb, adj_woba,
    adj_gb, n_minor): the change to a rookie's prior mean, already on the
    prior layer's scale. Empty when the minor-league inputs are absent."""
    empty = """create or replace table rookie_adj as
               select null::bigint player_id, null::int season, null::varchar g, null::double adj_k,
                      null::double adj_bb, null::double adj_woba, null::double adj_gb, null::double n_minor where false"""
    if not ENABLED:
        con.execute(empty)
        return {"note": "rookie translation off (E19 not adopted); rookie priors unchanged"}
    if not _have_inputs(con):
        con.execute(empty)
        return {"note": "no minor-league inputs; rookie priors unchanged"}
    ratio = build_minor_rel(con)
    mutated = CURRENT_SEASON_LINE or POOL_LEAGUE or REFIT_ALL
    if REFIT_ALL:
        seasons = [s for (s,) in con.execute("select distinct season from season_lines where season >= 2018 order by 1").fetchall()]
        coef = fit_translation(con, seasons)
    else:
        coef = fit_translation(con, FIT_SEASONS)
        if FROZEN and not mutated:
            bad = [(k, v["b"], FROZEN[k]["b"]) for k, v in coef.items()
                   if k not in FROZEN or abs(v["b"] - FROZEN[k]["b"]) > 1e-9 or abs(v["c"] - FROZEN[k]["c"]) > 1e-9]
            if bad or set(FROZEN) != set(coef):
                raise RuntimeError(f"rookie translation refit differs from the frozen constants: {bad[:3]}")
    # every (player, season) the model can ask about: seasons with games
    con.execute("""
        create or replace table rk_keys as
        select distinct m.player_id, gs.season as ts, m.g
        from minor_rel m join (select distinct season from games) gs on m.season < gs.season
    """)
    if CURRENT_SEASON_LINE:
        con.execute("""
            create or replace table rk_keys as
            select distinct m.player_id, gs.season as ts, m.g
            from minor_rel m join (select distinct season from games) gs on m.season <= gs.season
        """)
    _xs_table(con, "rk_keys")
    con.execute("create or replace temp table rk_coef (stat varchar, s integer, b double, c double)")
    con.executemany("insert into rk_coef values (?, ?, ?, ?)", [(k[0], k[1], v["b"], v["c"]) for k, v in coef.items()])
    scale = con.execute("select woba_scale, gb_b from lines_map").fetchone()
    con.execute(f"""
        create or replace table rk_adj_long as
        select x.player_id, x.ts as season, x.stat,
               sum(x.n * (c.b * x.xs - c.c)) / sum(x.n)
                 * case x.stat when 'hwoba' then {scale[0]} when 'pgb' then {scale[1]} else 1.0 end as adj,
               sum(x.n) as n
        from rk_xs x join rk_coef c on c.stat = x.stat and c.s = x.s
        group by 1, 2, 3
    """)
    con.execute("""
        create or replace table rookie_adj as
        select player_id, season, left(stat, 1) as g,
               max(case when stat in ('hk', 'pk') then adj end) as adj_k,
               max(case when stat in ('hbb', 'pbb') then adj end) as adj_bb,
               max(case when stat = 'hwoba' then adj end) as adj_woba,
               max(case when stat = 'pgb' then adj end) as adj_gb,
               max(n) as n_minor
        from rk_adj_long group by 1, 2, 3
    """)
    notes = {"gb_2019_rescale": ratio, "coef": {f"{k[0]}_{k[1]}": v for k, v in coef.items()}}
    notes.update(con.execute("select count(*) as rows, count(distinct player_id) as players from rookie_adj").fetchdf().iloc[0].to_dict())
    return {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in notes.items()}
