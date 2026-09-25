"""
Pitching features: the opposing starter, the bullpen, and one composite.

Everything is in one unit, expected runs allowed per batter faced, so the
starter and bullpen can be mixed by how many batters each is expected to
face in the segment (first five innings, or innings 1-8).

Two small fitted pieces, both fitted on 2021 only. 2021 is a warm-up season
(it is never trained on, tuned on or tested on), so fitting here can't leak
into any season the model is judged on:
  1. SKILL: runs allowed per batter ~ strikeout, walk and ground-ball rates
     (a SIERA-style composite; ERA is never used)
  2. EXPECTED BATTERS FACED: how deep a starter goes ~ his usual length,
     his last pitch count, days of rest and his team's hook
"""
from __future__ import annotations

import duckdb
import numpy as np

from . import priors

FIT_SEASON = 2021
# Extra wOBA for hitters facing a starter the third time through the order
# (design 5.5: about .024), turned into runs per batter with the usual wOBA
# scale of about 1.2.
TTO3_RUNS_PER_PA = 0.024 / 1.2
# Rookie pitchers allowed about .012 more wOBA (design 5.4), in runs.
ROOKIE_PIT_RUNS = 0.012 / 1.2
REST_BUCKETS = ["r4", "r5_6", "r7_10", "r11_20", "r21p", "first"]


def fit_skill(con: duckdb.DuckDBPyConnection) -> dict:
    """Runs allowed per batter faced from K, BB and GB rates, 2021 pitchers
    with 50+ batters faced, weighted by batters faced. Runs come from the
    official 2021 lines; rates from the pitch files."""
    df = con.execute(f"""
        with s as (
            select player_id, sum(bf) bf, sum(k)::double / sum(bf) k, sum(bb)::double / sum(bf) bb,
                   sum(gb)::double / nullif(sum(bip), 0) gb
            from pit_game where season = {FIT_SEASON} group by 1 having sum(bf) >= 50
        )
        select s.*, l.r::double / l.bf as rpb
        from s join season_lines l on l.player_id = s.player_id and l.season = {FIT_SEASON} and l.grp = 'pitching'
        where l.bf > 0 and s.gb is not null
    """).fetchdf()
    X = np.column_stack([np.ones(len(df)), df.k, df.bb, df.gb])
    w = np.sqrt(df.bf.to_numpy())
    beta, *_ = np.linalg.lstsq(X * w[:, None], df.rpb.to_numpy() * w, rcond=None)
    coef = dict(zip(["b0", "b_k", "b_bb", "b_gb"], map(float, beta)))
    coef["n_pitchers"] = int(len(df))
    return coef


def skill_sql(c: dict, k: str, bb: str, gb: str) -> str:
    return f"({c['b0']} + {c['b_k']} * {k} + {c['b_bb']} * {bb} + {c['b_gb']} * {gb})"


def starter_history(con: duckdb.DuckDBPyConnection, keys: str) -> None:
    """For each (pitcher_id, season, cutoff) in `keys`: starts and batters
    faced so far this season and last season, the last appearance before the
    cutoff (date, role, pitch count), and this season's start share."""
    con.execute("""
        create or replace table start_cum as
        select pitcher_id as player_id, season, data_date,
               sum(count(*)) over w as n_starts, sum(sum(bf)) over w as bf_starts
        from app where started and game_type = 'R'
        group by 1, 2, 3
        window w as (partition by pitcher_id, season order by data_date rows unbounded preceding)
    """)
    con.execute("""
        create or replace table app_cum as
        select pitcher_id as player_id, season, data_date,
               sum(count(*)) over w as n_apps, sum(sum(started::int)) over w as n_starts_all
        from app where game_type = 'R'
        group by 1, 2, 3
        window w as (partition by pitcher_id, season order by data_date rows unbounded preceding)
    """)
    con.execute("""
        create or replace table start_season as
        select pitcher_id as player_id, season, count(*) filter (where started) n_starts,
               sum(bf) filter (where started) bf_starts, count(*) n_apps
        from app where game_type = 'R' group by 1, 2
    """)
    con.execute("""
        -- Last appearance WITHIN the same season. A first start of the season
        -- has no last appearance (so it falls in the "first" rest bucket
        -- and "no last pitch count"), exactly as in the 2021 fit. Using last
        -- October's appearance here made first starts look like normal
        -- starts and underestimated them by about 3 batters (review, 25 Sep).
        create or replace table app_last as
        select pitcher_id as player_id, season, data_date, arg_max(pitches, game_id) pitches, bool_or(started) started
        from app group by 1, 2, 3
    """)
    con.execute(f"""
        create or replace table sp_hist as
        with q as (select distinct player_id, season, cutoff from {keys}),
        a as (select q.*, c.n_starts, c.bf_starts from q asof left join start_cum c
              on c.player_id = q.player_id and c.season = q.season and q.cutoff > c.data_date),
        b as (select q.player_id, q.season, q.cutoff, c.n_apps, c.n_starts_all from q asof left join app_cum c
              on c.player_id = q.player_id and c.season = q.season and q.cutoff > c.data_date),
        l as (select q.player_id, q.season, q.cutoff, c.data_date last_date, c.pitches last_pitches, c.started last_started
              from q asof left join app_last c on c.player_id = q.player_id and c.season = q.season and q.cutoff > c.data_date)
        select a.player_id, a.season, a.cutoff,
               coalesce(a.n_starts, 0) n_starts, coalesce(a.bf_starts, 0) bf_starts,
               coalesce(b.n_apps, 0) n_apps, coalesce(b.n_starts_all, 0) n_starts_all,
               p.n_starts p_starts, p.bf_starts p_bf_starts, p.n_apps p_apps,
               l.last_date, l.last_pitches, l.last_started,
               date_diff('day', l.last_date, a.cutoff) as days_since_last
        from a join b using (player_id, season, cutoff) join l using (player_id, season, cutoff)
        left join start_season p on p.player_id = a.player_id and p.season = a.season - 1
    """)


def team_hook(con: duckdb.DuckDBPyConnection, keys: str) -> None:
    """Average batters faced by a team's starters over the 30 days before
    the cutoff, as a difference from the league's previous-season average."""
    con.execute(f"""
        create or replace table hook as
        with q as (select distinct team_id, season, cutoff from {keys}),
        lg as (select season + 1 as season, avg(bf) lg_bf from app where started and game_type = 'R' group by season)
        select q.team_id, q.season, q.cutoff,
               avg(a.bf) - coalesce(any_value(lg.lg_bf), 22.0) as hook, count(a.bf) as hook_n
        from q
        left join app a on a.team_id = q.team_id and a.started and a.game_type = 'R'
             and a.data_date < q.cutoff and a.data_date >= q.cutoff - interval 30 day
        left join lg on lg.season = q.season
        group by 1, 2, 3
    """)


def rest_bucket_sql(days: str, first: str) -> str:
    return (f"case when {first} then 'first' when {days} <= 4 then 'r4' when {days} <= 6 then 'r5_6' "
            f"when {days} <= 10 then 'r7_10' when {days} <= 20 then 'r11_20' else 'r21p' end")


def fit_exp_bf(con: duckdb.DuckDBPyConnection) -> dict:
    """Expected batters faced for a start, fitted on 2021 regular-season
    starts: usual length + last pitch count + rest bucket + team hook."""
    con.execute(f"""
        create or replace table fit_starts as
        select a.game_id, a.pitcher_id as player_id, a.season, a.date as cutoff, a.team_id, a.bf
        from app a where a.started and a.game_type = 'R' and a.season = {FIT_SEASON}
    """)
    starter_history(con, "fit_starts")
    team_hook(con, "fit_starts")
    df = con.execute(exp_bf_inputs_sql("fit_starts", "player_id", "team_id") + " , f.bf as y from fit_starts f"
                     " join sp_hist h on h.player_id = f.player_id and h.season = f.season and h.cutoff = f.cutoff"
                     " left join hook k on k.team_id = f.team_id and k.season = f.season and k.cutoff = f.cutoff").fetchdf()
    X, cols = exp_bf_design(df)
    beta, *_ = np.linalg.lstsq(X, df.y.to_numpy(float), rcond=None)
    coef = dict(zip(cols, map(float, beta)))
    resid = df.y.to_numpy(float) - X @ beta
    coef["_rmse"] = float(np.sqrt(np.mean(resid ** 2)))
    coef["_n"] = int(len(df))
    return coef


def exp_bf_inputs_sql(tbl: str, pid: str, tid: str) -> str:
    """Select list shared by fitting and applying the expected-BF model."""
    return f"""
        select
          -- usual length: this season's starts, plus last season's at half
          -- weight, plus 5 league-average starts (22 batters) as the prior
          (coalesce(h.bf_starts, 0) + 0.5 * coalesce(h.p_bf_starts, 0) + 5 * 22.0)
            / (coalesce(h.n_starts, 0) + 0.5 * coalesce(h.p_starts, 0) + 5) as usual_bf,
          coalesce(h.last_pitches, 85) as last_pitches,
          (h.last_pitches is null) as no_last,
          {rest_bucket_sql('h.days_since_last', 'h.n_apps = 0')} as rest_bucket,
          coalesce(k.hook, 0) as hook
    """


def exp_bf_design(df):
    cols = ["const", "usual_bf", "last_pitches", "no_last", "hook"] + [f"rest_{b}" for b in REST_BUCKETS[1:]]
    X = np.column_stack(
        [np.ones(len(df)), df.usual_bf, df.last_pitches, df.no_last.astype(float), df.hook]
        + [(df.rest_bucket == b).astype(float) for b in REST_BUCKETS[1:]]
    )
    return X, cols


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def _norm_cdf(z):
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(z, dtype=float) / np.sqrt(2)))


def expected_excess(mu, sd: float, t: float):
    """E[max(X - t, 0)] for X ~ Normal(mu, sd): expected batters beyond t."""
    mu = np.asarray(mu, dtype=float)
    z = (t - mu) / sd
    return sd * _norm_pdf(z) + (mu - t) * (1 - _norm_cdf(z))


def expected_capped(mu, sd: float, cap: float):
    """E[min(X, cap)]: expected batters the starter faces within a segment."""
    return np.asarray(mu, dtype=float) - expected_excess(mu, sd, cap)


def apply_exp_bf(df, coef: dict):
    X, cols = exp_bf_design(df)
    beta = np.array([coef[c] for c in cols])
    return np.clip(X @ beta, 3.0, 30.0)


def bullpen(con: duckdb.DuckDBPyConnection, keys: str, skill: dict) -> None:
    """Bullpen skill per (team_id, season, cutoff) in `keys`, two weightings.

    Pool: every pitcher who relieved for the team in the 30 days before the
    cutoff. Unavailable (phase A rule, measured on 2022-2025 usage): pitched
    on both of the last two days, or threw 25+ pitches yesterday.
      pen_skill_8:  weighted by share of late close-game batters faced
                    (7th inning or later, score within 3)
      pen_skill_f5: weighted by share of early relief batters faced
                    (innings 1-5, not the starter)
    Either falls back to plain batters-faced shares if the team has none of
    that kind of work in the window. A pool with nobody available gets the
    league average.
    """
    con.execute(f"""
        create or replace table pen_pool as
        with q as (select distinct team_id, season, cutoff from {keys})
        select q.team_id, q.season, q.cutoff, a.pitcher_id as player_id,
               sum(a.bf) bf, sum(a.late_close_bf) late_bf, sum(a.early_relief_bf) early_bf,
               bool_or(a.data_date = q.cutoff - interval 1 day) as pitched_d1,
               bool_or(a.data_date = q.cutoff - interval 2 day) as pitched_d2,
               coalesce(sum(a.pitches) filter (where a.data_date = q.cutoff - interval 1 day), 0) as pitches_d1
        from q join app a on a.team_id = q.team_id and not a.started
             and a.data_date < q.cutoff and a.data_date >= q.cutoff - interval 30 day
        group by 1, 2, 3, 4
    """)
    con.execute("create or replace table pen_keys as select distinct player_id, season, cutoff from pen_pool")
    priors.pitcher_rates(con, "pen_keys", "pen_rates")
    s = skill_sql(skill, "r.k_rate", "r.bb_rate", "r.gb_rate")
    con.execute(f"""
        create or replace table pen as
        with p as (
            select p.*, {s} + case when r.rookie then {ROOKIE_PIT_RUNS} else 0 end as skill,
                   not ((p.pitched_d1 and p.pitched_d2) or p.pitches_d1 >= 25) as available
            from pen_pool p join pen_rates r using (player_id, season, cutoff)
        ), t as (
            select team_id, season, cutoff, sum(late_bf) tot_late, sum(early_bf) tot_early, sum(bf) tot_bf
            from p group by 1, 2, 3
        ), wts as (
            select p.*,
                   case when t.tot_late > 0 then p.late_bf / t.tot_late else p.bf / t.tot_bf end as w8,
                   case when t.tot_early > 0 then p.early_bf / t.tot_early else p.bf / t.tot_bf end as w5
            from p join t using (team_id, season, cutoff)
        )
        select team_id, season, cutoff,
               sum(w8 * skill) filter (where available) / nullif(sum(w8) filter (where available), 0) as pen_skill_8,
               sum(w5 * skill) filter (where available) / nullif(sum(w5) filter (where available), 0) as pen_skill_f5,
               1 - coalesce(sum(w8) filter (where available), 0) as pen_unavail_share_8,
               count(*) as pen_pool_n
        from wts group by 1, 2, 3
    """)
