"""
The prior layer: each player's rates, as known before a cutoff date.

A pitcher with 3 starts in April has a strikeout rate that is mostly noise,
so it gets blended with his earlier seasons and pulled toward the league
average, less so as his sample grows. One formula for every component
(a Marcel-style system, design 5.4):

    rate = (sum over seasons s of  w_s * count_s  +  k * mu)
           / (sum over seasons s of  w_s * n_s    +  k)

  count_s  the events in season s (strikeouts, say)
  n_s      the chances in season s (plate appearances, batters faced...)
  w_s      the season's weight: this season so far = 1, earlier seasons less
  mu       the league average (the prior mean), from the previous season
  k        how many chances' worth of league average to mix in; bigger k =
           more shrinkage. Starting values are the standard stabilization
           points; the model phase tunes them inside each fold's training
           seasons only

"This season so far" means games whose data date is BEFORE the cutoff date,
so a game never sees itself or anything later (the leak rule).

Seasons 2021 onward come from the pitch files (Statcast). Earlier seasons
come from the official season lines, which define ground balls differently
and have no park adjustment, so they are mapped onto the Statcast scale
using 2021, the season both sources cover, and given a little less weight.
"""
from __future__ import annotations

import duckdb

FIRST_STATCAST_SEASON = 2021

# Season weights relative to the current season (fixed at 1).
# Hitters 5/4/3 and pitchers 3/2/1 for the three previous seasons (design
# 5.4), scaled so the latest past season is a bit below the current one.
HIT_WEIGHTS = {1: 0.8, 2: 0.64, 3: 0.48}
PIT_WEIGHTS = {1: 0.75, 2: 0.5, 3: 0.25}
LINES_DISCOUNT = 0.8  # extra weight multiplier for pre-2021 official lines

# Shrinkage k, in the component's own chances (design 5.4 starting values;
# wOBA and runs-per-batter aren't listed there, so these are our starting
# points, to be tuned).
K_HIT = {"k": 60, "bb": 120, "woba": 300}
K_PIT = {"k": 70, "bb": 170, "gb": 70, "hr_fb": 400}

# Players with no MLB history start below average (design 5.4: about .020
# wOBA for hitters). Re-estimated inside the folds later.
ROOKIE_WOBA_OFFSET = -0.020

# Marcel age curve on wOBA-type rates: +0.6% per year younger than 29,
# -0.3% per year older, applied to PRIOR seasons only before blending (the
# current season is already at the player's current age). A starting value.
PEAK_AGE = 29


def build_player_seasons(con: duckdb.DuckDBPyConnection) -> None:
    """Season totals per player, and daily running totals for the current
    season. Needs tables pa, app, games, park_factors, season_lines."""

    # Park factor of every regular-season game's venue for that year. Missing
    # (new park) -> neutral 1.0.
    con.execute("""
        create or replace table game_pf as
        select g.game_id, coalesce(pf.pf_runs, 1.0) as pf
        from games g left join park_factors pf on pf.venue_id = g.venue_id and pf.year = g.season
    """)

    # ---- hitters, per game (regular season only) ----
    # wOBA is park-neutralized per plate appearance: runs scale roughly with
    # the square of wOBA, so a park that inflates runs by pf inflates wOBA by
    # about sqrt(pf).
    con.execute("""
        create or replace table hit_game as
        select p.batter_id as player_id, p.season, p.data_date,
               count(*) - sum(p.sac_bunt) as pa,
               sum(p.k) as k, sum(p.bb) as bb, sum(p.hbp) as hbp,
               sum(p.woba_value / sqrt(gp.pf)) as woba_num,
               sum(p.woba_denom) as woba_den
        from pa p join game_pf gp using (game_id)
        where p.game_type = 'R'
        group by 1, 2, 3
    """)
    # ---- pitchers, per game ----
    con.execute("""
        create or replace table pit_game as
        select p.pitcher_id as player_id, p.season, p.data_date,
               count(*) - sum(p.sac_bunt) as bf,
               sum(p.k) as k, sum(p.bb) as bb, sum(p.hbp) as hbp, sum(p.hr) as hr,
               sum(p.in_play) as bip, sum(p.gb) as gb, sum(p.fb) + sum(p.hr) as fb_all
        from pa p
        where p.game_type = 'R'
        group by 1, 2, 3
    """)
    # Fastball velocity per pitcher per game (four-seam and sinker only), for
    # the velocity-change feature.
    con.execute("""
        create or replace table velo_game as
        select pitcher_id as player_id, season, data_date,
               sum(release_speed) as velo_sum, count(release_speed) as velo_n
        from pitches
        where game_type = 'R' and pitch_type in ('FF', 'SI') and release_speed is not null
        group by 1, 2, 3
    """)

    # ---- mapping pre-2021 official lines onto the Statcast scale ----
    # wOBA: lines use fixed weights; scale so 2021 league wOBA matches.
    # Ground balls: lines give ground outs / (ground outs + air outs); map to
    # Statcast's ground balls / balls in play with a straight line fitted on
    # 2021 pitchers with 100+ balls in play in both sources.
    con.execute("""
        create or replace table lines_hit as
        select player_id, season,
               pa, so as k, bb - ibb as bb, hbp,
               0.69 * (bb - ibb) + 0.72 * hbp + 0.88 * (h - d2 - d3 - hr) + 1.25 * d2 + 1.58 * d3 + 2.03 * hr as woba_num,
               ab + bb - ibb + sf + hbp as woba_den
        from season_lines where grp = 'hitting'
    """)
    con.execute("""
        create or replace table lines_pit as
        select player_id, season, bf, so as k, bb - ibb as bb, hbp, hr,
               go + ao as bip_proxy,
               case when go + ao > 0 then go::double / (go + ao) end as go_share
        from season_lines where grp = 'pitching'
    """)
    scale = con.execute(f"""
        select (select sum(woba_num) / sum(woba_den) from hit_game where season = {FIRST_STATCAST_SEASON})
             / (select sum(woba_num) / sum(woba_den) from lines_hit where season = {FIRST_STATCAST_SEASON})
    """).fetchone()[0]
    a, b = con.execute(f"""
        with s as (select player_id, sum(gb)::double / sum(bip) sc, sum(bip) n
                   from pit_game where season = {FIRST_STATCAST_SEASON} group by 1 having sum(bip) >= 100),
             l as (select player_id, go_share, bip_proxy from lines_pit where season = {FIRST_STATCAST_SEASON} and bip_proxy >= 100)
        select regr_intercept(sc, go_share), regr_slope(sc, go_share) from s join l using (player_id)
    """).fetchone()
    con.execute(f"create or replace table lines_map as select {scale} as woba_scale, {a} as gb_a, {b} as gb_b")

    # ---- season totals: Statcast for 2021+, lines before ----
    con.execute(f"""
        create or replace table hit_season as
        select player_id, season, sum(pa) pa, sum(k) k, sum(bb) bb, sum(hbp) hbp,
               sum(woba_num) woba_num, sum(woba_den) woba_den, 1.0 as src_w
        from hit_game group by 1, 2
        union all
        select player_id, season, pa, k, bb, hbp,
               woba_num * (select woba_scale from lines_map), woba_den, {LINES_DISCOUNT}
        from lines_hit where season < {FIRST_STATCAST_SEASON}
    """)
    con.execute(f"""
        create or replace table pit_season as
        select player_id, season, sum(bf) bf, sum(k) k, sum(bb) bb, sum(hbp) hbp, sum(hr) hr,
               sum(bip) bip, sum(gb) gb, sum(fb_all) fb_all, 1.0 as src_w
        from pit_game group by 1, 2
        union all
        select player_id, season, bf, k, bb, hbp, hr,
               bip_proxy as bip,
               bip_proxy * greatest(0.2, least(0.7, (select gb_a from lines_map) + (select gb_b from lines_map) * go_share)) as gb,
               null as fb_all, {LINES_DISCOUNT}
        from lines_pit where season < {FIRST_STATCAST_SEASON}
    """)
    con.execute("""
        create or replace table velo_season as
        select player_id, season, sum(velo_sum) / sum(velo_n) as velo, sum(velo_n) as velo_n
        from velo_game group by 1, 2
    """)

    # ---- league averages per season (the prior mean for the NEXT season) ----
    con.execute("""
        create or replace table league_season as
        select h.season,
               h.k / h.pa as hit_k, h.bb / h.pa as hit_bb, h.woba_num / h.woba_den as woba,
               p.k / p.bf as pit_k, p.bb / p.bf as pit_bb, p.gb / p.bip as pit_gb,
               p.hr / nullif(p.fb_all, 0) as pit_hr_fb
        from (select season, sum(pa) pa, sum(k) k, sum(bb) bb, sum(woba_num) woba_num, sum(woba_den) woba_den
              from hit_season group by 1) h
        join (select season, sum(bf) bf, sum(k) k, sum(bb) bb, sum(gb) gb, sum(bip) bip, sum(hr) hr, sum(fb_all) fb_all
              from pit_season group by 1) p using (season)
    """)

    # ---- running totals within each season, by data date ----
    for src, dst, cols in [
        ("hit_game", "hit_cum", ["pa", "k", "bb", "hbp", "woba_num", "woba_den"]),
        ("pit_game", "pit_cum", ["bf", "k", "bb", "hbp", "hr", "bip", "gb", "fb_all"]),
        ("velo_game", "velo_cum", ["velo_sum", "velo_n"]),
    ]:
        sums = ", ".join(f"sum(sum({c})) over w as {c}" for c in cols)
        con.execute(f"""
            create or replace table {dst} as
            select player_id, season, data_date, {sums}
            from {src} group by player_id, season, data_date
            window w as (partition by player_id, season order by data_date rows unbounded preceding)
        """)


def _prev_join(alias: str, table: str, lag: int) -> str:
    return (f"left join {table} {alias} on {alias}.player_id = q.player_id "
            f"and {alias}.season = q.season - {lag}")


def hitter_rates(con: duckdb.DuckDBPyConnection, keys: str, out: str) -> None:
    """Blended hitter rates for every row of `keys` (columns player_id,
    season, cutoff). Writes table `out` with woba, k_rate, bb_rate,
    pa_current, pa_history, rookie flag."""
    w1, w2, w3 = HIT_WEIGHTS[1], HIT_WEIGHTS[2], HIT_WEIGHTS[3]
    con.execute(f"""
        create or replace table {out} as
        with q as (select distinct player_id, season, cutoff from {keys}),
        cur as (
            select q.player_id, q.season, q.cutoff, c.pa, c.k, c.bb, c.hbp, c.woba_num, c.woba_den
            from q asof left join hit_cum c
              on c.player_id = q.player_id and c.season = q.season and q.cutoff > c.data_date
        ),
        hist as (
            select q.player_id, q.season, q.cutoff,
                   exists (select 1 from hit_season h where h.player_id = q.player_id and h.season < q.season) as has_history
            from q
        ),
        joined as (
            select q.player_id, q.season, q.cutoff, c.pa, c.k, c.bb, c.woba_num, c.woba_den,
                   s1.src_w s1w, s1.pa s1pa, s1.k s1k, s1.bb s1bb, s1.woba_num * {_age_factor('pl.birth_date', 'q.cutoff')} s1wn, s1.woba_den s1wd,
                   s2.src_w s2w, s2.pa s2pa, s2.k s2k, s2.bb s2bb, s2.woba_num * {_age_factor('pl.birth_date', 'q.cutoff')} s2wn, s2.woba_den s2wd,
                   s3.src_w s3w, s3.pa s3pa, s3.k s3k, s3.bb s3bb, s3.woba_num * {_age_factor('pl.birth_date', 'q.cutoff')} s3wn, s3.woba_den s3wd,
                   l.hit_k, l.hit_bb, l.woba, h.has_history, pl.birth_date
            from q
            join cur c using (player_id, season, cutoff)
            join hist h using (player_id, season, cutoff)
            {_prev_join('s1', 'hit_season', 1)}
            {_prev_join('s2', 'hit_season', 2)}
            {_prev_join('s3', 'hit_season', 3)}
            left join league_season l on l.season = q.season - 1
            left join players pl on pl.player_id = q.player_id
        )
        select player_id, season, cutoff,
               {_blend_cols('k', 'pa', K_HIT['k'], 'hit_k', w1, w2, w3)} as k_rate,
               {_blend_cols('bb', 'pa', K_HIT['bb'], 'hit_bb', w1, w2, w3)} as bb_rate,
               {_blend_cols('wn', 'wd', K_HIT['woba'], f"woba + case when has_history then 0 else {ROOKIE_WOBA_OFFSET} end", w1, w2, w3, cur_count='woba_num', cur_n='woba_den')} as woba,
               coalesce(pa, 0) as pa_current,
               coalesce(s1pa, 0) + coalesce(s2pa, 0) + coalesce(s3pa, 0) as pa_history,
               not has_history as rookie
        from joined
    """)


def _blend_cols(count: str, n: str, k: float, mu: str, w1: float, w2: float, w3: float,
                cur_count: str | None = None, cur_n: str | None = None) -> str:
    """SQL for the blend formula over columns named like s1k / s1pa."""
    cc = cur_count or count
    cn = cur_n or n
    num = (f"coalesce({cc},0) + {w1}*coalesce(s1w*s1{count},0) + {w2}*coalesce(s2w*s2{count},0) "
           f"+ {w3}*coalesce(s3w*s3{count},0)")
    den = (f"coalesce({cn},0) + {w1}*coalesce(s1w*s1{n},0) + {w2}*coalesce(s2w*s2{n},0) "
           f"+ {w3}*coalesce(s3w*s3{n},0)")
    return f"((({num}) + {k} * ({mu})) / (({den}) + {k}))"


def _age_factor(birth: str, asof: str) -> str:
    """Marcel-style age multiplier on wOBA-type rates (1 when age unknown)."""
    age = f"(date_diff('day', {birth}, {asof}) / 365.25)"
    return (f"(case when {birth} is null then 1.0 "
            f"when {age} < {PEAK_AGE} then 1 + 0.006 * ({PEAK_AGE} - {age}) "
            f"else 1 - 0.003 * ({age} - {PEAK_AGE}) end)")


def pitcher_rates(con: duckdb.DuckDBPyConnection, keys: str, out: str) -> None:
    """Blended pitcher rates for every row of `keys` (player_id, season,
    cutoff). Writes `out` with k_rate, bb_rate, gb_rate, hr_fb, bf_current,
    bf_history, rookie, velo_now, velo_prev (fastball mph)."""
    w1, w2, w3 = PIT_WEIGHTS[1], PIT_WEIGHTS[2], PIT_WEIGHTS[3]
    con.execute(f"""
        create or replace table {out} as
        with q as (select distinct player_id, season, cutoff from {keys}),
        cur as (
            select q.player_id, q.season, q.cutoff, c.bf, c.k, c.bb, c.hr, c.bip, c.gb, c.fb_all
            from q asof left join pit_cum c
              on c.player_id = q.player_id and c.season = q.season and q.cutoff > c.data_date
        ),
        vcur as (
            select q.player_id, q.season, q.cutoff, v.velo_sum / nullif(v.velo_n, 0) as velo_now, v.velo_n
            from q asof left join velo_cum v
              on v.player_id = q.player_id and v.season = q.season and q.cutoff > v.data_date
        ),
        joined as (
            select q.player_id, q.season, q.cutoff, c.bf, c.k, c.bb, c.hr, c.bip, c.gb, c.fb_all,
                   s1.src_w s1w, s1.bf s1bf, s1.k s1k, s1.bb s1bb, s1.bip s1bip, s1.gb s1gb, case when s1.fb_all is null then null else s1.hr end s1hr, s1.fb_all s1fb,
                   s2.src_w s2w, s2.bf s2bf, s2.k s2k, s2.bb s2bb, s2.bip s2bip, s2.gb s2gb, case when s2.fb_all is null then null else s2.hr end s2hr, s2.fb_all s2fb,
                   s3.src_w s3w, s3.bf s3bf, s3.k s3k, s3.bb s3bb, s3.bip s3bip, s3.gb s3gb, case when s3.fb_all is null then null else s3.hr end s3hr, s3.fb_all s3fb,
                   l.pit_k, l.pit_bb, l.pit_gb, l.pit_hr_fb,
                   exists (select 1 from pit_season h where h.player_id = q.player_id and h.season < q.season) as has_history,
                   vc.velo_now, vc.velo_n, vp.velo as velo_prev
            from q
            join cur c using (player_id, season, cutoff)
            join vcur vc using (player_id, season, cutoff)
            {_prev_join('s1', 'pit_season', 1)}
            {_prev_join('s2', 'pit_season', 2)}
            {_prev_join('s3', 'pit_season', 3)}
            left join league_season l on l.season = q.season - 1
            left join velo_season vp on vp.player_id = q.player_id and vp.season = q.season - 1
        )
        select player_id, season, cutoff,
               {_blend_cols('k', 'bf', K_PIT['k'], 'pit_k', w1, w2, w3)} as k_rate,
               {_blend_cols('bb', 'bf', K_PIT['bb'], 'pit_bb', w1, w2, w3)} as bb_rate,
               {_blend_cols('gb', 'bip', K_PIT['gb'], 'pit_gb', w1, w2, w3)} as gb_rate,
               {_blend_cols('hr', 'fb', K_PIT['hr_fb'], 'coalesce(pit_hr_fb, 0.125)', w1, w2, w3, cur_count='hr', cur_n='fb_all')} as hr_fb,
               coalesce(bf, 0) as bf_current,
               coalesce(s1bf, 0) + coalesce(s2bf, 0) + coalesce(s3bf, 0) as bf_history,
               not has_history as rookie,
               velo_now, coalesce(velo_n, 0) as velo_n_now, velo_prev
        from joined
    """)
