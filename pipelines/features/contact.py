"""
Round 2 group 1: contact quality (design 5.7 item 1, as built: design E7).

How hard and at what angle a batter hits the ball says more about his real
hitting than whether the ball found a gap. Per batted ball:

  contact xwOBA   the average wOBA value of balls hit at that exit velocity
                  and launch angle, from a lookup (2 mph x 4 degree bins,
                  smoothed toward coarser bins) fitted on 2021 batted balls
                  and then frozen. Exit velocity and angle only: the pitch
                  files have no sprint speed, and Savant's own xwOBA can use
                  same-season speed, so it isn't used
  hard hit        exit velocity 95 mph or more
  barrel          Statcast's barrel zone, approximated: 98+ mph, launch
                  angle from 26-(ev-98) down to 8, up to 30 at 98 mph,
                  31 at 99, then 33 + 17/16 per mph above 100, up to 50

Each becomes a rate per batted ball for hitters (what they hit) and pitchers
(what they allow), blended with earlier seasons through the prior layer
(design 5.4): this season so far plus up to three earlier Statcast seasons
with the same season weights as the other rates, shrunk toward last
season's league average with k = 200 batted balls (contact xwOBA, hard hit)
and 400 (barrel). Starting values, not tuned. Only games final before the
cutoff count (leak rule).

Features per team-game (the batting team's view):
  ct_lu_xw, ct_lu_brl, ct_lu_hh   lineup averages, weighted like lineup_woba
  ct_sp_xw, ct_sp_brl             opposing starter's rates allowed
  ct_pen_xw_f5, ct_pen_xw_8       opposing bullpen's contact xwOBA allowed,
                                  weighted and filtered for availability like
                                  pen_skill
"""
from __future__ import annotations

import duckdb

from . import priors

FIT_SEASON = 2021
K_CT = {"xw": 200, "hh": 200, "brl": 400}
SMOOTH_FINE = 20     # pseudo-count pulling a fine bin toward its coarse bin
SMOOTH_COARSE = 50   # pseudo-count pulling a coarse bin toward the overall mean


def _barrel_sql(ev: str, la: str) -> str:
    upper = f"least(50, case when {ev} < 99 then 30 when {ev} < 100 then 31 else 33 + ({ev} - 100) * 17.0 / 16 end)"
    lower = f"greatest(8, 26 - ({ev} - 98))"
    return f"({ev} >= 98 and {la} between {lower} and {upper})"


def build_contact(con: duckdb.DuckDBPyConnection) -> None:
    """Batted balls, the frozen xwOBA lookup, per-game and per-season
    contact totals for hitters and pitchers, and league averages."""
    con.execute("""
        create or replace table bb_raw as
        select game_id, season, date, data_date, game_type, batter_id, pitcher_id,
               exit_velocity as ev, launch_angle::double as la, bb_type,
               coalesce(woba_value, 0) as woba_value
        from pitches
        where bb_type is not null and events is not null and events <> 'truncated_pa'
        qualify row_number() over (partition by game_id, at_bat_id order by pitch_number desc) = 1
    """)
    # the lookup, fitted on one season only and frozen
    con.execute(f"""
        create or replace table xw_fit as
        with b as (select *, floor(least(greatest(ev, 40), 121) / 2) as evb,
                             floor((least(greatest(la, -90), 90) + 90) / 4) as lab
                   from bb_raw where season = {FIT_SEASON} and game_type = 'R' and ev is not null and la is not null),
        g as (select avg(woba_value) as m from b),
        coarse as (select floor(evb / 3) as evc, floor(lab / 3) as lac, sum(woba_value) s, count(*) n
                   from b group by 1, 2),
        cs as (select evc, lac, (s + {SMOOTH_COARSE} * (select m from g)) / (n + {SMOOTH_COARSE}) as cm from coarse),
        fine as (select evb, lab, sum(woba_value) s, count(*) n from b group by 1, 2)
        select f.evb, f.lab, (f.s + {SMOOTH_FINE} * c.cm) / (f.n + {SMOOTH_FINE}) as xw
        from fine f join cs c on c.evc = floor(f.evb / 3) and c.lac = floor(f.lab / 3)
    """)
    con.execute(f"""
        create or replace table xw_coarse as
        with b as (select *, floor(least(greatest(ev, 40), 121) / 2) as evb,
                             floor((least(greatest(la, -90), 90) + 90) / 4) as lab
                   from bb_raw where season = {FIT_SEASON} and game_type = 'R' and ev is not null and la is not null),
        g as (select avg(woba_value) as m from b)
        select floor(evb / 3) as evc, floor(lab / 3) as lac,
               (sum(woba_value) + {SMOOTH_COARSE} * any_value(g.m)) / (count(*) + {SMOOTH_COARSE}) as cm
        from b, g group by 1, 2
    """)
    con.execute(f"""
        create or replace table xw_missing as
        select bb_type, avg(woba_value) as xw from bb_raw
        where season = {FIT_SEASON} and game_type = 'R' and (ev is null or la is null) group by 1
    """)
    con.execute(f"""
        create or replace table bb as
        with b as (select *, floor(least(greatest(ev, 40), 121) / 2) as evb,
                             floor((least(greatest(la, -90), 90) + 90) / 4) as lab from bb_raw)
        select b.game_id, b.season, b.data_date, b.game_type, b.batter_id, b.pitcher_id,
               coalesce(case when b.ev is not null and b.la is not null then coalesce(f.xw, c.cm) end,
                        m.xw, (select avg(woba_value) from bb_raw where season = {FIT_SEASON})) as xw,
               coalesce(b.ev >= 95, false)::int as hh,
               coalesce({_barrel_sql('b.ev', 'b.la')}, false)::int as brl
        from b
        left join xw_fit f on f.evb = b.evb and f.lab = b.lab
        left join xw_coarse c on c.evc = floor(b.evb / 3) and c.lac = floor(b.lab / 3)
        left join xw_missing m on m.bb_type = b.bb_type
    """)
    for role, id_col in (("hit", "batter_id"), ("pit", "pitcher_id")):
        con.execute(f"""
            create or replace table ct_{role}_game as
            select {id_col} as player_id, season, data_date,
                   count(*) as n, sum(xw) as xw, sum(hh) as hh, sum(brl) as brl
            from bb where game_type = 'R' group by 1, 2, 3
        """)
        con.execute(f"""
            create or replace table ct_{role}_season as
            select player_id, season, sum(n) n, sum(xw) xw, sum(hh) hh, sum(brl) brl, 1.0 as src_w
            from ct_{role}_game group by 1, 2
        """)
        con.execute(f"""
            create or replace table ct_{role}_cum as
            select player_id, season, data_date,
                   sum(sum(n)) over w n, sum(sum(xw)) over w xw, sum(sum(hh)) over w hh, sum(sum(brl)) over w brl
            from ct_{role}_game group by player_id, season, data_date
            window w as (partition by player_id, season order by data_date rows unbounded preceding)
        """)
    # last season's league rates are the prior mean; the first season borrows
    # its own (only 2021, the warm-up season, is affected)
    con.execute(f"""
        create or replace table ct_league as
        with s as (select season, sum(xw) / sum(n) xw, sum(hh) / sum(n) hh, sum(brl) / sum(n) brl
                   from ct_hit_season group by 1)
        select season + 1 as season, xw, hh, brl from s
        union all select season, xw, hh, brl from s where season = (select min(season) from s)
    """)


def contact_rates(con: duckdb.DuckDBPyConnection, role: str, keys: str, out: str) -> None:
    """Blended contact rates for (player_id, season, cutoff) rows of keys."""
    w = priors.HIT_WEIGHTS if role == "hit" else priors.PIT_WEIGHTS
    w1, w2, w3 = w[1], w[2], w[3]

    def blend(col, k):
        num = (f"coalesce(c.{col},0) + {w1}*coalesce(s1.{col},0) + {w2}*coalesce(s2.{col},0) + {w3}*coalesce(s3.{col},0)")
        den = (f"coalesce(c.n,0) + {w1}*coalesce(s1.n,0) + {w2}*coalesce(s2.n,0) + {w3}*coalesce(s3.n,0)")
        return f"(({num}) + {k} * l.{col}) / (({den}) + {k})"

    con.execute(f"""
        create or replace table {out} as
        with q as (select distinct player_id, season, cutoff from {keys}),
        cur as (
            select q.player_id, q.season, q.cutoff, c.n, c.xw, c.hh, c.brl
            from q asof left join ct_{role}_cum c
              on c.player_id = q.player_id and c.season = q.season and q.cutoff > c.data_date
        )
        select q.player_id, q.season, q.cutoff,
               {blend('xw', K_CT['xw'])} as ct_xw,
               {blend('hh', K_CT['hh'])} as ct_hh,
               {blend('brl', K_CT['brl'])} as ct_brl
        from q join cur c using (player_id, season, cutoff)
        left join ct_{role}_season s1 on s1.player_id = q.player_id and s1.season = q.season - 1
        left join ct_{role}_season s2 on s2.player_id = q.player_id and s2.season = q.season - 2
        left join ct_{role}_season s3 on s3.player_id = q.player_id and s3.season = q.season - 3
        left join ct_league l on l.season = q.season
    """)


def contact_features(con: duckdb.DuckDBPyConnection, lineup_tbl: str, keys: str = "tg") -> None:
    """Writes table ct_feat (game_id, bat_team, seven features). Needs the
    lineup table used for the other lineup features, slot_w, and pen_pool
    (built by pitching.bullpen)."""
    build_contact(con)
    con.execute(f"""
        create or replace table ct_lu_keys as
        select distinct l.player_id, q.season, q.cutoff
        from {lineup_tbl} l join {keys} q on q.game_id = l.game_id and q.bat_team = l.team_id
    """)
    contact_rates(con, "hit", "ct_lu_keys", "ct_lu_rates")
    con.execute(f"create or replace table ct_sp_keys as select distinct sp_id as player_id, season, cutoff from {keys} where sp_id is not null")
    contact_rates(con, "pit", "ct_sp_keys", "ct_sp_rates")
    con.execute("create or replace table ct_pen_keys as select distinct player_id, season, cutoff from pen_pool")
    contact_rates(con, "pit", "ct_pen_keys", "ct_pen_rates")
    con.execute(f"""
        create or replace table ct_lu as
        select q.game_id, q.bat_team,
               sum(w.w_8 * l.weight_share * r.ct_xw) / sum(w.w_8 * l.weight_share) as ct_lu_xw,
               sum(w.w_8 * l.weight_share * r.ct_brl) / sum(w.w_8 * l.weight_share) as ct_lu_brl,
               sum(w.w_8 * l.weight_share * r.ct_hh) / sum(w.w_8 * l.weight_share) as ct_lu_hh
        from {keys} q
        join {lineup_tbl} l on l.game_id = q.game_id and l.team_id = q.bat_team
        join slot_w w on w.slot = l.slot
        join ct_lu_rates r on r.player_id = l.player_id and r.season = q.season and r.cutoff = q.cutoff
        group by q.game_id, q.bat_team
    """)
    con.execute("""
        create or replace table ct_pen as
        with p as (
            select p.*, r.ct_xw,
                   not ((p.pitched_d1 and p.pitched_d2) or p.pitches_d1 >= 25) as available
            from pen_pool p join ct_pen_rates r using (player_id, season, cutoff)
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
               sum(w8 * ct_xw) filter (where available) / nullif(sum(w8) filter (where available), 0) as ct_pen_xw_8,
               sum(w5 * ct_xw) filter (where available) / nullif(sum(w5) filter (where available), 0) as ct_pen_xw_f5
        from wts group by 1, 2, 3
    """)
    con.execute(f"""
        create or replace table ct_feat as
        select q.game_id, q.bat_team, lu.ct_lu_xw, lu.ct_lu_brl, lu.ct_lu_hh,
               sp.ct_xw as ct_sp_xw, sp.ct_brl as ct_sp_brl,
               pn.ct_pen_xw_f5, pn.ct_pen_xw_8
        from {keys} q
        left join ct_lu lu on lu.game_id = q.game_id and lu.bat_team = q.bat_team
        left join ct_sp_rates sp on sp.player_id = q.sp_id and sp.season = q.season and sp.cutoff = q.cutoff
        left join ct_pen pn on pn.team_id = q.fld_team and pn.season = q.season and pn.cutoff = q.cutoff
    """)
