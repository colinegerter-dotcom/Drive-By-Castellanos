"""
Environment and team-strength features.

  park_factor      runs park factor built from prior seasons only; a park
                   with no factor yet (new, or moved) is neutral with a flag
  temp_f           game-time temperature; 72 in a fixed dome
  roof_park        the park has a retractable roof (whether it was closed
                   is unknown historically, so the model sees the flag)
  league_env_*     league runs per team-game for the segment: last season
                   blended with this season so far
  team_off_* /     the batting team's runs scored and the fielding team's
  team_def_*       runs allowed per game, park-adjusted, blended with last
                   season. These feed baseline B1 (team strength), the bar
                   every model must beat; the main model doesn't use them
                   (design 5.6: they would double-count the lineup)

Known limit (flagged): temp_f here is the OBSERVED game-time temperature.
P1 is supposed to use the forecast issued before 10am. Archived forecasts
(Open-Meteo historical forecast service) are a separate step; until then
the weather effect in backtests is slightly overstated.
"""
from __future__ import annotations

import duckdb

from .inputs import DOME_VENUE_IDS, ROOFED_VENUE_IDS

K_LEAGUE = 500      # team-games of last season's level mixed into this season's
K_TEAM = 30         # games of a team's prior level mixed into its season so far
TEAM_PRIOR_REGRESS = 1 / 3  # last season's team level pulled a third to the league


def environment(con: duckdb.DuckDBPyConnection, keys: str) -> None:
    """Per row of `keys` (game_id, season, cutoff, venue_id, bat_team,
    fld_team): writes table `env`."""
    domes = ",".join(map(str, DOME_VENUE_IDS))
    roofs = ",".join(map(str, ROOFED_VENUE_IDS - DOME_VENUE_IDS))

    # Park-adjusted runs per team-game, regular season, with the date the
    # game's data became usable.
    con.execute("""
        create or replace table tr_adj as
        select t.game_id, t.team_id, g.season, g.data_date,
               case when t.team_id = g.home_team then g.away_team else g.home_team end as opp_id,
               t.runs_f5, t.runs_8, gp.pf,
               t.runs_f5 / gp.pf as off_f5, t.runs_8 / gp.pf as off_8
        from team_runs t join games g using (game_id) join game_pf gp using (game_id)
        where g.game_type = 'R' and t.runs_8 is not null and t.innings >= 8
    """)
    con.execute("""
        create or replace table league_cum as
        select season, data_date,
               sum(count(*)) over w n, sum(sum(runs_f5)) over w r5, sum(sum(runs_8)) over w r8
        from tr_adj group by 1, 2
        window w as (partition by season order by data_date rows unbounded preceding)
    """)
    con.execute("""
        create or replace table league_prev as
        select season + 1 as season, avg(runs_f5) r5, avg(runs_8) r8 from tr_adj group by season
        -- The first season has no earlier inning scores, so it borrows its
        -- own full-season average. Only 2021 (warm-up, never scored) is hit.
        union all
        select season, avg(runs_f5), avg(runs_8) from tr_adj
        where season = (select min(season) from tr_adj) group by season
    """)
    # team offense (runs scored) and defense (runs allowed), park-adjusted
    for side, id_col, val5, val8 in [("off", "team_id", "off_f5", "off_8"), ("def", "opp_id", "off_f5", "off_8")]:
        con.execute(f"""
            create or replace table team_{side}_cum as
            select {id_col} as team_id, season, data_date,
                   sum(count(*)) over w n, sum(sum({val5})) over w s5, sum(sum({val8})) over w s8
            from tr_adj group by 1, 2, 3
            window w as (partition by {id_col}, season order by data_date rows unbounded preceding)
        """)
        con.execute(f"""
            create or replace table team_{side}_prev as
            select {id_col} as team_id, season + 1 as season, avg({val5}) m5, avg({val8}) m8
            from tr_adj group by 1, 2
        """)

    con.execute(f"""
        create or replace table env as
        with q as (select * from {keys}),
        lc as (select q.game_id, q.bat_team, c.n, c.r5, c.r8 from q asof left join league_cum c
               on c.season = q.season and q.cutoff > c.data_date),
        oc as (select q.game_id, q.bat_team, c.n, c.s5, c.s8 from q asof left join team_off_cum c
               on c.team_id = q.bat_team and c.season = q.season and q.cutoff > c.data_date),
        dc as (select q.game_id, q.bat_team, c.n, c.s5, c.s8 from q asof left join team_def_cum c
               on c.team_id = q.fld_team and c.season = q.season and q.cutoff > c.data_date)
        select q.game_id, q.bat_team,
               coalesce(pf.pf_runs, 1.0) as park_factor,
               (pf.pf_runs is null) as new_park,
               case when q.venue_id in ({domes}) then 72.0 else coalesce(gc.temp_f, 72.0) end as temp_f,
               (gc.temp_f is null and q.venue_id not in ({domes})) as temp_missing,
               (q.venue_id in ({roofs})) as roof_park,
               -- league level: last season's mean, blended with this season so far
               (coalesce(lc.r5, 0) + {K_LEAGUE} * lp.r5) / (coalesce(lc.n, 0) + {K_LEAGUE}) as league_env_f5,
               (coalesce(lc.r8, 0) + {K_LEAGUE} * lp.r8) / (coalesce(lc.n, 0) + {K_LEAGUE}) as league_env_8,
               -- team strength (baseline B1 inputs)
               (coalesce(oc.s5, 0) + {K_TEAM} * (lp.r5 + (1 - {TEAM_PRIOR_REGRESS}) * (coalesce(op.m5, lp.r5) - lp.r5)))
                 / (coalesce(oc.n, 0) + {K_TEAM}) as team_off_f5,
               (coalesce(oc.s8, 0) + {K_TEAM} * (lp.r8 + (1 - {TEAM_PRIOR_REGRESS}) * (coalesce(op.m8, lp.r8) - lp.r8)))
                 / (coalesce(oc.n, 0) + {K_TEAM}) as team_off_8,
               (coalesce(dc.s5, 0) + {K_TEAM} * (lp.r5 + (1 - {TEAM_PRIOR_REGRESS}) * (coalesce(dp.m5, lp.r5) - lp.r5)))
                 / (coalesce(dc.n, 0) + {K_TEAM}) as team_def_f5,
               (coalesce(dc.s8, 0) + {K_TEAM} * (lp.r8 + (1 - {TEAM_PRIOR_REGRESS}) * (coalesce(dp.m8, lp.r8) - lp.r8)))
                 / (coalesce(dc.n, 0) + {K_TEAM}) as team_def_8
        from q
        left join park_factors pf on pf.venue_id = q.venue_id and pf.year = q.season
        left join conditions gc on gc.game_id = q.game_id
        left join lc on lc.game_id = q.game_id and lc.bat_team = q.bat_team
        left join oc on oc.game_id = q.game_id and oc.bat_team = q.bat_team
        left join dc on dc.game_id = q.game_id and dc.bat_team = q.bat_team
        left join league_prev lp on lp.season = q.season
        left join team_off_prev op on op.team_id = q.bat_team and op.season = q.season
        left join team_def_prev dp on dp.team_id = q.fld_team and dp.season = q.season
    """)
