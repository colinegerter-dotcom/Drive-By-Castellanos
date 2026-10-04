"""
Round 2 group 7: catcher framing (design E18, drafted 3 Oct 2026 and revised
after an independent pre-build review, before the build). The team-defense half
of the first draft was dropped by that review (it is mostly park, and the fielder
zone is an outcome); it can return later as its own group.

A catcher who receives borderline pitches well gets more called strikes than the
location alone predicts. The measure reuses E14's frozen 2021 called-strike grid
(table ump_grid, built by umpire.py):

  frm_cs   extra called strikes per 100 taken pitches credited to the fielding
           team's starting catcher, compared with the expected call at the
           same location, batter side and count; league-centred; shrunk to 0
  frm_runs_opp = frm_cs x RUNS_PER_POINT (76 taken pitches a game x 0.14 runs
           per extra strike): runs per game the fielding side saves

Steps are the same as umpire.py: residual per taken pitch (called strike minus
expected chance), minus the league's average residual over the same span (each
prior season's own average; the current season over days before the game's
date), pooled per catcher over the current season's days before the game
(weight 1) plus up to 3 prior seasons (0.8, 0.64, 0.48), shrunk as
100 x sum / (weighted pitches + K).

The catcher is the lineup's `C` slot of the fielding team (`players.position`
is not used). No lineup or no C row: feature 0 and frm_missing.

Prediction points: P2 only (the starting catcher is a lineup fact); at P1 every
row is 0 and missing. Needs umpire.umpire_features to have run first at P2
(it builds ump_taken and ump_grid and the league tables).

Table frm_feat: game_id, bat_team, frm_runs_opp, frm_missing.

Module switches for the leak test's mutation checks (E18), defaults must stay:
  SAME_DAY, LEAGUE_FULL_SEASON, CENTRE_PRIOR_WITH_CURRENT, FULL_SEASON_N
"""
from __future__ import annotations

import sys

import duckdb
import numpy as np

from .priors import HIT_WEIGHTS

K_SHRINK = 1000.0            # taken pitches (design E18)
RUNS_PER_POINT = 76.0 * 0.14 / 100.0   # 0.1064 runs a game per extra called strike per 100 taken pitches
SAME_DAY = False             # mutation only
LEAGUE_FULL_SEASON = False   # mutation only
CENTRE_PRIOR_WITH_CURRENT = False   # mutation only: prior seasons centred with the CURRENT season's league mean
FULL_SEASON_N = False        # mutation only: shrink with the current season's full-season pitch count
DATE_CUTOFF = False          # mutation only: cut pitches by the game's scheduled date, not the date its pitches were thrown (resumed games)


def framing_features(con: duckdb.DuckDBPyConnection, keys: str = "tg", point: str = "P2") -> dict:
    if point != "P2":
        con.execute(f"""
            create or replace table frm_feat as
            select game_id, bat_team, 0.0::double as frm_runs_opp, true as frm_missing from {keys}
        """)
        return {"point": point, "note": "P1: framing features are 0 by design (E18)"}

    # the starting catcher of each team-game: lineup slot C, exactly one expected
    con.execute("""
        create or replace table frm_catcher as
        select game_id, team_id, min(player_id) as catcher_id, count(*) as n_c
        from lineup where pos = 'C' and slot between 1 and 9 group by 1, 2
    """)
    bad = con.execute("select count(*) from frm_catcher where n_c <> 1").fetchone()[0]
    if bad:
        print(f"warning: {bad} team-game(s) with more than one C in the lineup; first id used", file=sys.stderr)
    dd = "(select g.date::date from games g where g.game_id = t.game_id)" if DATE_CUTOFF else "t.data_date"
    con.execute(f"""
        create or replace table frm_game as
        select t.game_id, t.season, {dd} as data_date, c.catcher_id,
               sum(t.cs - gr.p_cs) as r, count(*) as n
        from ump_taken t
        join ump_grid gr using (sd, cg, ix, iz)
        left join frm_catcher c on c.game_id = t.game_id and c.team_id = t.fld_team
        group by all
    """)
    con.execute("""
        create or replace table frm_season as
        select f.catcher_id, f.season, sum(f.r) - sum(f.n) * any_value(l.l) as rc, sum(f.n) as n
        from frm_game f join ump_lg_season l using (season)
        where f.catcher_id is not null
        group by all
    """)
    day_cmp = "<=" if SAME_DAY else "<"
    w1, w2, w3 = HIT_WEIGHTS[1], HIT_WEIGHTS[2], HIT_WEIGHTS[3]
    lg_cur = ("(select l from ump_lg_season s where s.season = k.season)" if LEAGUE_FULL_SEASON else
              "(select sum(d.r) / nullif(sum(d.n), 0) from ump_lg_day d "
              f"where d.season = k.season and d.data_date {day_cmp} k.date)")
    if CENTRE_PRIOR_WITH_CURRENT:
        pri_rc = ("sum(case k.season - s.season when 1 then {w1} when 2 then {w2} else {w3} end * "
                  "(s.rc + s.n * (select l from ump_lg_season z where z.season = s.season) - s.n * "
                  "(select l from ump_lg_season z where z.season = k.season)))")
    else:
        pri_rc = "sum(case k.season - s.season when 1 then {w1} when 2 then {w2} else {w3} end * s.rc)"
    pri_rc = pri_rc.format(w1=w1, w2=w2, w3=w3)
    n_den = ("coalesce((select sum(z.n) from frm_game z where z.catcher_id = j.catcher_id and z.season = "
             "(select season from games gg where gg.game_id = j.game_id)), 0)" if FULL_SEASON_N else "j.cur_n")
    con.execute(f"""
        create or replace table frm_feat_game as
        with k as (
            select distinct q.game_id, q.season, q.date, q.fld_team, c.catcher_id
            from {keys} q left join frm_catcher c on c.game_id = q.game_id and c.team_id = q.fld_team
        ),
        cur as (
            select k.game_id, k.fld_team, sum(u.r) as r, sum(u.n) as n
            from k join frm_game u
              on u.catcher_id = k.catcher_id and u.season = k.season and u.data_date {day_cmp} k.date
            group by k.game_id, k.fld_team
        ),
        pri as (
            select k.game_id, k.fld_team,
                   {pri_rc} as rc,
                   sum(case k.season - s.season when 1 then {w1} when 2 then {w2} else {w3} end * s.n) as n
            from k join frm_season s
              on s.catcher_id = k.catcher_id and s.season between k.season - 3 and k.season - 1
            group by k.game_id, k.fld_team
        ),
        j as (
            select k.game_id, k.fld_team, k.catcher_id,
                   coalesce(cur.r, 0) - coalesce(cur.n, 0) * coalesce({lg_cur}, 0) as cur_rc,
                   coalesce(cur.n, 0) as cur_n,
                   coalesce(pri.rc, 0) as pri_rc, coalesce(pri.n, 0) as pri_n
            from k left join cur using (game_id, fld_team) left join pri using (game_id, fld_team)
        )
        select j.game_id, j.fld_team, j.catcher_id, cur_n, pri_n,
               (case when catcher_id is null then 0.0
                     else {RUNS_PER_POINT} * 100.0 * (cur_rc + pri_rc) / ({n_den} + pri_n + {K_SHRINK}) end)::double as frm_runs_opp,
               (catcher_id is null) as frm_missing
        from j
    """)
    con.execute(f"""
        create or replace table frm_feat as
        select q.game_id, q.bat_team, coalesce(f.frm_runs_opp, 0.0)::double as frm_runs_opp,
               coalesce(f.frm_missing, true) as frm_missing
        from {keys} q left join frm_feat_game f on f.game_id = q.game_id and f.fld_team = q.fld_team
    """)
    notes = con.execute("""
        select count(*) as games, avg(frm_runs_opp) as mean, stddev(frm_runs_opp) as sd,
               sum(frm_missing::int) as missing
        from frm_feat_game
    """).fetchdf().iloc[0].to_dict()
    if notes["missing"]:
        ids = con.execute("select game_id from frm_feat_game where frm_missing order by 1 limit 10").fetchdf().game_id.tolist()
        print(f"warning: {int(notes['missing'])} team-game(s) have no lineup catcher; frm_runs_opp set to 0 (first ids: {ids})",
              file=sys.stderr)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in notes.items()}
