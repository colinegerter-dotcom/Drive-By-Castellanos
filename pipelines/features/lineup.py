"""
Lineup features for the batting team.

  lineup_woba     park-neutral blended wOBA of the 9 starters, weighted by
                  the plate appearances each batting-order slot gets in the
                  segment (the leadoff man bats more than the 9th hitter)
  lineup_k_bb     same weighting, blended strikeout rate minus walk rate
  lineup_platoon  share of weighted plate appearances with the platoon
                  advantage against the opposing starter (a lefty batter vs
                  a righty pitcher, or the reverse; switch hitters always
                  have it). Worth about +.014 wOBA per advantaged PA

Two versions of the lineup:
  actual     the confirmed starting 9 (P2, the bet decision point)
  projected  who is likely to start, from the team's last 14 games (P1);
             see projected_lineup()
"""
from __future__ import annotations

import duckdb

from . import priors

FIT_SEASON = 2021


def slot_weights(con: duckdb.DuckDBPyConnection) -> None:
    """Average plate appearances per batting-order slot per team-game, in
    innings 1-5 and 1-8, measured on the warm-up season. A plate
    appearance's slot is its position in the team's batting order cycle,
    which stays right even after substitutions."""
    con.execute(f"""
        create or replace table slot_w as
        with p as (
            select game_id, bat_team, inning,
                   (row_number() over (partition by game_id, bat_team order by at_bat_id) - 1) % 9 + 1 as slot
            from pa where season = {FIT_SEASON} and game_type = 'R'
        ), n as (select count(distinct (game_id, bat_team)) n from p)
        select slot,
               count(*) filter (where inning <= 5)::double / any_value(n.n) as w_f5,
               count(*) filter (where inning <= 8)::double / any_value(n.n) as w_8
        from p, n group by slot order by slot
    """)


def lineup_features(con: duckdb.DuckDBPyConnection, lineup_tbl: str, keys: str, out: str) -> None:
    """lineup_tbl: (game_id, team_id, slot, player_id, bats, weight_share)
    where weight_share is 1 for an actual lineup and the start chance for a
    projected one. keys: (game_id, season, cutoff, bat_team, sp_id)."""
    con.execute(f"""
        create or replace table lu_keys as
        select distinct l.player_id, q.season, q.cutoff
        from {lineup_tbl} l join {keys} q on q.game_id = l.game_id and q.bat_team = l.team_id
    """)
    priors.hitter_rates(con, "lu_keys", "lu_rates")
    con.execute(f"""
        create or replace table {out} as
        select q.game_id, q.bat_team,
               sum(w.w_f5 * l.weight_share * r.woba) / sum(w.w_f5 * l.weight_share) as lineup_woba_f5,
               sum(w.w_8 * l.weight_share * r.woba) / sum(w.w_8 * l.weight_share) as lineup_woba_8,
               sum(w.w_f5 * l.weight_share * (r.k_rate - r.bb_rate)) / sum(w.w_f5 * l.weight_share) as lineup_k_bb_f5,
               sum(w.w_8 * l.weight_share * (r.k_rate - r.bb_rate)) / sum(w.w_8 * l.weight_share) as lineup_k_bb_8,
               sum(w.w_8 * l.weight_share * (case when coalesce(l.bats, pb.bats) = 'S' then 1
                                                 when coalesce(l.bats, pb.bats) <> sp.throws then 1 else 0 end))
                 / sum(w.w_8 * l.weight_share) as lineup_platoon,
               sum(l.weight_share * r.rookie::int) as lineup_rookies,
               sum(l.weight_share) as lineup_n
        from {keys} q
        join {lineup_tbl} l on l.game_id = q.game_id and l.team_id = q.bat_team
        join slot_w w on w.slot = l.slot
        join lu_rates r on r.player_id = l.player_id and r.season = q.season and r.cutoff = q.cutoff
        left join players pb on pb.player_id = l.player_id
        left join players sp on sp.player_id = q.sp_id
        group by q.game_id, q.bat_team
    """)


PROJ_WINDOW = 14  # team games looked back over for a projected lineup (design 5.3)


def projected_lineup(con: duckdb.DuckDBPyConnection, keys: str) -> None:
    """Projected lineup for P1 (10am, lineups not posted yet), design 5.3.

    For every hitter who started for the team in its last 14 games that were
    final before the cutoff: his chance of starting today = the share of
    those games against a starter of today's throwing hand that he started
    (all 14 games if the team faced no starter of that hand). He is placed
    in his most common batting slot in those games. The lineup features then
    weight each hitter by start chance x his slot's plate appearances.

    The 14 games can reach back into last season (opening day), which is
    fine: they are all before the cutoff.
    """
    con.execute("""
        create or replace table team_game_seq as
        with tgm as (
            select distinct l.game_id, l.team_id, g.data_date,
                   ps.throws as opp_sp_throws
            from lineup l join games g using (game_id)
            left join players ps on ps.player_id =
                 case when l.team_id = g.home_team then g.away_sp else g.home_sp end
            where g.game_type in ('R', 'F', 'D', 'L', 'W')
        )
        select *, row_number() over (partition by team_id order by data_date, game_id) as seq
        from tgm
    """)
    con.execute(f"""
        create or replace table lineup_proj as
        with q as (
            select q.game_id, q.bat_team as team_id, q.cutoff, ps.throws as sp_throws
            from {keys} q left join players ps on ps.player_id = q.sp_id
        ),
        seq_by_day as (
            -- one row per team and day with the day's highest sequence
            -- number, so the as-of match below can never land on a tie
            -- (doubleheaders share a day)
            select team_id, data_date, max(seq) as seq from team_game_seq group by 1, 2
        ),
        last_seq as (
            select q.*, s.seq as last_seq
            from q asof left join seq_by_day s on s.team_id = q.team_id and q.cutoff > s.data_date
        ),
        win as (
            select l.game_id, l.team_id, l.sp_throws, s.game_id as past_game, s.opp_sp_throws
            from last_seq l join team_game_seq s on s.team_id = l.team_id
                 and s.seq between l.last_seq - {PROJ_WINDOW - 1} and l.last_seq
        ),
        hand as (
            select game_id, team_id,
                   count(*) filter (where opp_sp_throws = sp_throws) as n_same
            from win group by 1, 2
        ),
        use as (
            select w.* from win w join hand h using (game_id, team_id)
            where h.n_same = 0 or w.opp_sp_throws = w.sp_throws
        ),
        denom as (select game_id, team_id, count(*) as n_games from use group by 1, 2),
        by_slot as (
            select u.game_id, u.team_id, lu.player_id, lu.slot, count(*) as n,
                   max(nullif(lu.bats, '')) as bats
            from use u join lineup lu on lu.game_id = u.past_game and lu.team_id = u.team_id
            where lu.slot between 1 and 9
            group by 1, 2, 3, 4
        ),
        starts as (
            -- most common slot; ties go to the higher spot in the order, so
            -- the result never depends on row order (deterministic builds)
            select game_id, team_id, player_id, sum(n) as n_started,
                   arg_max(slot, n * 100 - slot) as usual_slot, max(bats) as bats
            from by_slot group by 1, 2, 3
        )
        select s.game_id, s.team_id, s.usual_slot as slot, s.player_id, s.bats,
               s.n_started::double / d.n_games as weight_share
        from starts s join denom d using (game_id, team_id)
    """)


def actual_lineup(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        create or replace table lineup_actual as
        select game_id, team_id, slot, player_id, nullif(bats, '') as bats, 1.0 as weight_share
        from lineup where slot between 1 and 9
    """)
