"""
Round 2 group 2: park effects by batter hand (design 5.7 item 2, as built:
design E10, approved by Colin 1 Oct 2026 before the build).

The runs park factor treats every batter alike. Some parks play very
differently for left- and right-handed hitters (a short right field helps
left-handed power, for example), so a lineup's mix of hands changes what the
park is worth to it. Two factors per venue and batter side:

  hr    home runs per plate appearance
  hit   singles, doubles and triples per ball in play (home runs excluded)

Method, the same as the runs factor (pipelines/reference/park_factors.py):
the rate for one batter side, both teams batting, in the home team's games
at the venue, divided by what the same home teams' road games give for that
side. Regular season only, prior seasons only (up to 3), so nothing from the
season being predicted is used. Shrunk toward 1:

  factor = 1 + (raw - 1) * n / (n + k)

with n = home plate appearances (hr) or balls in play (hit) for that side,
k = 3,000 and 2,000: roughly one season of one side at one park. Starting
values, not tuned.

Batter side: the player's listed bats; switch hitters bat opposite the
pitcher's arm. The pitch files carry no batter side per pitch.

Features per team-game (the batting team's view), the log of each factor
averaged over the lineup's expected plate appearances (slot weights, the same
lineup table as the other lineup features), each batter's side taken against
the OPPOSING STARTER's arm (the bullpen's arm isn't known in advance):

  pk_hr, pk_hit

A venue with no history (a new park, a one-off neutral site) gets factor 1,
log 0; `new_park` already flags new parks.
"""
from __future__ import annotations

import duckdb

K_HR = 3000      # plate appearances
K_HIT = 2000     # balls in play
LOOKBACK = 3     # prior seasons


def build_park_hand(con: duckdb.DuckDBPyConnection) -> None:
    """Writes table pk_factor (venue_id, season, side, f_hr, f_hit): the
    factors in force for games of `season`, from seasons before it only."""
    # every regular-season plate appearance with venue, home team and batter side
    con.execute("""
        create or replace table pk_pa as
        select p.season, g.venue_id, g.home_team,
               case when p.bat_team = g.home_team then p.fld_team else p.bat_team end as away_team,
               case when coalesce(b.bats, 'R') = 'S'
                    then case when coalesce(t.throws, 'R') = 'L' then 'R' else 'L' end
                    else coalesce(b.bats, 'R') end as side,
               p.hr, p.in_play,
               -- a ball in play with a positive wOBA value is a single, double or triple
               (p.in_play = 1 and p.woba_value > 0)::int as hit_bip
        from pa p
        join games g using (game_id)
        left join players b on b.player_id = p.batter_id
        left join players t on t.player_id = p.pitcher_id
        where p.game_type = 'R' and p.sac_bunt = 0
    """)
    # home side: each (season, venue, home team, side)
    con.execute("""
        create or replace table pk_home as
        select season, venue_id, home_team as team, side,
               count(*) as pa, sum(hr) as hr, sum(in_play) as bip, sum(hit_bip) as hit
        from pk_pa group by 1, 2, 3, 4
    """)
    # road side: the same team's road games, both teams batting
    con.execute("""
        create or replace table pk_road as
        select season, away_team as team, side,
               count(*) as pa, sum(hr) as hr, sum(in_play) as bip, sum(hit_bip) as hit
        from pk_pa group by 1, 2, 3
    """)
    # per target season Y: seasons Y-3..Y-1. Expected events at the venue =
    # home volume x the home team's road rate, summed over seasons and home teams
    con.execute(f"""
        create or replace table pk_factor as
        with targets as (select distinct season + d as target
                         from pk_home, (select unnest(range(1, {LOOKBACK} + 1)) as d)),
        j as (
            select t.target as season, h.venue_id, h.side,
                   h.pa, h.hr, h.bip, h.hit,
                   h.pa * r.hr / nullif(r.pa, 0) as exp_hr,
                   h.bip * r.hit / nullif(r.bip, 0) as exp_hit
            from targets t
            join pk_home h on h.season between t.target - {LOOKBACK} and t.target - 1
            join pk_road r on r.season = h.season and r.team = h.team and r.side = h.side
        ),
        a as (
            select season, venue_id, side,
                   sum(pa) as n_pa, sum(hr) as hr, sum(exp_hr) as exp_hr,
                   sum(bip) as n_bip, sum(hit) as hit, sum(exp_hit) as exp_hit
            from j group by 1, 2, 3
        )
        select season, venue_id, side, n_pa, n_bip,
               1 + (coalesce(hr / nullif(exp_hr, 0), 1) - 1) * n_pa / (n_pa + {K_HR}) as f_hr,
               1 + (coalesce(hit / nullif(exp_hit, 0), 1) - 1) * n_bip / (n_bip + {K_HIT}) as f_hit
        from a
    """)


def park_hand_features(con: duckdb.DuckDBPyConnection, lineup_tbl: str, keys: str = "tg") -> None:
    """Writes table pk_feat (game_id, bat_team, pk_hr, pk_hit). Needs the
    lineup table used for the other lineup features, slot_w, players and
    the keys table (with venue_id and the opposing starter sp_id)."""
    build_park_hand(con)
    con.execute(f"""
        create or replace table pk_feat as
        with lu as (
            select q.game_id, q.bat_team, q.season, q.venue_id,
                   w.w_8 * l.weight_share as wt,
                   case when coalesce(nullif(l.bats, ''), pb.bats, 'R') = 'S'
                        then case when coalesce(ps.throws, 'R') = 'L' then 'R' else 'L' end
                        else coalesce(nullif(l.bats, ''), pb.bats, 'R') end as side
            from {keys} q
            join {lineup_tbl} l on l.game_id = q.game_id and l.team_id = q.bat_team
            join slot_w w on w.slot = l.slot
            left join players pb on pb.player_id = l.player_id
            left join players ps on ps.player_id = q.sp_id
        )
        select lu.game_id, lu.bat_team,
               sum(lu.wt * ln(coalesce(f.f_hr, 1.0))) / sum(lu.wt) as pk_hr,
               sum(lu.wt * ln(coalesce(f.f_hit, 1.0))) / sum(lu.wt) as pk_hit
        from lu
        left join pk_factor f on f.venue_id = lu.venue_id and f.season = lu.season and f.side = lu.side
        group by lu.game_id, lu.bat_team
    """)
