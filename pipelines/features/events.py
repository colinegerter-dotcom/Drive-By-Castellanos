"""
Pitches -> plate appearances, and pitches -> pitcher appearances.

Statcast marks the outcome on the last pitch of each plate appearance (the
`events` column). Almost every feature is a rate per plate appearance, so the
first step is one row per plate appearance.
"""
from __future__ import annotations

import duckdb

# Outcomes that end the at-bat without a real plate appearance (a runner is
# caught stealing for the third out, say). They count for nothing.
NOT_PA = ("truncated_pa",)
# Sacrifice bunts are plate appearances but not part of wOBA or strikeout
# rate denominators in the standard definitions; they're kept as PAs and
# flagged so each rate can decide.


def build_pa(con: duckdb.DuckDBPyConnection) -> None:
    """Create table `pa`: one row per plate appearance.

    Columns: game_id, season, date, data_date, game_type, at_bat_id, inning,
    bat_team, fld_team, batter_id, pitcher_id, and outcome flags k, bb (walk
    excluding intentional), ibb, hbp, hr, sac_bunt, in_play (ball in play,
    not a home run), gb (ground ball), fb (fly ball or popup), ld (line
    drive), woba_value, woba_denom, times_through (1 = first time this
    batter faced this pitcher in the game), pa_of_pitcher (1, 2, 3... for
    the pitcher's batters faced in the game).
    """
    con.execute(f"""
        create or replace table pa as
        with last_pitch as (
            -- the pitch that carries the outcome; if Statcast marks two
            -- pitches in one at-bat (rare), take the later one
            select * from pitches
            where events is not null and events not in {NOT_PA}
            qualify row_number() over (partition by game_id, at_bat_id order by pitch_number desc) = 1
        )
        select game_id, season, date, data_date, game_type, at_bat_id, inning,
               bat_team, fld_team, batter_id, pitcher_id,
               (events in ('strikeout', 'strikeout_double_play'))::int as k,
               (events = 'walk')::int as bb,
               (events = 'intent_walk')::int as ibb,
               (events = 'hit_by_pitch')::int as hbp,
               (events = 'home_run')::int as hr,
               (events = 'sac_bunt')::int as sac_bunt,
               (bb_type is not null and events <> 'home_run')::int as in_play,
               (bb_type = 'ground_ball')::int as gb,
               (bb_type in ('fly_ball', 'popup'))::int as fb,
               (bb_type = 'line_drive')::int as ld,
               coalesce(woba_value, 0) as woba_value,
               coalesce(woba_denom, 0) as woba_denom,
               row_number() over (partition by game_id, batter_id, pitcher_id order by at_bat_id) as times_through,
               row_number() over (partition by game_id, pitcher_id order by at_bat_id) as pa_of_pitcher
        from last_pitch
    """)


def build_appearances(con: duckdb.DuckDBPyConnection) -> None:
    """Create table `app`: one row per pitcher per game.

    started: he threw the game's first pitch for his team. bf: batters
    faced. pitches: pitch count. first_inning / last_inning: where he
    pitched. late_close_bf: batters faced in the 7th inning or later with
    the score within 3 runs (the design's stand-in for high leverage, since
    the data has no leverage index). early_relief_bf: batters faced in
    innings 1-5 when not the starter.
    """
    con.execute("""
        create or replace table app as
        with p as (
            select game_id, season, date, data_date, game_type, fld_team as team_id,
                   pitcher_id, at_bat_id, pitch_number, inning, events,
                   abs(coalesce(bat_score, 0) - coalesce(fld_score, 0)) as margin
            from pitches
        ), firsts as (
            select game_id, team_id, arg_min(pitcher_id, at_bat_id * 1000 + pitch_number) as starter_id
            from p group by 1, 2
        )
        select p.game_id, any_value(p.season) season, any_value(p.date) date,
               any_value(p.data_date) data_date, any_value(p.game_type) game_type,
               p.team_id, p.pitcher_id,
               (p.pitcher_id = f.starter_id) as started,
               count(distinct p.at_bat_id) as bf,
               count(*) as pitches,
               min(p.inning) as first_inning, max(p.inning) as last_inning,
               count(distinct case when p.inning >= 7 and p.margin <= 3 then p.at_bat_id end) as late_close_bf,
               count(distinct case when p.inning <= 5 and p.pitcher_id <> f.starter_id then p.at_bat_id end) as early_relief_bf
        from p join firsts f using (game_id, team_id)
        group by p.game_id, p.team_id, p.pitcher_id, f.starter_id
    """)
