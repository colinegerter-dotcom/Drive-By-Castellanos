"""
Shared SQL building blocks for the player-performance tables. Everything
here takes an explicit `as_of_date` and filters strictly BEFORE it
(`g.date < as_of_date`) -- this is the mechanical enforcement of the
schema's #1 rule: no rolling/season stat may see the game it's being
attached to, or anything after it.

These query mlb.pitches joined to mlb.games (for the date filter), so they
require pitches to already be ingested for the lookback window before
being called -- see the backfill/daily-pull ordering in scripts/.
"""
from __future__ import annotations


def pitcher_events_query(days: int | None) -> str:
    """days=None means "this season" (no lower bound beyond season start,
    which the season/date filter already handles via g.season).
    """
    date_filter = ""
    if days is not None:
        date_filter = "and g.date >= (%(as_of_date)s::date - (%(days)s || ' days')::interval)"
    return f"""
        select p.pitch_result, p.events, p.bb_type, p.release_speed, p.game_id, g.date
        from mlb.pitches p
        join mlb.games g on g.game_id = p.game_id
        where p.pitcher_id = %(player_id)s
          and g.season = %(season)s
          and g.date < %(as_of_date)s
          {date_filter}
    """


def batter_events_query(days: int | None) -> str:
    date_filter = ""
    if days is not None:
        date_filter = "and g.date >= (%(as_of_date)s::date - (%(days)s || ' days')::interval)"
    return f"""
        select p.pitch_result, p.events, p.exit_velocity, p.launch_angle, p.game_id, g.date
        from mlb.pitches p
        join mlb.games g on g.game_id = p.game_id
        where p.batter_id = %(player_id)s
          and g.season = %(season)s
          and g.date < %(as_of_date)s
          {date_filter}
    """
