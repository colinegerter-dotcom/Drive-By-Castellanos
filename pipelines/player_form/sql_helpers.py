"""
Shared SQL building blocks for the player-performance tables. Everything
here takes an explicit `as_of_date` and filters strictly BEFORE it
(`g.date < as_of_date`) -- this is the mechanical enforcement of the
schema's #1 rule: no rolling/season stat may see the game it's being
attached to, or anything after it.

DIALECT NOTE (21 Sep 2026)
--------------------------
These queries now run against DuckDB over Parquet rather than Postgres --
see pipelines/pitch_store.py for why. Two things changed and nothing else:

  - Named parameters are `$name` (DuckDB) rather than `%(name)s` (psycopg2).
  - The rolling window's lower bound is passed in as a ready-made date
    (`since_date`) instead of being computed in SQL from a day count.
    Interval arithmetic is the one piece of syntax that differs meaningfully
    between the two engines, and a silently-wrong window here would be a
    lookahead-class bug -- the kind this module exists to prevent. Computing
    it in Python makes it explicit, testable, and identical on both engines.

The `g.date < as_of_date` guard is byte-for-byte what it always was. If you
touch these queries, that comparison is the line to leave alone: it is the
only thing stopping a game's own result from feeding the prediction of that
same game.
"""
from __future__ import annotations

from datetime import date, timedelta


def window_start(as_of_date: str, days: int) -> str:
    """Lower bound for a rolling window, as an ISO date string.

    Inclusive: a 30-day window ending the day before `as_of_date` covers
    [as_of_date - 30, as_of_date). Matches the old
    `g.date >= as_of_date::date - '30 days'::interval` behaviour exactly.
    """
    return (date.fromisoformat(as_of_date) - timedelta(days=days)).isoformat()


def pitcher_events_query(days: int | None) -> str:
    """days=None means "this season to date" (the season filter plus the
    as-of cutoff already bound it); an integer adds a rolling lower bound,
    which the caller supplies as the `since_date` parameter.
    """
    date_filter = "and g.date >= CAST($since_date AS DATE)" if days is not None else ""
    return f"""
        select p.pitch_result, p.events, p.bb_type, p.release_speed, p.game_id, g.date
        from pitches p
        join games g on g.game_id = p.game_id
        where p.pitcher_id = $player_id
          and g.season = $season
          and g.date < CAST($as_of_date AS DATE)
          {date_filter}
    """


def batter_events_query(days: int | None) -> str:
    date_filter = "and g.date >= CAST($since_date AS DATE)" if days is not None else ""
    return f"""
        select p.pitch_result, p.events, p.exit_velocity, p.launch_angle, p.game_id, g.date
        from pitches p
        join games g on g.game_id = p.game_id
        where p.batter_id = $player_id
          and g.season = $season
          and g.date < CAST($as_of_date AS DATE)
          {date_filter}
    """


def umpire_pitches_query() -> str:
    """Every called pitch in the games this umpire has worked this season,
    before the as-of date. Joins through games.umpire_id -- the umpire isn't
    recorded on the pitch itself.
    """
    return """
        select p.pitch_result, p.plate_x, p.plate_z, p.sz_top, p.sz_bot, p.game_id
        from pitches p
        join games g on g.game_id = p.game_id
        where g.umpire_id = $umpire_id
          and g.season = $season
          and g.date < CAST($as_of_date AS DATE)
    """
