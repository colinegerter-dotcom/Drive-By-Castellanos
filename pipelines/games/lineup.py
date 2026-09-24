"""
lineup table -- the STARTING lineup for completed games, built from the
box score, never a pregame posting (per the schema doc's probable-vs-actual
rule). A pregame lineup puller for upcoming games is still not built (see
roadmap).

HOW STARTERS ARE IDENTIFIED (fixed 24 Sep 2026)
The box score has two batting-order views:

  teams.<side>.battingOrder   -- a list of 9 player ids, one per slot, holding
                                 whoever occupied that slot at the END of the
                                 game. A pinch hitter or defensive sub
                                 replaces the starter here.
  players.<ID>.battingOrder   -- each player's own code: "300" is the slot-3
                                 starter, "301" the first player to replace
                                 him, "302" the next, and so on.

This module used the first list until 24 Sep 2026, so the table held
end-of-game lineups: 14.2% of 2025 rows were substitutes and 74.7% of
team-games had at least one wrong player (verified on game 777294: the team
list had Romy Gonzalez "302" in slot 3 where Abraham Toro "300" started).
Starters are now the players whose own code ends in "00"; slot = code // 100.

Other fixes in the same change:
  * defensive_position is the FIRST position the starter played
    (allPositions[0]); `position` can be a later move
  * bats_hand comes from gameData.players, because box-score person objects
    carry no batSide -- the column was empty for every row before this

playing_through_injury_flag is deliberately NEVER set by this module. It's
the one manual field on this table (per the schema doc). Callers write rows
through db.replace_game_rows with keep_cols=["playing_through_injury_flag"],
which carries any hand-set value across a rebuild.
"""
from __future__ import annotations

import logging

from pipelines.mlb_stats_client import get_live_feed

log = logging.getLogger(__name__)

LINEUP_KEY = ["game_id", "team_id", "player_id"]
LINEUP_KEEP_COLS = ["playing_through_injury_flag"]


def _starter_slot(code) -> int | None:
    """'300' -> 3; '301' (a substitute) -> None; missing/garbage -> None."""
    try:
        n = int(str(code))
    except (TypeError, ValueError):
        return None
    if n <= 0 or n % 100 != 0:
        return None
    return n // 100


def build_lineup_rows(game_pk: int, feed: dict | None = None) -> list[dict]:
    """Starting lineup rows for both teams in one game.

    feed: an already-fetched live feed for this game, so a caller that also
    needs it for something else (bullpen lines, game results) fetches once.
    """
    feed = feed if feed is not None else get_live_feed(game_pk)
    boxscore = feed.get("liveData", {}).get("boxscore", {})
    game_data = feed.get("gameData", {})
    team_ids = {
        "home": ((game_data.get("teams") or {}).get("home") or {}).get("id"),
        "away": ((game_data.get("teams") or {}).get("away") or {}).get("id"),
    }
    people = game_data.get("players") or {}

    rows = []
    for side in ("home", "away"):
        team_box = boxscore.get("teams", {}).get(side, {})
        team_id = team_ids[side]
        for entry in (team_box.get("players") or {}).values():
            person = entry.get("person") or {}
            player_id = person.get("id")
            if player_id is None:
                continue
            slot = _starter_slot(entry.get("battingOrder"))
            if slot is None:
                continue  # substitute, bench, or a pitcher who never batted
            positions = entry.get("allPositions") or []
            first_pos = (positions[0] or {}).get("abbreviation") if positions else None
            bat_side = ((people.get(f"ID{player_id}") or {}).get("batSide") or {}).get("code")
            rows.append(
                {
                    "game_id": game_pk,
                    "team_id": team_id,
                    "player_id": player_id,
                    "batting_order_slot": slot,
                    "defensive_position": first_pos or (entry.get("position") or {}).get("abbreviation"),
                    "bats_hand": bat_side,
                }
            )
    return rows


def lineup_problems(rows: list[dict], team_ids: list[int]) -> list[str]:
    """Structural checks for one game's starting lineup rows.

    Every team must have exactly 9 starters in slots 1-9, each slot once.
    Returns human-readable problems (empty list = clean). The caller decides
    whether a problem is fatal: the repair script counts them per season and
    fails past a small threshold, so one odd historical box score doesn't
    block a whole season.
    """
    problems = []
    for tid in team_ids:
        slots = sorted(r["batting_order_slot"] for r in rows if r["team_id"] == tid)
        if slots != list(range(1, 10)):
            problems.append(f"team {tid}: starter slots {slots}")
    return problems
