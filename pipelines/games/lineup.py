"""
lineup table -- for historical (completed) games, built from the ACTUAL
box score lineup, never a pregame posting -- per the schema doc's
probable-vs-actual rule. For a game that's still upcoming, a separate
live_lineup pull (not built yet -- see roadmap) would read the pregame
posted lineup instead and simply get overwritten by this same function
once the game is final.

playing_through_injury_flag is deliberately NEVER set by this module. It's
the one manual field on this table (per the schema doc). Leaving the key
out of every row dict means our upsert (pipelines/db.py) will populate it
as NULL on a brand-new row and, critically, will NOT touch it on a
re-upsert of an existing row -- so re-running this pull can never clobber
a flag Colin set by hand.
"""
from __future__ import annotations

from pipelines.mlb_stats_client import get_live_feed


def build_lineup_rows(game_pk: int) -> list[dict]:
    feed = get_live_feed(game_pk)
    boxscore = feed.get("liveData", {}).get("boxscore", {})
    game_data = feed.get("gameData", {})
    team_ids = {
        "home": ((game_data.get("teams") or {}).get("home") or {}).get("id"),
        "away": ((game_data.get("teams") or {}).get("away") or {}).get("id"),
    }

    rows = []
    for side in ("home", "away"):
        team_box = boxscore.get("teams", {}).get(side, {})
        team_id = team_ids[side]
        batting_order = team_box.get("battingOrder", [])  # starters only, in order, as player ids
        players = team_box.get("players", {})

        slot_by_player = {pid: idx + 1 for idx, pid in enumerate(batting_order)}

        for key, entry in players.items():
            person = entry.get("person") or {}
            player_id = person.get("id")
            if player_id is None:
                continue
            slot = slot_by_player.get(player_id)
            if slot is None:
                # Not in the starting batting order (bench/bullpen who didn't
                # start) -- lineup.py only covers the starting lineup, per
                # the schema's roster-facts intent. Skip.
                continue
            rows.append(
                {
                    "game_id": game_pk,
                    "team_id": team_id,
                    "player_id": player_id,
                    "batting_order_slot": slot,
                    "defensive_position": (entry.get("position") or {}).get("abbreviation"),
                    "bats_hand": (person.get("batSide") or {}).get("code"),
                    # playing_through_injury_flag intentionally omitted -- see module docstring
                }
            )
    return rows
