"""
players table -- MLB Stats API.

Two-step pull: (1) each team's full-season roster gives us the set of
player_ids that were on a 40-man roster at some point in the season, then
(2) a batched /people call gets the bio fields (bats/throws/debut date)
that the roster endpoint doesn't include.

current_team_id is refreshed every time this runs (not just once), which is
exactly what makes trades show up automatically without a schema change --
that's called out explicitly in the schema doc.
"""
from __future__ import annotations

from pipelines.mlb_stats_client import get_roster, get_people


def collect_player_ids_for_season(team_ids: list[int], season: int) -> list[int]:
    ids: set[int] = set()
    for team_id in team_ids:
        for entry in get_roster(team_id, season):
            person = entry.get("person") or {}
            if person.get("id"):
                ids.add(person["id"])
    return sorted(ids)


def build_player_rows(player_ids: list[int], current_team_by_player: dict[int, int]) -> list[dict]:
    """current_team_by_player: player_id -> team_id, built by the caller from
    the same roster pulls used in collect_player_ids_for_season (so a player
    who was traded mid-season and appears on two rosters gets whichever team
    call happened to be processed -- callers should pass the *latest* known
    team, e.g. by iterating rosters in a stable order and letting later
    writes win).
    """
    people = get_people(player_ids)
    rows = []
    for p in people:
        pid = p["id"]
        rows.append(
            {
                "player_id": pid,
                "full_name": p.get("fullName"),
                "primary_position": (p.get("primaryPosition") or {}).get("abbreviation"),
                "bats": (p.get("batSide") or {}).get("code"),
                "throws": (p.get("pitchHand") or {}).get("code"),
                "debut_date": p.get("mlbDebutDate"),
                "current_team_id": current_team_by_player.get(pid),
            }
        )
    return rows
