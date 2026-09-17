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

import logging

from pipelines.mlb_stats_client import get_roster, get_people

log = logging.getLogger(__name__)


def collect_player_ids_for_season(team_ids: list[int], season: int, extra_ids: set[int] | None = None) -> list[int]:
    """extra_ids: player ids known from OTHER sources (e.g. probable starters
    pulled off the schedule -- see backfill.py) to fold in alongside whatever
    the roster pulls return.

    Confirmed live (17 Sep 2026, first real backfill): MLB's `rosterType:
    fullSeason` does NOT reliably return every player who appears elsewhere
    in a season's data (a real, active starter -- not some replacement-level
    fringe case -- was missing from every team's fullSeason roster pull,
    which then broke games.home_starter_id's foreign key). Root cause
    unconfirmed (a mid-season trade/DFA edge in how the Stats API's
    "fullSeason" roster type is scoped is the leading guess), so rather than
    trust roster pulls as complete, callers are expected to also pass in any
    player id they already know is referenced elsewhere.
    """
    ids: set[int] = set()
    for team_id in team_ids:
        for entry in get_roster(team_id, season):
            person = entry.get("person") or {}
            if person.get("id"):
                ids.add(person["id"])
    if extra_ids:
        missing = extra_ids - ids
        if missing:
            log.warning(
                "[%s] %d player id(s) referenced elsewhere weren't in any team's roster pull -- adding them directly: %s",
                season, len(missing), sorted(missing),
            )
        ids |= extra_ids
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
