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


def ensure_players_exist(conn, player_ids, known_ids: set[int] | None = None) -> set[int]:
    """Make sure every id in player_ids has a row in mlb.players, fetching and
    inserting any that don't. Returns the set of ids now known to exist.

    Why this is needed on top of the roster pulls: confirmed live twice now
    (17-18 Sep 2026), MLB's `fullSeason` roster type does not return every
    player who actually appears in a box score. The first case broke
    games.home_starter_id's foreign key outright; the second showed up as
    lineup rows being silently dropped for a couple of players (668904,
    506702) on every one of their games -- ~100 lineup rows lost, plus the
    row-by-row retry in db.upsert_rows firing on nearly every chunk, which
    cost more wall-clock time than it saved data.

    Callers pass `known_ids` (the ids they've already confirmed) so this
    doesn't re-query the players table on every chunk.
    """
    from pipelines.db import upsert_rows  # local import: db imports nothing from here, keeps the cycle impossible

    if known_ids is None:
        with conn.cursor() as cur:
            cur.execute("select player_id from mlb.players")
            known_ids = {r[0] for r in cur.fetchall()}

    missing = {pid for pid in player_ids if pid is not None} - known_ids
    if not missing:
        return known_ids

    log.warning(
        "%d player(s) appear in game data but weren't in mlb.players -- fetching them now: %s",
        len(missing),
        sorted(missing),
    )
    rows = build_player_rows(sorted(missing), {})
    upsert_rows(conn, "players", rows, conflict_cols=["player_id"])
    return known_ids | {r["player_id"] for r in rows}


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
