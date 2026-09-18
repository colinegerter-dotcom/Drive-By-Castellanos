"""
Thin client over statsapi.mlb.com (the official, free, unauthenticated MLB
Stats API). Going with a direct requests wrapper here instead of a wrapper
library (like MLB-StatsAPI on PyPI) on purpose: this pipeline needs specific
`hydrate` parameters (broadcasts, officials, probablePitcher, etc.) that
third-party wrappers don't always expose, and controlling the requests
directly makes it obvious exactly what's being asked for and why.
"""
from __future__ import annotations

import logging
import time

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pipelines.config import MLB_STATS_API_BASE, MLB_STATS_API_BASE_V1_1

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({"User-Agent": "drive-from-castellanos/0.1 (personal research project)"})


class MlbApiError(Exception):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, MlbApiError)),
)
def _get(url: str, params: dict | None = None) -> dict:
    """GET with retry/backoff. The Stats API doesn't publish a rate limit,
    but hammering it during a 5-season backfill is a good way to start
    getting throttled or blocked -- retrying with backoff on transient
    failures (rather than crashing the whole backfill on one hiccup) is
    cheap insurance.
    """
    resp = _session.get(url, params=params, timeout=30)
    if resp.status_code == 429:
        raise MlbApiError(f"rate limited: {url}")
    if resp.status_code >= 500:
        raise MlbApiError(f"server error {resp.status_code}: {url}")
    resp.raise_for_status()
    return resp.json()


def get_teams(season: int) -> list[dict]:
    """All active MLB teams for a season (sportId=1 is the majors)."""
    data = _get(f"{MLB_STATS_API_BASE}/teams", {"sportId": 1, "season": season})
    return data.get("teams", [])


def get_roster(team_id: int, season: int) -> list[dict]:
    """40-man roster for a team/season. Used to seed the players table."""
    data = _get(
        f"{MLB_STATS_API_BASE}/teams/{team_id}/roster",
        {"rosterType": "fullSeason", "season": season},
    )
    return data.get("roster", [])


def get_person(player_id: int) -> dict | None:
    """Full bio for one player (bats/throws/debut date/etc, not on the roster endpoint)."""
    data = _get(f"{MLB_STATS_API_BASE}/people/{player_id}")
    people = data.get("people", [])
    return people[0] if people else None


def get_people(player_ids: list[int]) -> list[dict]:
    """Batch version of get_person -- the API accepts a comma-joined list of
    ids in one call, which is far cheaper than one request per player when
    seeding thousands of players during backfill.
    """
    if not player_ids:
        return []
    out: list[dict] = []
    # The API has a practical limit on how many ids it'll accept in one
    # call; chunk conservatively.
    CHUNK = 100
    for i in range(0, len(player_ids), CHUNK):
        chunk = player_ids[i : i + CHUNK]
        data = _get(f"{MLB_STATS_API_BASE}/people", {"personIds": ",".join(str(p) for p in chunk)})
        out.extend(data.get("people", []))
    return out


def get_schedule(start_date: str, end_date: str, season: int | None = None) -> list[dict]:
    """Schedule for a date range (YYYY-MM-DD), with probable starters,
    broadcasts (for national_tv_flag), and venue hydrated in.

    One call can span an arbitrary date range -- the API paginates by date
    internally and returns a list of "dates", each with its games. We flatten
    that here so callers just get a flat list of game dicts.
    """
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "probablePitcher,broadcasts(all),venue,game(content(summary))",
    }
    if season is not None:
        params["season"] = season
    data = _get(f"{MLB_STATS_API_BASE}/schedule", params)
    games: list[dict] = []
    for d in data.get("dates", []):
        games.extend(d.get("games", []))
    return games


def get_boxscore(game_pk: int) -> dict:
    """Post-game box score: actual starters, officials (umpires), lineups."""
    return _get(f"{MLB_STATS_API_BASE_V1_1}/game/{game_pk}/feed/live").get("liveData", {}).get(
        "boxscore", {}
    )


def get_player_stats_by_date_range(
    player_id: int, group: str, start_date: str, end_date: str
) -> dict:
    """Official, scorer-verified stats (ERA, earned runs, IP, K, BB, HR,
    wOBA inputs, etc) for one player over an exact date range.

    This is the right source for anything that depends on "earned" runs or
    other official-scorer judgment calls (ERA, and therefore FIP's ERA-
    anchored constant) -- those can't be reconstructed correctly from raw
    Statcast pitch events alone (Statcast has no concept of an error or an
    unearned run). group is "pitching" or "hitting". Returns {} if the
    player had no games in range (e.g. hadn't debuted yet, or is on IL).

    EMPTY WINDOW (added 18 Sep 2026, after this crashed a run): callers now
    end their ranges the day BEFORE the game being scored (the lookahead fix
    in the form modules), which means a player making his MLB debut in that
    game gets career_start = his debut date = the game date, and an end date
    one day earlier. MLB answers a backwards range with a 400 and the whole
    backfill dies -- it died on game 1 of 2025, a debut in the Tokyo Series
    opener. A backwards window is not an error, it's the correct statement
    that the player has no prior games, so return no stats without calling
    the API at all. ISO dates compare correctly as strings.
    """
    if start_date > end_date:
        return {}

    try:
        data = _get(
            f"{MLB_STATS_API_BASE}/people/{player_id}/stats",
            {"stats": "byDateRange", "group": group, "startDate": start_date, "endDate": end_date},
        )
    except requests.HTTPError as exc:
        # A 400 here means MLB rejected this particular player/range combination,
        # not that the API is down (5xx and rate limits are retried above, and
        # still raise). One unusable stat line should leave one set of columns
        # null, not kill a multi-hour backfill -- but it's logged every time so
        # a systemic break shows up as a wall of warnings rather than silence.
        if exc.response is not None and exc.response.status_code == 400:
            log.warning(
                "MLB rejected %s stats for player %s over %s..%s (400) -- leaving those columns null",
                group, player_id, start_date, end_date,
            )
            return {}
        raise
    stats_list = data.get("stats", [])
    if not stats_list:
        return {}
    splits = stats_list[0].get("splits", [])
    if not splits:
        return {}
    return splits[0].get("stat", {})


def get_venues() -> list[dict]:
    """All MLB venues, hydrated with lat/long -- used to seed park lookups
    (park_factors.park_id and the coordinates game_conditions.py needs to
    query Open-Meteo).
    """
    data = _get(f"{MLB_STATS_API_BASE}/venues", {"hydrate": "location"})
    return data.get("venues", [])


def get_live_feed(game_pk: int) -> dict:
    """Full live-feed payload for a game -- linescore (for F5 score / final
    score / status) plus box score in one call, so game_results and lineup
    ingestion can share a single fetch instead of hitting the API twice.
    """
    return _get(f"{MLB_STATS_API_BASE_V1_1}/game/{game_pk}/feed/live")
