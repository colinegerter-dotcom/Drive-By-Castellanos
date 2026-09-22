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


# Widened from 4 attempts / 20s max on 18 Sep 2026: MLB was measurably slower
# that afternoon after a day of heavy backfilling (the postgame stage ran at
# 100 games/min versus 171 earlier), and a burst of 429s or 5xxs that outlasts
# four quick retries would kill a multi-hour job outright. Only transient
# failures are retried -- a 400 still fails fast, since retrying a malformed
# request just wastes the clock.
@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=2, max=45),
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


def _pick_mlb_split(splits: list[dict]) -> dict:
    """Pick the MLB line out of a stats split list.

    Confirmed live (18 Sep 2026): MLB returns the SAME window more than once,
    once per sport -- a sport.id == 1 (MLB) split and a sport.id == 0 ("All")
    roll-up, and for a player who spent part of the window in the minors there
    are additional splits for those sports too. The old per-player code took
    splits[0] blindly, which is right only by luck: for an optioned player or
    someone on a rehab assignment it could return a minor-league or
    minors-inclusive line and quietly pass it off as major-league form. Prefer
    the explicit MLB split, and fall back to the first one only when no split
    carries a sport id at all.
    """
    if not splits:
        return {}
    for split in splits:
        if (split.get("sport") or {}).get("id") == 1:
            return split.get("stat", {})
    return splits[0].get("stat", {})


def get_stats_by_date_range_bulk(
    player_ids: list[int], group: str, start_date: str, end_date: str
) -> dict[int, dict]:
    """byDateRange stats for MANY players in one request, keyed by player_id.

    This exists because the per-player version was the pipeline's dominant
    cost: the form tables asked for 3 windows per batter per game, about 60
    HTTP calls for a single game and ~135,000 for a season, which measured out
    at 20+ minutes per 100 games against GitHub's 6-hour job cap. The
    /people endpoint accepts a list of personIds and will hydrate the same
    byDateRange window onto all of them at once, so a whole lineup costs one
    call instead of eighteen.

    Players with no games in the window simply come back absent from the
    result; callers should treat a missing id as {} (no stats), exactly like
    the single-player function does.
    """
    if not player_ids:
        return {}
    # Same empty-window guard as the single-player call -- see its docstring.
    if start_date > end_date:
        return {}

    out: dict[int, dict] = {}
    CHUNK = 100
    hydrate = (
        f"stats(group=[{group}],type=[byDateRange],"
        f"startDate={start_date},endDate={end_date})"
    )
    ids = sorted({int(p) for p in player_ids if p is not None})
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i : i + CHUNK]
        try:
            data = _get(
                f"{MLB_STATS_API_BASE}/people",
                {"personIds": ",".join(str(p) for p in chunk), "hydrate": hydrate},
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                log.warning(
                    "MLB rejected bulk %s stats for %d players over %s..%s (400) -- those columns stay null",
                    group, len(chunk), start_date, end_date,
                )
                continue
            raise

        for person in data.get("people", []):
            pid = person.get("id")
            if pid is None:
                continue
            for entry in person.get("stats", []):
                # Match BOTH group and type. Only byDateRange is requested, but
                # matching on group alone would silently accept some other
                # stats block if MLB ever returned an extra one -- and a wrong
                # number here looks exactly like a right one downstream.
                if (entry.get("group") or {}).get("displayName") != group:
                    continue
                if (entry.get("type") or {}).get("displayName") != "byDateRange":
                    continue
                stat = _pick_mlb_split(entry.get("splits", []))
                if stat:
                    out[pid] = stat
                break
    return out


def get_career_totals_before_season(player_ids: list[int], group: str, season: int) -> dict[int, dict]:
    """Summed yearByYear totals for every season BEFORE `season`, per player.

    Used to rebuild "career to date" without a per-game API call. Career PA is
    the only career figure the form tables use, and career-to-date at any point
    in a season is just (everything before this season) + (this season so far).
    The first half is a constant for the whole backfill of that season, so it's
    fetched once here instead of once per batter per game -- and it dodges the
    lookahead trap of asking for career totals "as of now", which would include
    games that hadn't been played yet at the point being modelled.

    Only counting stats are summed; rate stats would be meaningless added up
    and are deliberately not returned.
    """
    if not player_ids:
        return {}
    # `outs` rather than `inningsPitched` on purpose: MLB returns innings as a
    # string like "123.2", meaning 123 and two THIRDS, so adding those as
    # decimals is silently wrong. Outs are a plain integer and divide cleanly
    # by 3 back into innings at the point of use.
    SUMMABLE = ("plateAppearances", "atBats", "gamesPlayed", "battersFaced", "outs")
    out: dict[int, dict] = {}
    # Smaller chunk than the byDateRange call on purpose: a yearByYear hydrate
    # returns one split per season PER PLAYER (a veteran can have 15+), so 100
    # players is a much heavier response than 100 players over one window, and
    # this runs against a 30-second request timeout.
    CHUNK = 50
    hydrate = f"stats(group=[{group}],type=[yearByYear])"
    ids = sorted({int(p) for p in player_ids if p is not None})
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i : i + CHUNK]
        try:
            data = _get(
                f"{MLB_STATS_API_BASE}/people",
                {"personIds": ",".join(str(p) for p in chunk), "hydrate": hydrate},
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                log.warning("MLB rejected bulk yearByYear %s stats for %d players (400)", group, len(chunk))
                continue
            raise

        for person in data.get("people", []):
            pid = person.get("id")
            if pid is None:
                continue
            totals: dict[str, int] = {}
            for entry in person.get("stats", []):
                if (entry.get("group") or {}).get("displayName") != group:
                    continue
                for split in entry.get("splits", []):
                    # yearByYear splits carry a season; skip this season and
                    # anything later, and skip non-MLB lines so minor league
                    # plate appearances don't inflate a "career MLB PA" figure.
                    try:
                        split_season = int(split.get("season"))
                    except (TypeError, ValueError):
                        continue
                    if split_season >= season:
                        continue
                    if (split.get("sport") or {}).get("id") not in (1, None):
                        continue
                    stat = split.get("stat", {})
                    for key in SUMMABLE:
                        value = stat.get(key)
                        if isinstance(value, int):
                            totals[key] = totals.get(key, 0) + value
            if totals:
                out[pid] = totals
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

    Returns venues under their CURRENT names. For historical seasons see
    get_venue_names_by_season() -- parks get renamed, and a name-keyed
    lookup against this list silently misses every game at a park whose
    sponsor changed since.
    """
    data = _get(f"{MLB_STATS_API_BASE}/venues", {"hydrate": "location"})
    return data.get("venues", [])


def get_venue_names_by_season(season: int) -> dict[int, str]:
    """venue_id -> that venue's name AS OF `season`.

    Exists because games.venue stores a NAME, captured from the schedule
    feed at the time the game was played, while get_venues() returns names
    as they are today. When a park is renamed, the two stop matching and
    every name-keyed lookup for that park silently falls through.

    Found live on 22 Sep 2026 during the 2021 backfill: the log filled up
    with "no coordinates for venue 'Guaranteed Rate Field'" and
    "'Minute Maid Park'" -- the 2021 names of the parks now called Rate
    Field and Daikin Park -- and weather was skipped for roughly 162 games,
    about 6.7% of the season. Same class of bug as the "UNIQLO Field at
    Dodger Stadium" miss that venue_name_keys() was written for, but caused
    by a rename across seasons rather than a sponsor prefix, so
    venue_name_keys could not catch it.

    VERIFIED LIVE 22 Sep 2026 that the `season` parameter is actually
    honored here, rather than being silently ignored the way Savant ignores
    its date bounds: `/venues?season=2021` returns venue id 4 as
    "Guaranteed Rate Field" and id 2392 as "Minute Maid Park", i.e. the
    2021 names, not today's. Worth re-checking if this ever stops helping.

    `fields` keeps the response small -- this is called once per season per
    build and only the id/name pairs are wanted.
    """
    data = _get(
        f"{MLB_STATS_API_BASE}/venues",
        {"season": season, "sportId": 1, "fields": "venues,id,name"},
    )
    return {
        v["id"]: v["name"]
        for v in data.get("venues", [])
        if v.get("id") is not None and v.get("name")
    }


def get_live_feed(game_pk: int) -> dict:
    """Full live-feed payload for a game -- linescore (for F5 score / final
    score / status) plus box score in one call, so game_results and lineup
    ingestion can share a single fetch instead of hitting the API twice.
    """
    return _get(f"{MLB_STATS_API_BASE_V1_1}/game/{game_pk}/feed/live")
