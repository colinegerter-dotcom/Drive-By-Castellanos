"""
bullpen_status table.

Built from box-score pitching lines over a team's recent games (pulled via
mlb_stats_client.get_live_feed, same source game_results.py uses), not
from the raw pitches table -- box scores already have per-pitcher pitch
counts and earned runs per game, so there's no need to aggregate 700k rows
of pitch-level data just to answer "how many pitches has the bullpen thrown
in 3 days."

PERFORMANCE, NOW FIXED (this was the thing that made a full backfill
impossible): this used to call get_live_feed() once per lookback game, per
team, per game being scored, with no caching. Because the season-long
lookback window grows as the season goes on, the cost per game grew
linearly and the cost per season grew QUADRATICALLY -- measured live on 18
Sep 2026, form-table chunks went from ~9 minutes per 100 games early in
the season to ~36 minutes per 100 games by game 1400, on track for ~15
hours for one season against GitHub's 6-hour job cap. Roughly 400,000 HTTP
fetches for a single season, nearly all of them refetching a box score
already fetched.

_pitching_lines_by_team now caches each game's parsed pitching lines by
game_id (both teams at once, starters included -- the starter is filtered
out per-team at read time, so the cache doesn't depend on who started).
That makes it at most one fetch per game per run: ~2,500 for a season
instead of ~400,000. Only the small parsed lines are kept, not the
multi-MB live-feed payloads, so memory stays trivial; call clear_cache()
between seasons anyway.

AVAILABILITY RULES (rewritten 24 Sep 2026)
The old back_to_back_appearances flag was TRUE whenever the team had played
2+ games in the last 3 days (`len(dates) >= 1` on a non-empty list), so it
was TRUE on 95.6% of rows and never looked at a single reliever. The rules
now work per reliever and come from measured usage, not guesses. Relievers
2022-2025, chance of pitching today given recent use (pitch files):

    pitched each of the last 2 days          5.3%
    pitched yesterday, 25+ pitches           5.7%
    pitched yesterday, under 25 pitches     29.5%
    pitched yesterday and 3 days ago        19.4%
    none of the last 3 days                 39.6%

So a reliever counts as UNAVAILABLE if he pitched on each of the last two
calendar days, or threw HEAVY_PITCHES or more yesterday. Written per row:
  back_to_back_appearances  any reliever pitched on each of the last 2 days
  relievers_back_to_back    how many did
  unavailable_reliever_ids  everyone meeting either rule, sorted
  closer_available_flag     FALSE if the closer is in that list

The model does not read this table (it computes availability from the pitch
files, as a probability rather than a yes/no); these columns are for the
card and for anyone querying the database.

HEURISTIC, FLAGGED: MLB doesn't publish "who is the closer". The closer is
proxied as the reliever with the team's most saves so far this season. A
committee bullpen, or early season with few saves, gives a noisier signal.
Before 24 Sep 2026 he was marked unavailable if he pitched in the team's
most recent game at all, even when that game was 3 days earlier.

KNOWN LIMIT: a resumed game's box score is filed under its ORIGINAL date,
so relievers who pitched in the resumed part count as pitching on that
date. 24 resumed games in 2021-2026 (see mlb.resumed_games).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from pipelines.mlb_stats_client import get_live_feed

log = logging.getLogger(__name__)

# Pitches thrown yesterday at or above which a reliever is treated as
# unavailable today. Relievers at 25+ pitched the next day 5.7% of the time
# vs 29.5% under 25 (2022-2025). See the module docstring.
HEAVY_PITCHES = 25


def _team_recent_game_ids(conn, team_id: int, as_of_date: str, season: int, days: int) -> list[tuple[int, str]]:
    query = """
        select g.game_id, g.date
        from mlb.games g
        join mlb.game_results gr on gr.game_id = g.game_id
        where (g.home_team = %(team_id)s or g.away_team = %(team_id)s)
          and g.season = %(season)s
          and g.date < %(as_of_date)s
          and g.date >= (%(as_of_date)s::date - (%(days)s || ' days')::interval)
          and gr.game_status = 'completed'
        order by g.date desc
    """
    with conn.cursor() as cur:
        cur.execute(query, {"team_id": team_id, "season": season, "as_of_date": as_of_date, "days": days})
        return [(r[0], str(r[1])) for r in cur.fetchall()]


# game_id -> {team_id: [pitching lines, STARTER INCLUDED]}. See the module
# docstring: this cache is the difference between a season backfill taking
# ~15 hours and taking minutes.
_PITCHING_LINES_CACHE: dict[int, dict[int, list[dict]]] = {}


def clear_cache() -> None:
    """Drop the cached box-score pitching lines. Call between seasons in a
    multi-season backfill -- the cache is small per game, but there's no
    reason to hold a finished season's worth of it."""
    _PITCHING_LINES_CACHE.clear()


def prime_cache(game_id: int, feed: dict) -> None:
    """Parse an already-fetched live feed into the cache, so a caller that
    fetched the feed for another reason (the lineup rebuild) doesn't make
    this module fetch it a second time."""
    if game_id not in _PITCHING_LINES_CACHE:
        _pitching_lines_by_team(game_id, feed=feed)


def _pitching_lines_by_team(game_id: int, feed: dict | None = None) -> dict[int, list[dict]]:
    """Every pitcher's line for BOTH teams in one game, parsed once and
    cached. Starters are deliberately kept in here so the cache key is just
    game_id -- callers filter their own team's starter out below."""
    cached = _PITCHING_LINES_CACHE.get(game_id)
    if cached is not None:
        return cached

    feed = feed if feed is not None else get_live_feed(game_id)
    teams_meta = (feed.get("gameData", {}).get("teams") or {})
    box_teams = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})

    by_team: dict[int, list[dict]] = {}
    for side in ("home", "away"):
        team_id = (teams_meta.get(side) or {}).get("id")
        if team_id is None:
            continue
        lines = []
        for entry in ((box_teams.get(side) or {}).get("players") or {}).values():
            pid = (entry.get("person") or {}).get("id")
            if pid is None:
                continue
            pitching = ((entry.get("stats") or {}).get("pitching")) or {}
            if not pitching:
                continue  # this player didn't pitch (position player entry)
            lines.append(
                {
                    "pitcher_id": pid,
                    "pitches_thrown": pitching.get("numberOfPitches", 0),
                    "earned_runs": pitching.get("earnedRuns", 0),
                    "outs": pitching.get("outs", 0),
                    "saves": pitching.get("saves", 0),
                }
            )
        by_team[team_id] = lines

    _PITCHING_LINES_CACHE[game_id] = by_team
    return by_team


def _bullpen_pitching_lines(game_id: int, team_id: int, starter_id: int | None) -> list[dict]:
    """Per-relief-pitcher stat lines for one team in one game, excluding the starter."""
    lines = _pitching_lines_by_team(game_id).get(team_id, [])
    return [line for line in lines if line["pitcher_id"] != starter_id]


def build_bullpen_status_row(
    conn,
    team_id: int,
    game_id: int,
    as_of_date: str,
    season: int,
    starter_lookup: dict[tuple[int, int], int | None],
) -> dict:
    """starter_lookup: {(game_id, team_id): starter_pitcher_id}, covering
    every game in the lookback window for BOTH teams -- keyed by team_id as
    well as game_id because the home and away starter must each be excluded
    only from their own team's bullpen line, not the opponent's.
    """
    recent_3d = _team_recent_game_ids(conn, team_id, as_of_date, season, days=3)
    recent_15d = _team_recent_game_ids(conn, team_id, as_of_date, season, days=15)
    recent_season = _team_recent_game_ids(conn, team_id, as_of_date, season, days=250)

    def aggregate(game_ids_dates):
        total_pitches = 0
        total_er = 0
        total_outs = 0
        save_counts: dict[int, int] = {}
        for gid, _gdate in game_ids_dates:
            lines = _bullpen_pitching_lines(gid, team_id, starter_lookup.get((gid, team_id)))
            for line in lines:
                total_pitches += line["pitches_thrown"]
                total_er += line["earned_runs"]
                total_outs += line["outs"]
                if line["saves"]:
                    save_counts[line["pitcher_id"]] = save_counts.get(line["pitcher_id"], 0) + line["saves"]
        return total_pitches, total_er, total_outs, save_counts

    pitches_3d, _, _, _ = aggregate(recent_3d)
    pitches_15d, er_15d, outs_15d, _ = aggregate(recent_15d)
    pitches_season, er_season, outs_season, save_counts_season = aggregate(recent_season)

    def era(earned_runs, outs):
        if not outs:
            return None
        innings = outs / 3.0
        return round((earned_runs * 9.0) / innings, 2)

    # Per-reliever pitches by calendar day over the last 3 days. A
    # doubleheader day counts once, with both games' pitches added up.
    day1 = (date.fromisoformat(as_of_date) - timedelta(days=1)).isoformat()
    day2 = (date.fromisoformat(as_of_date) - timedelta(days=2)).isoformat()
    pitches_by_day: dict[int, dict[str, int]] = {}
    for gid, gdate in recent_3d:
        for line in _bullpen_pitching_lines(gid, team_id, starter_lookup.get((gid, team_id))):
            days = pitches_by_day.setdefault(line["pitcher_id"], {})
            days[gdate] = days.get(gdate, 0) + (line["pitches_thrown"] or 0)
    unavailable = reliever_availability(pitches_by_day, day1, day2)

    # Closer proxy: pitcher with the most saves this season so far.
    closer_id = max(save_counts_season, key=save_counts_season.get) if save_counts_season else None
    closer_available = closer_id is None or closer_id not in unavailable["unavailable"]

    return {
        "team_id": team_id,
        "game_id": game_id,
        "pitches_thrown_last_3d": pitches_3d,
        "back_to_back_appearances": bool(unavailable["back_to_back"]),
        "relievers_back_to_back": len(unavailable["back_to_back"]),
        "unavailable_reliever_ids": unavailable["unavailable"],
        "closer_available_flag": closer_available,
        "bullpen_era_last_15d": era(er_15d, outs_15d),
        "bullpen_era_season": era(er_season, outs_season),
    }


def reliever_availability(pitches_by_day: dict[int, dict[str, int]], day1: str, day2: str) -> dict[str, list[int]]:
    """Apply the availability rules to per-reliever daily pitch counts.

    pitches_by_day: {pitcher_id: {"YYYY-MM-DD": pitches}} for the team's
    relievers over recent days. day1 = yesterday, day2 = the day before.
    Returns sorted id lists: "back_to_back" (pitched on both days) and
    "unavailable" (back to back, or HEAVY_PITCHES+ yesterday).
    """
    b2b = sorted(pid for pid, days in pitches_by_day.items() if day1 in days and day2 in days)
    heavy = {pid for pid, days in pitches_by_day.items() if days.get(day1, 0) >= HEAVY_PITCHES}
    return {"back_to_back": b2b, "unavailable": sorted(set(b2b) | heavy)}
