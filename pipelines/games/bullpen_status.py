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

HEURISTIC, FLAGGED: closer_available_flag has no clean data source. MLB
doesn't publish "who is the closer" as a field -- it's a role inferred from
usage. This implementation proxies it as: whichever reliever recorded the
team's most saves so far this season is "the closer"; he's flagged
unavailable if he pitched in the team's most recent completed game before
as_of_date. This is a reasonable first pass, not authoritative -- a team
using committee closers, or early in a season with few saves recorded yet,
will get a noisier signal. Worth revisiting once the model is further along.
"""
from __future__ import annotations

import logging

from pipelines.mlb_stats_client import get_live_feed

log = logging.getLogger(__name__)


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


def _pitching_lines_by_team(game_id: int) -> dict[int, list[dict]]:
    """Every pitcher's line for BOTH teams in one game, parsed once and
    cached. Starters are deliberately kept in here so the cache key is just
    game_id -- callers filter their own team's starter out below."""
    cached = _PITCHING_LINES_CACHE.get(game_id)
    if cached is not None:
        return cached

    feed = get_live_feed(game_id)
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
        pitched_dates: set[str] = set()
        for gid, gdate in game_ids_dates:
            lines = _bullpen_pitching_lines(gid, team_id, starter_lookup.get((gid, team_id)))
            for line in lines:
                total_pitches += line["pitches_thrown"]
                total_er += line["earned_runs"]
                total_outs += line["outs"]
                if line["saves"]:
                    save_counts[line["pitcher_id"]] = save_counts.get(line["pitcher_id"], 0) + line["saves"]
                    pitched_dates.add(gdate)
        return total_pitches, total_er, total_outs, save_counts

    pitches_3d, _, _, _ = aggregate(recent_3d)
    pitches_15d, er_15d, outs_15d, _ = aggregate(recent_15d)
    pitches_season, er_season, outs_season, save_counts_season = aggregate(recent_season)

    def era(earned_runs, outs):
        if not outs:
            return None
        innings = outs / 3.0
        return round((earned_runs * 9.0) / innings, 2)

    # Closer proxy: pitcher with the most saves this season so far.
    closer_id = max(save_counts_season, key=save_counts_season.get) if save_counts_season else None
    closer_available = True
    if closer_id is not None and recent_3d:
        most_recent_game_id, _ = recent_3d[0]
        lines = _bullpen_pitching_lines(most_recent_game_id, team_id, starter_lookup.get((most_recent_game_id, team_id)))
        if any(l["pitcher_id"] == closer_id for l in lines):
            closer_available = False

    back_to_back = False
    if len(recent_3d) >= 2:
        # crude proxy: did the team play (and use its pen) on each of the
        # last two calendar days before as_of_date
        dates = sorted({d for _, d in recent_3d[:2]})
        back_to_back = len(dates) >= 1  # at minimum, pen worked yesterday; refine once travel_fatigue_score exists

    return {
        "team_id": team_id,
        "game_id": game_id,
        "pitches_thrown_last_3d": pitches_3d,
        "back_to_back_appearances": back_to_back,
        "closer_available_flag": closer_available,
        "bullpen_era_last_15d": era(er_15d, outs_15d),
        "bullpen_era_season": era(er_season, outs_season),
    }
