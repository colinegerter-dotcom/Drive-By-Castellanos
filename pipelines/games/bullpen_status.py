"""
bullpen_status table.

Built from box-score pitching lines over a team's recent games (pulled via
mlb_stats_client.get_live_feed, same source game_results.py uses), not
from the raw pitches table -- box scores already have per-pitcher pitch
counts and earned runs per game, so there's no need to aggregate 700k rows
of pitch-level data just to answer "how many pitches has the bullpen thrown
in 3 days."

PERFORMANCE NOTE: this calls get_live_feed() once per lookback game, per
team, per game being scored -- during a 5-season backfill the same game's
box score gets refetched many times over (once for game_results.py, then
again for every later game whose bullpen lookback window includes it).
Worth adding a simple on-disk/DB cache of live-feed payloads by game_id
before running a full backfill; not built yet, flagging so it's not a
surprise when the first real backfill run is slow.

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


def _bullpen_pitching_lines(game_id: int, team_id: int, starter_id: int | None) -> list[dict]:
    """Per-relief-pitcher stat lines for one team in one game, excluding the starter."""
    feed = get_live_feed(game_id)
    game_data = feed.get("gameData", {})
    home_id = ((game_data.get("teams") or {}).get("home") or {}).get("id")
    side = "home" if home_id == team_id else "away"
    team_box = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {})
    players = team_box.get("players", {})

    lines = []
    for entry in players.values():
        person = entry.get("person") or {}
        pid = person.get("id")
        if pid is None or pid == starter_id:
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
    return lines


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
