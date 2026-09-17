"""
team_form table.

Deliberately computed from OUR OWN already-ingested games/game_results
tables via SQL, not by re-pulling standings from the MLB API. Two reasons:
1. It's the only way to guarantee the no-lookahead rule mechanically --
   every query below has an explicit `date < as_of_date` filter, so it's
   structurally impossible to leak a result from the game being predicted
   or anything after it.
2. It means team_form is only ever as fresh as our own game_results data,
   which is exactly the dependency we want (this module MUST run after
   games.py + game_results.py have ingested the relevant date range).

def_oaa_season (team defense) is the one field NOT computed from our own
tables -- it comes from Savant's team OAA leaderboard (savant_client.py).
See that module's docstring: whether Savant's date-bounding actually works
is unverified (couldn't reach the live API from this environment). Until
verified, treat def_oaa_season as a soft/unverified column.

games_back_playoff and clinched_or_eliminated_flag use the standard
division-games-back formula and a simplified elimination heuristic
(remaining games < games back). Real MLB elimination numbers also account
for head-to-head tiebreakers in some cases -- this is a reasonable
approximation, not exact BBWAA elimination-number math.
"""
from __future__ import annotations

import logging

from pipelines.savant_client import get_team_outs_above_average

log = logging.getLogger(__name__)

GAMES_PER_SEASON = 162


def _team_record_as_of(conn, team_id: int, as_of_date: str, season: int, last_n: int | None = None):
    """Wins/losses (and optionally just the last N games) for a team,
    strictly before as_of_date. Returns (wins, losses, run_diff).
    """
    query = """
        select
            gr.game_status,
            case when g.home_team = %(team_id)s then g.home_team else g.away_team end as this_team,
            case when g.home_team = %(team_id)s then gr.home_score_final else gr.away_score_final end as team_score,
            case when g.home_team = %(team_id)s then gr.away_score_final else gr.home_score_final end as opp_score,
            gr.winning_team,
            g.date
        from mlb.games g
        join mlb.game_results gr on gr.game_id = g.game_id
        where (g.home_team = %(team_id)s or g.away_team = %(team_id)s)
          and g.season = %(season)s
          and g.date < %(as_of_date)s
          and gr.game_status = 'completed'
        order by g.date desc
    """
    with conn.cursor() as cur:
        cur.execute(query, {"team_id": team_id, "season": season, "as_of_date": as_of_date})
        results = cur.fetchall()

    if last_n:
        results = results[:last_n]

    wins = sum(1 for r in results if r[4] == team_id)
    losses = sum(1 for r in results if r[4] is not None and r[4] != team_id)
    run_diff = sum((r[2] or 0) - (r[3] or 0) for r in results)
    return wins, losses, run_diff, len(results)


def _division_standings_as_of(conn, division: str, as_of_date: str, season: int) -> list[tuple[int, int, int]]:
    """[(team_id, wins, losses), ...] for every team in a division, as of
    the day before a game. Used for games-back math.
    """
    with conn.cursor() as cur:
        cur.execute("select team_id from mlb.teams where division = %s", (division,))
        team_ids = [r[0] for r in cur.fetchall()]
    out = []
    for tid in team_ids:
        wins, losses, _, _ = _team_record_as_of(conn, tid, as_of_date, season)
        out.append((tid, wins, losses))
    return out


def build_team_form_row(conn, team_id: int, game_id: int, game_date: str, season: int, division: str) -> dict:
    wins10, losses10, run_diff10, n_games10 = _team_record_as_of(conn, team_id, game_date, season, last_n=10)
    record_last_10 = f"{wins10}-{losses10}" if n_games10 else None

    standings = _division_standings_as_of(conn, division, game_date, season)
    my_record = next((w, l) for (tid, w, l) in standings if tid == team_id)
    leader = max(standings, key=lambda t: (t[1] - t[2]))  # best win-loss differential
    games_back = ((leader[1] - my_record[0]) + (my_record[1] - leader[2])) / 2.0

    games_played = my_record[0] + my_record[1]
    games_remaining = max(GAMES_PER_SEASON - games_played, 0)
    # Simplified heuristic, not an exact elimination-number calc -- see module docstring.
    clinched_or_eliminated = games_back > games_remaining or (leader[0] == team_id and games_back == 0 and games_remaining < 1)

    def_oaa = None
    try:
        oaa_df = get_team_outs_above_average(season, through_date=game_date)
        # Column name for the team identifier/value is unverified -- see
        # savant_client.get_team_outs_above_average docstring.
        if "team_id" in oaa_df.columns and "outs_above_average" in oaa_df.columns:
            match = oaa_df[oaa_df["team_id"] == team_id]
            if not match.empty:
                def_oaa = float(match.iloc[0]["outs_above_average"])
    except Exception:
        log.exception("failed to pull team OAA for team=%s season=%s as_of=%s", team_id, season, game_date)

    return {
        "team_id": team_id,
        "game_id": game_id,
        "record_last_10": record_last_10,
        "run_diff_last_10": run_diff10 if n_games10 else None,
        "games_back_playoff": round(games_back, 1),
        "clinched_or_eliminated_flag": clinched_or_eliminated,
        "def_oaa_season": def_oaa,
        "travel_fatigue_score": None,  # see roadmap: needs schedule/venue-distance logic, not yet built
    }
