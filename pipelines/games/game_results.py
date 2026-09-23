"""
game_results table -- built from the post-game live-feed/box score, per
game. This is the table that stores what ACTUALLY happened (final score,
F5 score, actual starters, game status) as opposed to games.py's pregame
probable-starter view -- see the schema doc's "probable vs actual" rule.

Also responsible for backfilling games.umpire_id, since the plate umpire
is reliably available in the box score's "officials" list but not on the
pregame schedule.
"""
from __future__ import annotations

import logging

from pipelines.mlb_stats_client import get_live_feed

log = logging.getLogger(__name__)

_STATUS_MAP = {
    # MLB's abstractGameState/detailedState vocabulary is broader than the
    # 4 values our schema's game_status CHECK constraint allows -- this maps
    # every detailedState we're likely to see onto one of the 4.
    "Final": "completed",
    "Game Over": "completed",
    "Completed Early": "completed",
    "Postponed": "postponed",
    "Suspended": "suspended",
    "Suspended: Rain": "suspended",
    "Forfeit": "forfeit",
}


def _map_status(detailed_state: str | None, abstract_state: str | None = None) -> str | None:
    """Map MLB's detailedState onto our 4 allowed game_status values, or None.

    FIXED 23 Sep 2026. "Completed Early" used to be matched only as exact
    text, but MLB appends the reason: "Completed Early: Rain", "Completed
    Early: Wet Grounds". Those fell through to None, the caller logged "not
    final yet" at INFO level, and 27 official rain-shortened games across
    2021-2025 silently got no game_results row. They were filled by hand.
    Same reasoning now applies to "Final: <reason>" variants.

    Also new: if MLB says the game is over (abstractGameState "Final") and we
    still can't map it, that is logged as a WARNING rather than folded into
    the routine "not final yet" message, so the next unfamiliar label shows
    up in the log instead of disappearing.
    """
    if detailed_state in _STATUS_MAP:
        return _STATUS_MAP[detailed_state]
    lowered = (detailed_state or "").lower()
    if "postpon" in lowered:
        return "postponed"
    if "suspend" in lowered:
        return "suspended"
    if "cancel" in lowered:
        # Never played. No result row is correct; say so plainly.
        return None
    if lowered.startswith("completed early") or lowered.startswith("final") or lowered.startswith("game over"):
        return "completed"
    if abstract_state == "Final":
        log.warning(
            "unrecognised detailedState %r on a game MLB marks Final -- no game_results "
            "row written; add it to _STATUS_MAP if it is a real completed game",
            detailed_state,
        )
    # Anything still in-progress or unrecognized: don't force a bad value
    # into a CHECK-constrained column. Caller should skip writing a
    # game_results row at all until the game is actually final.
    return None


def _f5_score(innings: list[dict], side: str) -> int | None:
    """Sum runs for the first 5 innings. `innings` is linescore["innings"];
    each entry looks like {"num": 1, "home": {"runs": 2, ...}, "away": {...}}.
    Returns None (not 0) if we don't even have 5 innings of data yet, so a
    partial/rain-shortened game before the 5th doesn't silently look like
    a 0-0 F5.
    """
    first_five = [i for i in innings if i.get("num", 0) <= 5]
    if len(first_five) < 5:
        return None
    total = 0
    for i in first_five:
        runs = (i.get(side) or {}).get("runs")
        if runs is None:
            return None
        total += runs
    return total


def build_game_result_row(game_pk: int) -> tuple[dict | None, int | None]:
    """Returns (game_results row or None if not final, home_plate_umpire_id or None).

    Returning None for the row when the game isn't final yet is deliberate:
    a script that walks "yesterday's games" and calls this should just skip
    postponed/suspended/still-live games rather than writing garbage.
    """
    feed = get_live_feed(game_pk)
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})
    linescore = live_data.get("linescore", {})
    boxscore = live_data.get("boxscore", {})

    status_obj = game_data.get("status") or {}
    status = _map_status(status_obj.get("detailedState"), status_obj.get("abstractGameState"))

    home_team_id = ((game_data.get("teams") or {}).get("home") or {}).get("id")
    away_team_id = ((game_data.get("teams") or {}).get("away") or {}).get("id")

    ls_teams = linescore.get("teams", {})
    home_final = (ls_teams.get("home") or {}).get("runs")
    away_final = (ls_teams.get("away") or {}).get("runs")

    innings = linescore.get("innings", [])
    home_f5 = _f5_score(innings, "home")
    away_f5 = _f5_score(innings, "away")

    winning_team = None
    if home_final is not None and away_final is not None and home_final != away_final:
        winning_team = home_team_id if home_final > away_final else away_team_id

    box_teams = boxscore.get("teams", {})
    home_pitchers = (box_teams.get("home") or {}).get("pitchers", [])
    away_pitchers = (box_teams.get("away") or {}).get("pitchers", [])
    actual_home_starter = home_pitchers[0] if home_pitchers else None
    actual_away_starter = away_pitchers[0] if away_pitchers else None

    home_plate_umpire_id = None
    for official in boxscore.get("officials", []):
        if official.get("officialType") == "Home Plate":
            home_plate_umpire_id = (official.get("official") or {}).get("id")
            break

    if status is None:
        log.info("game %s not final yet (status=%s), skipping", game_pk, status)
        return None, home_plate_umpire_id

    row = {
        "game_id": game_pk,
        "home_score_final": home_final,
        "away_score_final": away_final,
        "home_score_f5": home_f5,
        "away_score_f5": away_f5,
        "winning_team": winning_team,
        "innings_played": len(innings) if innings else None,
        "game_status": status,
        "actual_home_starter_id": actual_home_starter,
        "actual_away_starter_id": actual_away_starter,
    }
    return row, home_plate_umpire_id


def update_game_umpire(conn, game_id: int, umpire_id: int | None, schema: str = "mlb") -> None:
    if umpire_id is None:
        return
    # Savepoint-wrapped for the same reason as db.upsert_rows's fallback: this
    # is the one write in the pipeline that goes around upsert_rows (it's an
    # UPDATE, not an insert), so an umpire_id whose player row hasn't been
    # seeded yet would otherwise poison the whole shared backfill transaction
    # over one missing umpire.
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT update_umpire")
        try:
            cur.execute(
                f"UPDATE {schema}.games SET umpire_id = %s WHERE game_id = %s",
                (umpire_id, game_id),
            )
            cur.execute("RELEASE SAVEPOINT update_umpire")
        except Exception as exc:
            cur.execute("ROLLBACK TO SAVEPOINT update_umpire")
            log.warning("could not set umpire_id=%s on game=%s (%s: %s) -- leaving it null", umpire_id, game_id, type(exc).__name__, exc)
