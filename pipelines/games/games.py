"""
games table -- MLB Stats API schedule endpoint.

Everything here is the PREGAME view: probable starters (not who actually
pitched -- see game_results.py for that), scheduled first pitch, and the
broadcast list used to derive national_tv_flag. Re-running this for a date
already in the past is still safe (idempotent upsert) but pointless --
you'd want game_results.py for anything that's already been played.
"""
from __future__ import annotations

import logging

from pipelines.config import NATIONAL_TV_NETWORKS
from pipelines.mlb_stats_client import get_schedule

log = logging.getLogger(__name__)


def _is_national_broadcast(broadcasts: list[dict]) -> bool:
    """Classify a game as nationally televised from its broadcast list.

    The Stats API's broadcast objects aren't 100% consistent about exposing
    an explicit "national" flag across seasons, so this checks a couple of
    plausible field shapes first and falls back to matching the broadcaster
    name against our curated NATIONAL_TV_NETWORKS list (config.py). Treat
    this as the piece most worth spot-checking once we can hit the live API
    -- if MLB's payload shape differs from what's assumed here, only this
    function needs to change.
    """
    for b in broadcasts:
        if b.get("isNational") is True:
            return True
        if str(b.get("type", "")).upper() == "NATIONAL":
            return True
        name = (b.get("name") or "").strip()
        if name in NATIONAL_TV_NETWORKS:
            return True
    return False


def build_game_rows(start_date: str, end_date: str, season: int | None = None) -> list[dict]:
    games = get_schedule(start_date, end_date, season=season)
    rows = []
    for g in games:
        # Skip spring training / exhibition / all-star game rows -- gameType
        # "R" = regular season, postseason types are D/F/L/W (etc). We keep
        # postseason in (schema explicitly wants game_type stored so it can
        # be modeled or excluded later), just drop preseason/exhibition (S/E).
        game_type = g.get("gameType")
        if game_type in ("S", "E", "A"):  # spring, exhibition, all-star
            continue

        teams = g.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})

        rows.append(
            {
                "game_id": g["gamePk"],
                "date": g.get("officialDate"),
                "season": int(g.get("season", season)),
                "home_team": (home.get("team") or {}).get("id"),
                "away_team": (away.get("team") or {}).get("id"),
                "home_starter_id": (home.get("probablePitcher") or {}).get("id"),
                "away_starter_id": (away.get("probablePitcher") or {}).get("id"),
                "first_pitch_time": g.get("gameDate"),  # ISO8601 UTC string; Postgres casts to timestamptz
                "venue": (g.get("venue") or {}).get("name"),
                "day_night": g.get("dayNight"),
                "doubleheader_flag": g.get("doubleHeader") not in (None, "N"),
                "national_tv_flag": _is_national_broadcast(g.get("broadcasts", [])),
                "game_type": game_type,
                # umpire_id intentionally left unset here -- MLB doesn't reliably
                # publish the plate umpire on the pregame schedule. game_results.py
                # fills it in from the box score's "officials" list once available.
            }
        )
    log.info("built %d game rows for %s..%s", len(rows), start_date, end_date)
    return rows
