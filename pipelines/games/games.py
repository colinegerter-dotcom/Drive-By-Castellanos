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


def dedupe_schedule_entries(games: list[dict]) -> list[dict]:
    """One schedule entry per gamePk.

    MLB's schedule feed lists a postponed game TWICE under the same gamePk:
    once on the original slot, marked postponed and still carrying that
    day's probable starters, and once on the makeup slot with the real ones.
    Verified live 22 Sep 2026 on gamePk 632226 (2021-04-13): the first entry
    is `codedGameState: "D"` / "Postponed" with probables 502624 and 656849,
    the second is `"F"` / "Final" with 605400 and 573186. Both share the
    same officialDate, so a date-based rule can't separate them.

    Before this existed, the 2021 backfill built 2,550 rows for 2,467 real
    games. `games` upserts on game_id so only one survived there, but the
    form build looped over all 2,550 and wrote form rows for the stale
    probables: 99 orphan pitcher-form rows across 60 games in 2021, 36 in
    2025. Not a lookahead leak (they were computed as of the original,
    earlier slot), but a model joining pitcher form on game_id alone would
    have attached the wrong pitcher. Team form and bullpen rows were also
    written twice, with the right one winning only because the feed happens
    to arrive in date order.

    Rule:
    1. Drop postponed entries (codedGameState "D") whenever the same gamePk
       has another entry. A postponed entry that is the ONLY entry is kept,
       so behaviour for a genuinely unplayed game is unchanged.
    2. If more than one entry still remains (e.g. a suspended game listed
       on both its start and resume dates), keep the one with the EARLIEST
       gameDate. That's when first pitch was thrown and the starters
       actually pitched, so dating the game there can never pull later
       games into its form rows. Keeping the latest instead would be a
       lookahead risk.
    """
    by_pk: dict[int, list[dict]] = {}
    order: list[int] = []
    for g in games:
        pk = g.get("gamePk")
        if pk not in by_pk:
            by_pk[pk] = []
            order.append(pk)
        by_pk[pk].append(g)

    out = []
    dropped = 0
    for pk in order:
        entries = by_pk[pk]
        if len(entries) > 1:
            live = [e for e in entries if (e.get("status") or {}).get("codedGameState") != "D"]
            if live:
                entries = live
            if len(entries) > 1:
                entries = sorted(entries, key=lambda e: e.get("gameDate") or "")
                log.info(
                    "gamePk %s still has %d non-postponed schedule entries after dropping "
                    "postponed ones; keeping the earliest (%s)",
                    pk,
                    len(entries),
                    entries[0].get("gameDate"),
                )
        dropped += len(by_pk[pk]) - 1
        out.append(entries[0])

    if dropped:
        log.info("deduped schedule: dropped %d duplicate entries for %d games", dropped, len(out))
    return out


def build_game_rows(start_date: str, end_date: str, season: int | None = None) -> list[dict]:
    games = dedupe_schedule_entries(get_schedule(start_date, end_date, season=season))
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
                # 24 Sep 2026: the stable key for a park. Names change with
                # sponsors (Minute Maid Park -> Daikin Park, Miller Park ->
                # American Family Field); the id doesn't. park_factors.park_id
                # is this same id.
                "venue_id": (g.get("venue") or {}).get("id"),
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
