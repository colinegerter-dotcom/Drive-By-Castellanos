"""
resumed_games table -- games suspended on one date and finished on another.

Why it exists (24 Sep 2026, model design item A4): MLB files every pitch of
a resumed game under its ORIGINAL date. Game 746942 (TOR at BOS) started
26 Jun 2024 and finished 26 Aug 2024, so Nick Pivetta's 93 August pitches
showed up in his "last start" numbers on 28 June, two months early. Any
feature window that trusts the stored date leaks those later pitches.

The fix lives in whatever reads pitches by date: pitches from a game in
this table are treated as happening on resume_date. This module only
records which games those are. 24 games across 2021-2026: 20 finished
within two days, three much later (142 and 65 days in 2021, 61 days in
2024).

Source: the schedule feed lists a suspended game twice under one gamePk,
once on the original date carrying `resumeGameDate`, once on the resume
date carrying `resumedFromDate`. Either entry is enough.
"""
from __future__ import annotations

import logging
from datetime import datetime

from pipelines.config import CENTRAL

log = logging.getLogger(__name__)


def _central_date(iso_utc: str | None) -> str | None:
    """'2024-08-27T02:10:00Z' -> '2024-08-26'. gameDate is UTC, so a late
    West Coast start would otherwise land on the next day."""
    if not iso_utc:
        return None
    try:
        return datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(CENTRAL).date().isoformat()
    except ValueError:
        return iso_utc[:10]

RESUMED_KEY = ["game_id"]


def resumed_game_rows(schedule_entries: list[dict]) -> list[dict]:
    """One row per resumed gamePk from raw (unhydrated) schedule entries."""
    out: dict[int, dict] = {}
    for g in schedule_entries:
        pk = g.get("gamePk")
        status = (g.get("status") or {}).get("detailedState")
        if pk is None or status in ("Cancelled", "Postponed"):
            continue
        original = g.get("resumedFromDate") or (g.get("officialDate") if g.get("resumeGameDate") else None)
        resume = g.get("resumeGameDate")
        if resume is None and g.get("resumedFromDate"):
            resume = _central_date(g.get("gameDate"))
        if not (original and resume):
            continue
        row = out.setdefault(pk, {"game_id": pk, "original_date": original, "resume_date": resume, "status": status})
        # Prefer the entry for the resumed part: its status is the final one.
        if g.get("resumedFromDate"):
            row["status"] = status
    rows = sorted(out.values(), key=lambda r: (r["original_date"], r["game_id"]))
    return rows


def venue_ids_by_game(schedule_entries: list[dict]) -> dict[int, int]:
    """gamePk -> venue id, choosing the same schedule entry games.py keeps
    (dedupe_schedule_entries), so venue_id always matches games.venue."""
    from pipelines.games.games import dedupe_schedule_entries

    out: dict[int, int] = {}
    for g in dedupe_schedule_entries(schedule_entries):
        vid = (g.get("venue") or {}).get("id")
        if g.get("gamePk") is not None and vid is not None:
            out[g["gamePk"]] = vid
    return out
