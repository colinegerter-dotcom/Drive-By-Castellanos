"""Offline tests for resumed games, venue ids and the roof check (24 Sep 2026).

No network. The 746942 entries copy the live schedule response fetched
24 Sep 2026: TOR at BOS started 26 Jun 2024 and finished 26 Aug 2024, and
the feed lists it twice under one gamePk.

Run: python tests/test_resumed_and_venues.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.games.game_conditions import is_roofed  # noqa: E402
from pipelines.games.resumed_games import resumed_game_rows, venue_ids_by_game  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


def e(pk, official, game_date, state="Final", coded="F", venue=3, **extra):
    d = {"gamePk": pk, "officialDate": official, "gameDate": game_date,
         "status": {"detailedState": state, "codedGameState": coded},
         "venue": {"id": venue, "name": f"V{venue}"}}
    d.update(extra)
    return d


SCHED = [
    e(746942, "2024-06-26", "2024-06-26T23:10:00Z", resumeDate="2024-08-26T18:05:00Z", resumeGameDate="2024-08-26"),
    e(746942, "2024-06-26", "2024-08-26T18:05:00Z", resumedFrom="2024-06-26T23:10:00Z", resumedFromDate="2024-06-26"),
    e(745180, "2024-05-21", "2024-05-21T23:45:00Z", resumeGameDate="2024-05-22"),  # only the original listing
    e(700001, "2024-04-02", "2024-04-02T23:05:00Z"),  # normal game
    # only the resumed-part listing, resuming at 7:10 pm Pacific (02:10 UTC the next day)
    e(700004, "2024-07-01", "2024-07-03T02:10:00Z", resumedFromDate="2024-07-01"),
    e(700002, "2024-04-03", "2024-04-03T23:05:00Z", state="Cancelled", coded="C", resumeGameDate="2024-04-04"),
    # postponed at venue 5, played later at venue 7
    e(700003, "2024-04-10", "2024-04-09T23:05:00Z", state="Postponed", coded="D", venue=5),
    e(700003, "2024-04-10", "2024-04-10T17:05:00Z", venue=7),
]

print("\nresumed games")
rows = resumed_game_rows(SCHED)
by_id = {r["game_id"]: r for r in rows}
check("three resumed games found", sorted(by_id), [700004, 745180, 746942])
check("746942 original date", by_id[746942]["original_date"], "2024-06-26")
check("746942 resume date", by_id[746942]["resume_date"], "2024-08-26")
check("one row per gamePk", len(rows), 3)
check("resume date is the local date, not UTC", by_id[700004]["resume_date"], "2024-07-02")
check("original-only listing still works", by_id[745180]["resume_date"], "2024-05-22")
check("cancelled game ignored", 700002 in by_id, False)
check("normal game ignored", 700001 in by_id, False)

print("\nvenue ids")
v = venue_ids_by_game(SCHED)
check("normal game venue", v[700001], 3)
check("postponed listing never wins", v[700003], 7)
check("resumed game keeps its start venue", v[746942], 3)

print("\nroof check")
check("Astros by id, old name", is_roofed("Minute Maid Park", 2392), True)
check("Astros by old name alone", is_roofed("Minute Maid Park"), True)
check("Astros by new name alone", is_roofed("Daikin Park"), True)
check("Brewers by id under a name we never listed", is_roofed("Some Future Name", 32), True)
check("open-air park", is_roofed("Wrigley Field", 17), False)
check("unknown name, no id", is_roofed("Nowhere Park"), False)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all resumed/venue checks passed")
