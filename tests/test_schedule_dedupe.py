"""Offline tests for schedule de-duplication (22 Sep 2026).

No network. Fixtures mirror the live response for gamePk 632226, fetched
22 Sep 2026: MLB lists a postponed game twice under the same gamePk, and
the postponed entry keeps the stale probable starters. Without dedupe,
the form build wrote pitcher-form rows for those stale probables -- 99
orphan rows across 60 games in 2021.

Run: python tests/test_schedule_dedupe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipelines.games.games as gm  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


def entry(pk, game_date, coded, home_p, away_p, official="2021-04-13"):
    return {
        "gamePk": pk,
        "officialDate": official,
        "gameDate": game_date,
        "gameType": "R",
        "status": {"codedGameState": coded},
        "teams": {
            "home": {"team": {"id": 1}, "probablePitcher": {"id": home_p}},
            "away": {"team": {"id": 2}, "probablePitcher": {"id": away_p}},
        },
        "venue": {"name": "Somewhere"},
        "doubleHeader": "Y",
    }


# The live case, verbatim ids.
POSTPONED = entry(632226, "2021-04-12T23:10:00Z", "D", 656849, 502624)
FINAL = entry(632226, "2021-04-13T20:15:00Z", "F", 573186, 605400)
OTHER = entry(999, "2021-04-13T17:00:00Z", "F", 11, 22)

print("\nthe live 632226 case")
out = gm.dedupe_schedule_entries([POSTPONED, FINAL, OTHER])
check("one entry per gamePk", sorted(g["gamePk"] for g in out), [999, 632226])
kept = next(g for g in out if g["gamePk"] == 632226)
check("postponed entry dropped", kept["status"]["codedGameState"], "F")
check("real home starter kept", kept["teams"]["home"]["probablePitcher"]["id"], 573186)
check("stale home probable gone", kept["teams"]["home"]["probablePitcher"]["id"] != 656849, True)

print("\norder of arrival doesn't matter")
out = gm.dedupe_schedule_entries([FINAL, POSTPONED])
check("still keeps the Final entry", out[0]["status"]["codedGameState"], "F")

print("\na lone postponed game is kept (behaviour unchanged)")
lone = entry(555, "2021-05-01T23:00:00Z", "D", 1, 2)
out = gm.dedupe_schedule_entries([lone])
check("single postponed entry survives", len(out), 1)

print("\ntwo non-postponed entries: keep the EARLIEST (no lookahead)")
# e.g. a suspended game listed on its start and resume dates. The starters
# pitched on the start date, so dating the game there is the safe choice.
start = entry(777, "2021-06-01T23:00:00Z", "T", 10, 20, official="2021-06-01")
resume = entry(777, "2021-06-15T17:00:00Z", "F", 10, 20, official="2021-06-15")
out = gm.dedupe_schedule_entries([resume, start])
check("earliest gameDate kept", out[0]["gameDate"], "2021-06-01T23:00:00Z")

print("\nbuild_game_rows produces unique game_ids end to end")
gm.get_schedule = lambda s, e, season=None: [POSTPONED, FINAL, OTHER]
rows = gm.build_game_rows("2021-04-13", "2021-04-13", season=2021)
ids = [r["game_id"] for r in rows]
check("no duplicate game_ids", len(ids), len(set(ids)))
row = next(r for r in rows if r["game_id"] == 632226)
check("row carries the real probables", (row["home_starter_id"], row["away_starter_id"]), (573186, 605400))

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all schedule dedupe tests passed")
