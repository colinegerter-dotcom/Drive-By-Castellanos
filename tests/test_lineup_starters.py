"""Offline tests for the starting-lineup parser (24 Sep 2026).

No network. The fixture copies the shape of the live feed for gamePk 777294
(CIN at BOS, 1 Jul 2025), fetched 24 Sep 2026. In that game the team-level
battingOrder list held two substitutes: Romy Gonzalez ("302") in slot 3
where Abraham Toro ("300") started, and Santiago Espinal ("901") in slot 9
where Christian Encarnacion-Strand ("900") started. The old parser stored
the substitutes. The fixed one must store the starters.

Run: python tests/test_lineup_starters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.games.lineup import _starter_slot, build_lineup_rows, lineup_problems  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


def player(pid, code, positions):
    return {
        "person": {"id": pid, "fullName": f"P{pid}"},
        "battingOrder": code,
        "position": {"abbreviation": positions[-1]},
        "allPositions": [{"abbreviation": p} for p in positions],
    }


HOME = 111
AWAY = 113
# home: slot 3 starter 647351 replaced by pinch runner 681987 ("301") then 663853 ("302");
# slot 7 starter 691785 moved from 3B to SS mid-game
home_players = {
    "ID680776": player(680776, "100", ["LF"]),
    "ID701350": player(701350, "200", ["DH"]),
    "ID647351": player(647351, "300", ["1B"]),
    "ID681987": player(681987, "301", ["PR"]),
    "ID663853": player(663853, "302", ["1B"]),
    "ID665966": player(665966, "400", ["C"]),
    "ID677800": player(677800, "500", ["RF"]),
    "ID596115": player(596115, "600", ["SS"]),
    "ID691785": player(691785, "700", ["3B", "SS"]),
    "ID666152": player(666152, "800", ["2B"]),
    "ID678882": player(678882, "900", ["CF"]),
    "ID999001": {"person": {"id": 999001}, "position": {"abbreviation": "P"}},  # pitcher, never batted
}
away_players = {f"ID{9000 + s}": player(9000 + s, f"{s}00", ["X"]) for s in range(1, 10)}
away_players["ID9100"] = player(9100, "901", ["3B"])

FEED = {
    "gameData": {
        "teams": {"home": {"id": HOME}, "away": {"id": AWAY}},
        "players": {
            "ID647351": {"batSide": {"code": "R"}},
            "ID691785": {"batSide": {"code": "L"}},
            "ID678882": {"batSide": {"code": "R"}},
            "ID680776": {"batSide": {"code": "L"}},
        },
    },
    "liveData": {"boxscore": {"teams": {
        "home": {"battingOrder": [680776, 701350, 663853, 665966, 677800, 596115, 691785, 666152, 678882],
                 "players": home_players},
        "away": {"battingOrder": [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9100],
                 "players": away_players},
    }}},
}

print("\nslot codes")
check("300 is slot 3", _starter_slot("300"), 3)
check("302 is a substitute", _starter_slot("302"), None)
check("missing code", _starter_slot(None), None)
check("garbage code", _starter_slot("abc"), None)

rows = build_lineup_rows(777294, feed=FEED)
home = {r["batting_order_slot"]: r for r in rows if r["team_id"] == HOME}
away = {r["batting_order_slot"]: r for r in rows if r["team_id"] == AWAY}

print("\nstarters, not end-of-game occupants")
check("18 rows", len(rows), 18)
check("home slot 3 is the starter Toro", home[3]["player_id"], 647351)
check("sub 663853 not stored", any(r["player_id"] == 663853 for r in rows), False)
check("pinch runner not stored", any(r["player_id"] == 681987 for r in rows), False)
check("away slot 9 is the starter", away[9]["player_id"], 9009)
check("away sub 9100 not stored", any(r["player_id"] == 9100 for r in rows), False)
check("pitcher with no code not stored", any(r["player_id"] == 999001 for r in rows), False)

print("\nposition and batting side")
check("starting position is the first one played", home[7]["defensive_position"], "3B")
check("bats_hand from gameData.players", home[3]["bats_hand"], "R")
check("bats_hand missing stays None", home[2]["bats_hand"], None)

print("\nstructure checks")
check("clean game has no problems", lineup_problems(rows, [HOME, AWAY]), [])
short = [r for r in rows if not (r["team_id"] == HOME and r["batting_order_slot"] == 5)]
check("missing slot is flagged", len(lineup_problems(short, [HOME, AWAY])), 1)
dup = rows + [dict(home[1], player_id=123)]
check("duplicate slot is flagged", len(lineup_problems(dup, [HOME, AWAY])), 1)
check("team with no rows is flagged", len(lineup_problems([], [HOME, AWAY])), 2)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all lineup parser checks passed")
