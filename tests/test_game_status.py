"""Offline tests for game status mapping (23 Sep 2026).

No network. The live feed labels rain-shortened official games
"Completed Early: Rain" / "Completed Early: Wet Grounds" (verified live on
gamePk 778370). The old exact-text match on "Completed Early" skipped them
silently: 27 games across 2021-2025 got no game_results row.

Run: python tests/test_game_status.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipelines.games.game_results as gr  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


print("\nthe bug: reason suffixes")
check("Completed Early: Rain", gr._map_status("Completed Early: Rain", "Final"), "completed")
check("Completed Early: Wet Grounds", gr._map_status("Completed Early: Wet Grounds", "Final"), "completed")
check("Final: Tied", gr._map_status("Final: Tied", "Final"), "completed")

print("\nexisting behaviour unchanged")
check("Final", gr._map_status("Final", "Final"), "completed")
check("Game Over", gr._map_status("Game Over", "Final"), "completed")
check("Completed Early", gr._map_status("Completed Early", "Final"), "completed")
check("Postponed", gr._map_status("Postponed", "Final"), "postponed")
check("Suspended: Rain", gr._map_status("Suspended: Rain", "Live"), "suspended")
check("In Progress -> None", gr._map_status("In Progress", "Live"), None)
check("Scheduled -> None", gr._map_status("Scheduled", "Preview"), None)

print("\ncancelled games are never 'completed'")
check("Cancelled", gr._map_status("Cancelled", "Final"), None)
check("Cancelled: Rain", gr._map_status("Cancelled: Rain", "Final"), None)

print("\nan unknown label on a Final game warns instead of vanishing")
seen = []


class Catch(logging.Handler):
    def emit(self, record):
        seen.append(record)


h = Catch()
gr.log.addHandler(h)
check("unknown Final label -> None", gr._map_status("Some New Label", "Final"), None)
check("...and a WARNING was logged", any(r.levelno == logging.WARNING for r in seen), True)
seen.clear()
gr._map_status("Warmup", "Preview")
check("no warning for a game that simply isn't over", any(r.levelno == logging.WARNING for r in seen), False)
gr.log.removeHandler(h)

print("\nend to end: the 778370 shape builds a row")
inn = [{"num": n, "home": {"runs": 0}, "away": {"runs": 1 if n <= 3 else 0}} for n in range(1, 7)]
gr.get_live_feed = lambda pk: {
    "gameData": {
        "status": {"abstractGameState": "Final", "detailedState": "Completed Early: Rain", "codedGameState": "F"},
        "teams": {"home": {"id": 147}, "away": {"id": 1}},
    },
    "liveData": {
        "linescore": {"teams": {"home": {"runs": 1}, "away": {"runs": 9}}, "innings": inn},
        "boxscore": {"teams": {"home": {"pitchers": [11]}, "away": {"pitchers": [22]}}, "officials": []},
    },
}
row, _ = gr.build_game_result_row(778370)
check("row is written", row is not None, True)
check("status completed", row and row["game_status"], "completed")
check("innings recorded", row and row["innings_played"], 6)
check("final score", row and (row["home_score_final"], row["away_score_final"]), (1, 9))

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all game status tests passed")
