"""Offline tests for the bullpen availability rules (24 Sep 2026).

No network, no database: a fake connection answers the recent-games query
and the box-score cache is primed with fake feeds.

The old flag was TRUE whenever the team had played 2+ games in 3 days
(95.6% of rows). The new rules, from 2022-2025 usage: a reliever is
unavailable if he pitched on each of the last two days, or threw 25+
pitches yesterday.

Run: python tests/test_bullpen_flags.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.games import bullpen_status as bp  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


print("\nreliever_availability")
D1, D2, D3 = "2025-06-09", "2025-06-08", "2025-06-07"
r = bp.reliever_availability(
    {
        1: {D1: 15, D2: 12},   # back to back -> out
        2: {D1: 30},           # heavy yesterday -> out
        3: {D1: 24},           # light yesterday -> available
        4: {D2: 20, D3: 20},   # not yesterday -> available
        5: {D1: 25, D2: 10},   # both rules
    },
    D1, D2,
)
check("back to back list", r["back_to_back"], [1, 5])
check("unavailable list", r["unavailable"], [1, 2, 5])
check("nobody pitched", bp.reliever_availability({}, D1, D2), {"back_to_back": [], "unavailable": []})


# --- full row, with a fake connection and primed box scores -----------------
TEAM, OPP = 147, 111
AS_OF = "2025-06-10"
# (game_id, date, {pitcher_id: (pitches, saves)}); pitcher 50 is the starter each game
GAMES = [
    (1, "2025-05-20", {50: (95, 0), 61: (15, 1), 62: (12, 0)}),
    (2, "2025-05-21", {50: (90, 0), 61: (14, 1)}),
    (3, "2025-06-07", {50: (88, 0), 63: (20, 0)}),
    (4, "2025-06-08", {50: (92, 0), 61: (18, 1), 62: (10, 0)}),
    (5, "2025-06-09", {50: (85, 0), 61: (16, 0), 64: (27, 0), 63: (8, 0)}),
]


class FakeCursor:
    def __init__(self):
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, query, params):
        as_of = date.fromisoformat(params["as_of_date"])
        lo = as_of - timedelta(days=int(params["days"]))
        self.rows = sorted(
            [(gid, date.fromisoformat(d)) for gid, d, _ in GAMES if lo <= date.fromisoformat(d) < as_of],
            key=lambda x: x[1], reverse=True,
        )

    def fetchall(self):
        return self.rows


class FakeConn:
    def cursor(self):
        return FakeCursor()


def feed(lines):
    players = {
        f"ID{pid}": {"person": {"id": pid}, "stats": {"pitching": {
            "numberOfPitches": p, "earnedRuns": 0, "outs": 3, "saves": s}}}
        for pid, (p, s) in lines.items()
    }
    return {
        "gameData": {"teams": {"home": {"id": TEAM}, "away": {"id": OPP}}},
        "liveData": {"boxscore": {"teams": {"home": {"players": players}, "away": {"players": {}}}}},
    }


bp.clear_cache()
for gid, _, lines in GAMES:
    bp.prime_cache(gid, feed(lines))
starters = {(gid, TEAM): 50 for gid, _, _ in GAMES}

print("\nfull row: closer 61 pitched on each of the last two days")
row = bp.build_bullpen_status_row(FakeConn(), TEAM, 99, AS_OF, 2025, starters)
check("any back to back", row["back_to_back_appearances"], True)
check("count back to back", row["relievers_back_to_back"], 1)
check("unavailable ids (61 b2b, 64 heavy)", row["unavailable_reliever_ids"], [61, 64])
check("closer (most saves = 61) unavailable", row["closer_available_flag"], False)
check("starter never counted", 50 in row["unavailable_reliever_ids"], False)
check("pitches last 3 days exclude the starter", row["pitches_thrown_last_3d"], 20 + 18 + 10 + 16 + 27 + 8)

print("\nfull row: a day later, nobody went back to back")
GAMES.append((6, "2025-06-10", {50: (80, 0), 62: (9, 0)}))
bp.prime_cache(6, feed(GAMES[-1][2]))
starters[(6, TEAM)] = 50
row = bp.build_bullpen_status_row(FakeConn(), TEAM, 100, "2025-06-11", 2025, starters)
check("no back to back", row["back_to_back_appearances"], False)
check("count zero", row["relievers_back_to_back"], 0)
check("nobody unavailable", row["unavailable_reliever_ids"], [])
check("closer available", row["closer_available_flag"], True)

print("\nclosed-off-day case: closer pitched 3 days ago only")
row = bp.build_bullpen_status_row(FakeConn(), TEAM, 101, "2025-06-12", 2025, starters)
check("closer available after a day off", row["closer_available_flag"], True)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all bullpen checks passed")
