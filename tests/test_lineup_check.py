"""Offline tests for the pitch-file lineup cross-check (24 Sep 2026).

No network, no database: a tiny in-memory DuckDB stands in for a season's
pitch file.

Run: python tests/test_lineup_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402

from pipelines.games.lineup_check import compare_lineups, pitch_lineups  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


class Source:
    def __init__(self, con):
        self.con = con

    def query(self, sql, params=None):
        return self.con.execute(sql, params or {})


HOME, AWAY = 1, 2
con = duckdb.connect()
con.execute("create table pitches (game_id bigint, inning_topbot varchar, batter_id int, at_bat_id int)")
rows = []
# game 10: away bats in the top, batters 201-209 in order, then 201 again
for i, b in enumerate(list(range(201, 210)) + [201], start=1):
    rows.append((10, "Top", b, 2 * i - 1))
# home: slot 3 starter 103 was pinch-hit for before batting, so 150 bats third
for i, b in enumerate([101, 102, 150, 104, 105, 106, 107, 108, 109], start=1):
    rows.append((10, "Bot", b, 2 * i))
con.executemany("insert into pitches values (?, ?, ?, ?)", rows)

slots, batted = pitch_lineups(Source(con), {10: (HOME, AWAY)})
check("away slot 1", slots[(10, AWAY)][1], 201)
check("away slot 9", slots[(10, AWAY)][9], 209)
check("home slot 3 is the batter who came up third", slots[(10, HOME)][3], 150)
check("only 9 slots even when the order turns over", len(slots[(10, AWAY)]), 9)

box = [{"game_id": 10, "team_id": AWAY, "player_id": 200 + s, "batting_order_slot": s} for s in range(1, 10)]
box += [{"game_id": 10, "team_id": HOME, "player_id": p, "batting_order_slot": s}
        for s, p in enumerate([101, 102, 103, 104, 105, 106, 107, 108, 109], start=1)]
res = compare_lineups(box, slots, batted)
check("18 slots compared", res["slots_compared"], 18)
check("17 agree", res["slots_agree"], 17)
check("the one miss is a starter who never batted", res["never_batted"], 1)
check("no unexplained misses", res["other"], [])
check("17 of 18 is below the 99.5% bar", res["passed"], False)

print("\nan unexplained disagreement is reported")
box_bad = [dict(r) for r in box]
box_bad[0]["player_id"], box_bad[1]["player_id"] = 202, 201  # slots 1 and 2 swapped
res = compare_lineups(box_bad, slots, batted)
check("two other disagreements", len(res["other"]), 2)

print("\na game with no pitch data is skipped, not failed")
res = compare_lineups(box + [{"game_id": 11, "team_id": HOME, "player_id": 1, "batting_order_slot": 1}], slots, batted)
check("skipped team-games", res["team_games_skipped"], 1)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all lineup cross-check checks passed")
