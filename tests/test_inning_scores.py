"""Offline tests for runs by half inning (25 Sep 2026).

No network, no database: a tiny in-memory DuckDB stands in for a season's
pitch file. Each "pitch" carries the batting team's score before it, which
is all the method uses.

Run: python tests/test_inning_scores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402

from pipelines.games.inning_scores import build_inning_rows, compare_first_five, half_starts  # noqa: E402

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
con.execute("""create table pitches (game_id bigint, inning int, inning_topbot varchar,
               at_bat_id int, pitch_number int, away_score int, home_score int, outs int)""")
rows = []
ab = 0


def half(game_id, inning, tb, away, home, pitches=3, first_outs=0):
    """One half inning: a few pitches, all showing the score before them.
    The first pitch has `first_outs` outs (0 unless simulating a gap)."""
    global ab
    ab += 1
    for pn in range(pitches, 0, -1):  # inserted out of order on purpose
        rows.append((game_id, inning, tb, ab, pn, away, home, first_outs if pn == 1 else 1))


# game 10: away scores 2 in the 1st and 1 in the 7th; home 1 in the 3rd and
# walks off with 3 in the 9th (final 3-4). Home runs per inning: 0,0,1,0,...,3
a, h = 0, 0
for inn in range(1, 10):
    half(10, inn, "Top", a, h)
    a += {1: 2, 7: 1}.get(inn, 0)
    half(10, inn, "Bot", a, h)
    h += {3: 1}.get(inn, 0)
# (the walk-off 3 runs in the bottom 9th only show up in the final score)

# game 11: home leads after 8.5, bottom 9th never played. Final away 1, home 5
a, h = 0, 0
for inn in range(1, 10):
    half(11, inn, "Top", a, h)
    a += {4: 1}.get(inn, 0)
    if inn == 9:
        break
    half(11, inn, "Bot", a, h)
    h += {2: 3, 6: 2}.get(inn, 0)

# game 12: the pitch file is missing the top of the 4th
a, h = 0, 0
for inn in range(1, 10):
    if inn != 4:
        half(12, inn, "Top", a, h)
    half(12, inn, "Bot", a, h)

# game 14: pitch data for the home team stops after the 6th
for inn in range(1, 10):
    half(14, inn, "Top", 0, 0)
    if inn <= 6:
        half(14, inn, "Bot", 0, 0)

# game 15: the top of the 3rd is missing its leadoff at-bat
for inn in range(1, 10):
    half(15, inn, "Top", 0, 0, first_outs=1 if inn == 3 else 0)
    half(15, inn, "Bot", 0, 0)

# game 16: the home team never appears in the pitch data
for inn in range(1, 10):
    half(16, inn, "Top", 0, 0)

con.executemany("insert into pitches values (?, ?, ?, ?, ?, ?, ?, ?)", rows)
games = {
    10: {"home_team": HOME, "away_team": AWAY, "home_final": 4, "away_final": 3, "home_f5": 1, "away_f5": 2},
    11: {"home_team": HOME, "away_team": AWAY, "home_final": 5, "away_final": 1, "home_f5": 3, "away_f5": 1},
    12: {"home_team": HOME, "away_team": AWAY, "home_final": 0, "away_final": 0, "home_f5": 0, "away_f5": 0},
    13: {"home_team": HOME, "away_team": AWAY, "home_final": 2, "away_final": 1, "home_f5": 1, "away_f5": 0},  # no pitches
    14: {"home_team": HOME, "away_team": AWAY, "home_final": 0, "away_final": 0, "innings_played": 9},
    15: {"home_team": HOME, "away_team": AWAY, "home_final": 0, "away_final": 0},
    16: {"home_team": HOME, "away_team": AWAY, "home_final": 0, "away_final": 0},
}
games[10]["innings_played"] = 9
games[11]["innings_played"] = 9

out, problems = build_inning_rows(half_starts(Source(con)), games)
by = {(r["game_id"], r["inning"], r["half"]): r["runs"] for r in out}

print("\nwalk-off game")
check("away 1st", by[(10, 1, "top")], 2)
check("away 7th", by[(10, 7, "top")], 1)
check("home 3rd", by[(10, 3, "bottom")], 1)
check("home walk-off 9th from the final score", by[(10, 9, "bottom")], 3)
check("home total adds up to final", sum(v for (g, i, hb), v in by.items() if g == 10 and hb == "bottom"), 4)
check("18 half innings", sum(1 for k in by if k[0] == 10), 18)

print("\nskipped bottom of the 9th")
check("no bottom 9th row", (11, 9, "bottom") in by, False)
check("home last half is the 8th", by[(11, 8, "bottom")], 0)
check("away 9th", by[(11, 9, "top")], 0)
check("17 half innings", sum(1 for k in by if k[0] == 11), 17)

print("\nbad or missing data")
check("game with a missing half inning gets no rows", any(k[0] == 12 for k in by), False)
check("game with no pitches gets no rows", any(k[0] == 13 for k in by), False)
check("home data cut off after the 6th is caught", any(k[0] == 14 for k in by), False)
check("half inning missing its leadoff at-bat is caught", any(k[0] == 15 for k in by), False)
check("home team never batting is caught", any(k[0] == 16 for k in by), False)
check("all five listed as problems", sorted(g for g, _ in problems), [12, 13, 14, 15, 16])
check("walk-off and skipped-9th games fit their 9 innings", {10, 11} <= {k[0] for k in by}, True)

print("\nfirst-five cross-check")
res = compare_first_five(out, games)
check("two games compared", res["compared"], 2)
check("both agree", res["agree"], 2)
wrong = dict(games)
wrong[10] = dict(games[10], home_f5=2)
check("a mismatch is caught", compare_first_five(out, wrong)["agree"], 1)

print("\nnightly filter by game")
check("only asked-for games", {g for g, *_ in half_starts(Source(con), [11])}, {11})
check("empty list asks for nothing", half_starts(Source(con), []), [])

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all inning score checks passed")
