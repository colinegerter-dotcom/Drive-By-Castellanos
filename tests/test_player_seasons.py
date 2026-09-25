"""Offline tests for official season lines and birth dates (25 Sep 2026).

No network. The fixture is real /people yearByYear data fetched from the
Stats API on 25 Sep 2026 (trimmed to the fields used), for two players
traded mid-season: Juan Soto (WSH to SD, 2022) and Max Scherzer (WSH to LAD,
2021). The API returns each team stint AND a combined line; adding them all
up double-counts the season.

Run: python tests/test_player_seasons.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.reference.player_seasons import birth_date, player_season_rows, season_lines  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


PEOPLE = json.loads(r'''[{"id": 665742, "birthDate": "1998-10-25", "stats": [{"group": {"displayName": "hitting"}, "type": {"displayName": "yearByYear"}, "splits": [{"season": "2021", "team": {"id": 120}, "sport": {"id": 1}, "gameType": "R", "stat": {"age": 22, "gamesPlayed": 151, "plateAppearances": 654, "homeRuns": 29, "baseOnBalls": 145, "strikeOuts": 93, "groundOuts": 160, "airOuts": 97, "hitByPitch": 2}}, {"season": "2022", "team": {"id": 120}, "sport": {"id": 1}, "gameType": "R", "stat": {"age": 23, "gamesPlayed": 101, "plateAppearances": 436, "homeRuns": 21, "baseOnBalls": 91, "strikeOuts": 62, "groundOuts": 109, "airOuts": 87, "hitByPitch": 3}}, {"season": "2022", "team": {"id": 135}, "sport": {"id": 1}, "gameType": "R", "stat": {"age": 23, "gamesPlayed": 52, "plateAppearances": 228, "homeRuns": 6, "baseOnBalls": 44, "strikeOuts": 34, "groundOuts": 53, "airOuts": 52, "hitByPitch": 1}}, {"season": "2022", "sport": {"id": 1}, "numTeams": 2, "gameType": "R", "stat": {"age": 23, "gamesPlayed": 153, "plateAppearances": 664, "homeRuns": 27, "baseOnBalls": 135, "strikeOuts": 96, "groundOuts": 162, "airOuts": 139, "hitByPitch": 4}}]}]}, {"id": 453286, "birthDate": "1984-07-27", "stats": [{"group": {"displayName": "pitching"}, "type": {"displayName": "yearByYear"}, "splits": [{"season": "2021", "team": {"id": 120}, "sport": {"id": 1}, "gameType": "R", "stat": {"age": 36, "gamesPlayed": 19, "battersFaced": 428, "outs": 333, "homeRuns": 18, "baseOnBalls": 28, "strikeOuts": 147, "groundOuts": 62, "airOuts": 112, "hitByPitch": 8, "hitBatsmen": 8}}, {"season": "2021", "team": {"id": 119}, "sport": {"id": 1}, "gameType": "R", "stat": {"age": 36, "gamesPlayed": 11, "battersFaced": 265, "outs": 205, "homeRuns": 5, "baseOnBalls": 8, "strikeOuts": 89, "groundOuts": 46, "airOuts": 72, "hitByPitch": 2, "hitBatsmen": 2}}, {"season": "2021", "sport": {"id": 1}, "numTeams": 2, "gameType": "R", "stat": {"age": 36, "gamesPlayed": 30, "battersFaced": 693, "outs": 538, "homeRuns": 23, "baseOnBalls": 36, "strikeOuts": 236, "groundOuts": 108, "airOuts": 184, "hitByPitch": 10, "hitBatsmen": 10}}, {"season": "2022", "team": {"id": 121}, "sport": {"id": 1}, "gameType": "R", "stat": {"age": 37, "gamesPlayed": 23, "battersFaced": 565, "outs": 436, "homeRuns": 13, "baseOnBalls": 24, "strikeOuts": 173, "groundOuts": 83, "airOuts": 166, "hitByPitch": 11, "hitBatsmen": 11}}]}, {"group": {"displayName": "hitting"}, "type": {"displayName": "yearByYear"}, "splits": [{"season": "2021", "team": {"id": 120}, "sport": {"id": 1}, "gameType": "R", "stat": {"age": 36, "gamesPlayed": 19, "plateAppearances": 37, "homeRuns": 0, "baseOnBalls": 0, "strikeOuts": 13, "groundOuts": 20, "airOuts": 4, "hitByPitch": 0}}, {"season": "2021", "team": {"id": 119}, "sport": {"id": 1}, "gameType": "R", "stat": {"age": 36, "gamesPlayed": 11, "plateAppearances": 26, "homeRuns": 0, "baseOnBalls": 0, "strikeOuts": 15, "groundOuts": 7, "airOuts": 4, "hitByPitch": 0}}, {"season": "2021", "sport": {"id": 1}, "numTeams": 2, "gameType": "R", "stat": {"age": 36, "gamesPlayed": 30, "plateAppearances": 63, "homeRuns": 0, "baseOnBalls": 0, "strikeOuts": 28, "groundOuts": 27, "airOuts": 8, "hitByPitch": 0}}]}]}]''')
SOTO, SCHERZER = PEOPLE[0], PEOPLE[1]
if SOTO["id"] != 665742:
    SOTO, SCHERZER = SCHERZER, SOTO

rows = {(r["player_id"], r["season"], r["stat_group"]): r for p in PEOPLE for r in player_season_rows(p)}

print("\ntraded players use MLB's combined line, not the stints added up")
check("Soto 2022 PA", rows[(665742, 2022, "hitting")]["plate_appearances"], 664)
check("Soto 2022 teams", rows[(665742, 2022, "hitting")]["num_teams"], 2)
check("Scherzer 2021 batters faced", rows[(453286, 2021, "pitching")]["batters_faced"], 693)
check("Scherzer 2021 outs", rows[(453286, 2021, "pitching")]["outs"], 538)
check("one row per player, season and group", sum(1 for k in rows if k[:2] == (453286, 2021)), 2)

print("\nsingle-team seasons")
check("Soto 2021 PA", rows[(665742, 2021, "hitting")]["plate_appearances"], 654)
check("Scherzer 2022 teams", rows[(453286, 2022, "pitching")]["num_teams"], 1)

print("\nfield mapping")
r = rows[(453286, 2021, "pitching")]
check("pitcher HBP from hitBatsmen", r["hit_by_pitch"], 10)
check("strikeouts", r["strikeouts"], 236)
check("ground outs", r["ground_outs"], 108)
check("fields not in the data stay None", r["earned_runs"], None)
check("raw stat kept", r["stat_json"]["battersFaced"], 693)
check("Soto walks", rows[(665742, 2022, "hitting")]["walks"], 135)

print("\nminor league and postseason lines are ignored")
lines = season_lines([
    {"season": "2019", "team": {"id": 1}, "sport": {"id": 11}, "gameType": "R", "stat": {"plateAppearances": 400}},
    {"season": "2019", "team": {"id": 2}, "sport": {"id": 1}, "gameType": "R", "stat": {"plateAppearances": 50}},
    {"season": "2019", "team": {"id": 2}, "sport": {"id": 1}, "gameType": "P", "stat": {"plateAppearances": 9}},
])
check("only the MLB regular-season line", lines[2019][0]["plateAppearances"], 50)

print("\ncareer totals no longer double-count a traded season")
import pipelines.mlb_stats_client as msc  # noqa: E402
real_get = msc._get
msc._get = lambda url, params=None: {"people": [SCHERZER]}
tot = msc.get_career_totals_before_season([453286], "pitching", 2022)
msc._get = real_get
check("Scherzer batters faced before 2022 (2021 only in fixture)", tot[453286]["battersFaced"], 693)

print("\nbirth dates")
check("Soto", birth_date(SOTO), "1998-10-25")
check("missing", birth_date({"id": 1}), None)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all season line checks passed")
