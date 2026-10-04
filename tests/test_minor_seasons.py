"""Offline tests for minor-league season lines (3 Oct 2026).

No network. The fixture is SYNTHETIC: it copies the shape of the real
/people yearByYear response used in test_player_seasons.py and adds minor
league levels. It has not been checked against a live minor-league response
(none could be fetched when this was written); the first dry run of
scripts/build_minor_seasons.py is that check.

Run: python tests/test_minor_seasons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date  # noqa: E402

from pipelines.reference.minor_seasons import LEVELS, debut_coverage, minor_season_lines, minor_season_rows  # noqa: E402

failures = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}" + ("" if ok else f", want {want!r}"))
    if not ok:
        failures.append(label)


def split(season, sport, stat, team=True, num_teams=None, game_type="R"):
    s = {"season": str(season), "sport": {"id": sport}, "gameType": game_type, "stat": stat}
    if team:
        s["team"] = {"id": 1}
    if num_teams:
        s["numTeams"] = num_teams
    return s


PERSON = {"id": 1, "stats": [
    {"group": {"displayName": "hitting"}, "type": {"displayName": "yearByYear"}, "splits": [
        split(2019, 11, {"age": 22, "plateAppearances": 500, "atBats": 450, "hits": 130, "homeRuns": 20,
                         "baseOnBalls": 40, "strikeOuts": 100, "hitByPitch": 5}),
        split(2019, 12, {"age": 22, "plateAppearances": 100, "hits": 25}),
        # two Triple-A teams in 2021 plus the combined line
        split(2021, 11, {"plateAppearances": 200, "hits": 50}),
        split(2021, 11, {"plateAppearances": 100, "hits": 20}),
        split(2021, 11, {"plateAppearances": 300, "hits": 70}, team=False, num_teams=2),
        # MLB line mislabelled with sport 1 must not appear at either level
        split(2022, 1, {"plateAppearances": 600, "hits": 150}),
        # a playoff split must not count
        split(2021, 11, {"plateAppearances": 20, "hits": 5}, game_type="P"),
        # a split with no sport id is not accepted for a minor level
        {"season": "2018", "team": {"id": 2}, "gameType": "R", "stat": {"plateAppearances": 77}},
    ]},
    {"group": {"displayName": "pitching"}, "type": {"displayName": "yearByYear"}, "splits": [
        split(2020, 11, {"battersFaced": 400, "hitBatsmen": 6, "strikeOuts": 90}),
        split(2023, 11, {"battersFaced": 10}, team=True),
        split(2023, 11, {"battersFaced": 20}),   # two team lines, no combined line: added up
    ]},
    {"group": {"displayName": "hitting"}, "type": {"displayName": "career"}, "splits": [split(2019, 11, {"plateAppearances": 999})]},
]}

print("test_minor_seasons")
print("levels")
check("Triple-A is 11, Double-A is 12", (LEVELS["AAA"], LEVELS["AA"]), (11, 12))
rows11 = {(r["season"], r["stat_group"]): r for r in minor_season_rows(PERSON, 11)}
rows12 = {(r["season"], r["stat_group"]): r for r in minor_season_rows(PERSON, 12)}
check("Triple-A hitting seasons", sorted(k[0] for k in rows11 if k[1] == "hitting"), [2019, 2021])
check("Double-A hitting seasons", sorted(k[0] for k in rows12 if k[1] == "hitting"), [2019])
check("MLB-labelled split is not a minor line", (2022, "hitting") in rows11 or (2022, "hitting") in rows12, False)
check("split with no sport id is not accepted", (2018, "hitting") in rows11, False)
check("career block ignored (999 never appears)", rows11[(2019, "hitting")]["plate_appearances"], 500)
print("combined and stint lines")
check("2021 uses the combined line", rows11[(2021, "hitting")]["plate_appearances"], 300)
check("2021 teams", rows11[(2021, "hitting")]["num_teams"], 2)
check("playoff split not counted", rows11[(2021, "hitting")]["hits"], 70)
check("no combined line: team lines added up", rows11[(2023, "pitching")]["batters_faced"], 30)
check("no combined line: number of teams", rows11[(2023, "pitching")]["num_teams"], 2)
print("fields")
check("sport id on the row", rows11[(2019, "hitting")]["sport_id"], 11)
check("hit by pitch, hitting key", rows11[(2019, "hitting")]["hit_by_pitch"], 5)
check("hit by pitch, pitching key (hitBatsmen)", rows11[(2020, "pitching")]["hit_by_pitch"], 6)
check("missing stat is None", rows11[(2019, "hitting")]["sac_flies"], None)
check("one row per season and group", len(minor_season_rows(PERSON, 11)), 4)
check("no splits, no rows", minor_season_rows({"id": 2, "stats": []}, 11), [])
check("lines() empty", minor_season_lines([], 11), {})

print("debut coverage")
cov_rows = [{"player_id": 1, "season": 2020}, {"player_id": 2, "season": 2022},
            {"player_id": 3, "season": 2019}, {"player_id": 676601, "season": 2024}]
cov_debut = {1: date(2021, 5, 1), 2: date(2022, 6, 1), 3: date(2019, 4, 1), 4: date(2022, 4, 1), 5: None}
# 676601 has a minor line but no debut date (the first full build crashed on this)
check("player without a debut date is skipped, not a crash", debut_coverage(cov_rows, cov_debut, [1, 2, 3, 4, 5, 676601]), (3, 1))
check("only players in the fetch count", debut_coverage(cov_rows, cov_debut, [1]), (1, 1))
check("a line in the debut season itself does not count", debut_coverage([{"player_id": 2, "season": 2022}], cov_debut, [2]), (1, 0))

print("FAILED: " + ", ".join(failures) if failures else "all passed")
sys.exit(1 if failures else 0)
