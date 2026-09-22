"""Offline tests for historical venue-name aliasing (22 Sep 2026).

No network, no DB.

The bug: mlb.games.venue holds the name the schedule feed reported at the
time a game was played, while the venues endpoint returns the name the park
has today. When a park is renamed the two stop matching, and every
name-keyed lookup for that park falls through SILENTLY.

Found live during the 2021 backfill -- the log filled with "no coordinates
for venue 'Guaranteed Rate Field'" and "'Minute Maid Park'" (the 2021 names
of Rate Field and Daikin Park) and weather was skipped for ~162 games, about
6.7% of the season. The same latent bug was in the park factor matcher,
where it would have handed those parks a null factor instead.

Run: python tests/test_venue_aliases.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipelines.games.game_conditions as gc  # noqa: E402
import pipelines.reference.park_factors as pf  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


# Two real cases plus one control. Venue 4 and 2392 were renamed between
# 2021 and 2025; venue 19 (Coors) never was.
CURRENT_VENUES = [
    {"id": 4, "name": "Rate Field", "location": {"defaultCoordinates": {"latitude": 41.83, "longitude": -87.63}}},
    {"id": 2392, "name": "Daikin Park", "location": {"defaultCoordinates": {"latitude": 29.75, "longitude": -95.35}}},
    {"id": 19, "name": "Coors Field", "location": {"defaultCoordinates": {"latitude": 39.75, "longitude": -104.99}}},
]
NAMES_2021 = {4: "Guaranteed Rate Field", 2392: "Minute Maid Park", 19: "Coors Field"}


def stub(monkey_seasons=NAMES_2021, raise_on_season=False):
    gc.get_venues = lambda: CURRENT_VENUES

    def fake(season):
        if raise_on_season:
            raise RuntimeError("simulated MLB API failure")
        return monkey_seasons

    gc.get_venue_names_by_season = fake


print("\nwithout seasons: today's names only (the daily-pull case)")
stub()
coords = gc._venue_coords_by_name()
check("current name resolves", "Rate Field" in coords, True)
check("2021 name does NOT resolve", "Guaranteed Rate Field" in coords, False)
check("this is the bug, reproduced", "Minute Maid Park" in coords, False)

print("\nwith seasons=[2021]: historical names resolve too")
stub()
coords = gc._venue_coords_by_name(seasons=[2021])
check("2021 White Sox park resolves", "Guaranteed Rate Field" in coords, True)
check("2021 Astros park resolves", "Minute Maid Park" in coords, True)
check("current names still resolve", "Rate Field" in coords, True)
check("unrenamed park unaffected", coords.get("Coors Field"), (39.75, -104.99))

print("\nthe alias points at the SAME coordinates, not just any coordinates")
# The failure that would matter most is an alias resolving to the wrong
# park's lat/lon -- that produces plausible weather for the wrong city,
# which is worse than no weather at all.
check("Guaranteed Rate Field -> White Sox coords", coords.get("Guaranteed Rate Field"), (41.83, -87.63))
check("Minute Maid Park -> Astros coords", coords.get("Minute Maid Park"), (29.75, -95.35))
check("the two are not confused with each other", coords["Guaranteed Rate Field"] != coords["Minute Maid Park"], True)

print("\ncurrent names win over historical ones on collision")
# If a name is ever reused by a different park, the park that holds it TODAY
# must keep it -- the current list is written first and aliases only fill gaps.
stub(monkey_seasons={2392: "Rate Field"})  # pretend Houston was once called Rate Field
coords = gc._venue_coords_by_name(seasons=[2021])
check("Rate Field still points at the White Sox", coords.get("Rate Field"), (41.83, -87.63))

print("\na failed alias lookup degrades, it does not kill the backfill")
stub(raise_on_season=True)
try:
    coords = gc._venue_coords_by_name(seasons=[2021])
    check("no exception escapes", True, True)
    check("current names still work", coords.get("Coors Field"), (39.75, -104.99))
    check("historical names absent, as expected", "Guaranteed Rate Field" in coords, False)
except Exception as exc:  # noqa: BLE001
    check("no exception escapes", f"raised {type(exc).__name__}", True)

print("\npark factors match a factor keyed by a historical venue name")
# build_park_factor_rows keys factors by mlb.games.venue for the LOOKBACK
# seasons, i.e. the old name, then matches them to today's venue list.
pf.get_venues = lambda: CURRENT_VENUES
pf.get_venue_names_by_season = lambda season: NAMES_2021
pf._load_orientation_by_name = lambda: {}
# 2021 games recorded under the 2021 name.
pf.fetch_run_aggregates = lambda conn, seasons: (
    [{"venue": "Guaranteed Rate Field", "home_team": 145, "games": 81, "runs": 810}],
    {145: {"games": 81, "runs": 729}},
)

rows = pf.build_park_factor_rows(None, 2022, available_seasons={2021, 2022})
by_id = {r["park_id"]: r for r in rows}
check("White Sox park got a factor despite the rename", by_id["4"]["park_factor_runs"] is not None, True)
check("factor is above neutral (more runs at home)", by_id["4"]["park_factor_runs"] > 1.0, True)
check("a park with no games still returns a row", "19" in by_id, True)
check("and that row's factor is null, not a placeholder", by_id["19"]["park_factor_runs"], None)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all venue alias tests passed")
