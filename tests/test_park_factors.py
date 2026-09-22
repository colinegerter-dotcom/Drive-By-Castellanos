"""Offline tests for the in-house park factor calculation (22 Sep 2026).

No network, no DB. These test the arithmetic in
pipelines/reference/park_factors.py, which is the part that can be silently
wrong -- a park factor that is quietly 1.4 because a good offense played
there looks exactly like a real hitters' park to everything downstream.

The most important test here is `team quality does not create a park
factor`. That is the whole reason the method compares a team's home games
against its OWN away games rather than against the league average.

Run: python tests/test_park_factors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.reference.park_factors import (  # noqa: E402
    PARK_FACTOR_MIN_HOME_GAMES,
    PARK_FACTOR_REGRESSION_K,
    compute_park_factors,
    lookback_seasons,
)

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


def home(venue, team, games, rpg):
    """One aggregate row: `games` home games at `venue` averaging `rpg` total runs."""
    return {"venue": venue, "home_team": team, "games": games, "runs": int(round(games * rpg))}


def away(games, rpg):
    return {"games": games, "runs": int(round(games * rpg))}


print("\nlookback_seasons -- the no-lookahead guarantee")
# A park factor for year Y may only ever see seasons before Y. If this test
# fails, every park factor in the database is leaking its own outcome.
check("2025 window excludes 2025", 2025 in lookback_seasons(2025), False)
check("2025 window is the 3 prior seasons", lookback_seasons(2025), [2022, 2023, 2024])
check(
    "window intersects with what exists",
    lookback_seasons(2023, available_seasons={2021, 2022, 2023, 2024, 2025}),
    [2021, 2022],
)
check(
    "earliest season has no window at all",
    lookback_seasons(2021, available_seasons={2021, 2022, 2023, 2024, 2025}),
    [],
)
check(
    "no future season ever sneaks in",
    lookback_seasons(2022, available_seasons={2021, 2022, 2023, 2024, 2025}),
    [2021],
)

print("\ncompute_park_factors -- basic behaviour")
# A park where the home team scores exactly as much at home as on the road
# is neutral by definition, whatever the absolute run level.
neutral = compute_park_factors(
    [home("Neutral Park", 1, 162, 9.0)], {1: away(162, 9.0)}
)
check("neutral park is exactly 1.0", neutral.get("Neutral Park"), 1.0)

# 20% more runs at home than on the road, with home_games == K, so the
# regression keeps exactly half the deviation: 1 + 0.20 * 0.5 = 1.10
hitters = compute_park_factors(
    [home("Hitters Park", 1, PARK_FACTOR_REGRESSION_K, 12.0)],
    {1: away(PARK_FACTOR_REGRESSION_K, 10.0)},
)
check("hitters park regressed to 1.10", hitters.get("Hitters Park"), 1.1)

pitchers = compute_park_factors(
    [home("Pitchers Park", 1, PARK_FACTOR_REGRESSION_K, 8.0)],
    {1: away(PARK_FACTOR_REGRESSION_K, 10.0)},
)
check("pitchers park regressed to 0.90", pitchers.get("Pitchers Park"), 0.9)

print("\nteam quality does not create a park factor -- the key property")
# Two clubs, wildly different offences, both in genuinely neutral parks.
# A naive "runs here vs league average" method would call the good team's
# park a hitters' park. Comparing each club against its own road games must
# return 1.0 for both.
rows = [home("Good Team Park", 1, 162, 13.0), home("Bad Team Park", 2, 162, 7.0)]
aways = {1: away(162, 13.0), 2: away(162, 7.0)}
got = compute_park_factors(rows, aways)
check("great offence, neutral park -> 1.0", got.get("Good Team Park"), 1.0)
check("weak offence, neutral park -> 1.0", got.get("Bad Team Park"), 1.0)

print("\nsmall-sample guards")
# Below the floor the venue is OMITTED, not given a placeholder number.
# A missing key becomes a null column; a placeholder would be indistinguishable
# from a real measurement.
thin = compute_park_factors(
    [home("Thin Park", 1, PARK_FACTOR_MIN_HOME_GAMES - 1, 12.0)], {1: away(162, 10.0)}
)
check("under the games floor is omitted", "Thin Park" in thin, False)

at_floor = compute_park_factors(
    [home("Floor Park", 1, PARK_FACTOR_MIN_HOME_GAMES, 12.0)], {1: away(162, 10.0)}
)
check("exactly at the floor is kept", "Floor Park" in at_floor, True)

# Regression test for a real bug in the first draft of this file. The floor
# was 81 (a full home schedule), which silently dropped six actual MLB parks
# in 2025 -- Yankee Stadium, Citi Field, Rate Field and Target Field at 80
# games, Wrigley and Great American at 79 -- because each club gave up a
# home date to a neutral-site event. Real parks must survive; one-off
# novelty venues must not.
real_but_short = compute_park_factors(
    [
        home("Yankee Stadium", 1, 80, 9.0),
        home("Wrigley Field", 2, 79, 9.0),
        home("Great American Ball Park", 3, 79, 9.0),
    ],
    {1: away(81, 9.0), 2: away(83, 9.0), 3: away(83, 9.0)},
)
check("80-game park survives the floor", "Yankee Stadium" in real_but_short, True)
check("79-game park survives the floor", "Wrigley Field" in real_but_short, True)

novelty = compute_park_factors(
    [home("Tokyo Dome", 1, 2, 9.0), home("Bristol Motor Speedway", 2, 1, 9.0)],
    {1: away(81, 9.0), 2: away(81, 9.0)},
)
check("2-game neutral site is excluded", "Tokyo Dome" in novelty, False)
check("1-game neutral site is excluded", "Bristol Motor Speedway" in novelty, False)

no_road = compute_park_factors([home("Orphan Park", 1, 162, 10.0)], {})
check("no away games for the control -> omitted", "Orphan Park" in no_road, False)

zero_road = compute_park_factors([home("Zero Park", 1, 162, 10.0)], {1: away(162, 0.0)})
check("zero road runs -> omitted, no divide by zero", "Zero Park" in zero_road, False)

print("\nregression strength scales with sample size")
# More games -> the raw deviation survives more of the shrink. Same 20% raw
# edge, four times the sample, must land further from 1.0.
small = compute_park_factors([home("P", 1, 162, 12.0)], {1: away(162, 10.0)})["P"]
large = compute_park_factors([home("P", 1, 648, 12.0)], {1: away(648, 10.0)})["P"]
check("more games keeps more of the deviation", large > small, True)
check("both stay on the correct side of neutral", small > 1.0 and large > 1.0, True)
check("regression never overshoots the raw 1.20", large < 1.2, True)

print("\nmultiple home teams at one venue (relocation / shared park)")
# Two clubs sharing a venue are pooled, and the control group pools both
# clubs' road games -- the A's 2025 move makes this a real case, not theory.
shared = compute_park_factors(
    [home("Shared Park", 1, 81, 11.0), home("Shared Park", 2, 81, 9.0)],
    {1: away(81, 11.0), 2: away(81, 9.0)},
)
check("shared venue pools both clubs -> 1.0", shared.get("Shared Park"), 1.0)
check("shared venue clears the floor by pooling", "Shared Park" in shared, True)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all park factor tests passed")
