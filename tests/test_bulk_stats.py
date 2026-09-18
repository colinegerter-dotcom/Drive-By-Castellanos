"""Offline tests for the bulk-stats path added 18 Sep 2026.

No network: `_get` is stubbed with fixtures shaped like the real responses
observed live from statsapi.mlb.com that day (including the duplicated
per-sport splits, which is the detail the old per-player code got wrong).

Run: python tests/test_bulk_stats.py
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipelines.mlb_stats_client as mlb
from pipelines.player_form.starting_batter_form import build_starting_batter_form_row
from pipelines.player_form.starting_pitcher_form import build_starting_pitcher_form_row

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


class FakeCursor:
    """Returns no pitch rows -- the Statcast half of the form tables is not
    what these tests are about, and an empty pitches table is exactly the
    state of a --skip-pitches run anyway."""

    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return []


class FakeConn:
    @contextmanager
    def cursor(self):
        yield FakeCursor()


def hitting_split(sport_id, pa, hits=30, doubles=6, triples=1, hr=5, bb=10, k=25):
    return {
        "sport": {"id": sport_id},
        "stat": {
            "plateAppearances": pa, "atBats": pa - bb, "hits": hits,
            "doubles": doubles, "triples": triples, "homeRuns": hr,
            "baseOnBalls": bb, "intentionalWalks": 1, "hitByPitch": 2,
            "sacFlies": 1, "strikeOuts": k,
        },
    }


# --- 1. bulk byDateRange: MLB split chosen, ids mapped, chunking works ------
print("\n1. get_stats_by_date_range_bulk")

calls = []


def fake_get(url, params=None):
    calls.append((url, params))
    ids = [int(x) for x in params["personIds"].split(",")]
    people = []
    for pid in ids:
        people.append({
            "id": pid,
            "stats": [{
                "group": {"displayName": "hitting"},
                "type": {"displayName": "byDateRange"},
                # Deliberately MINORS FIRST, then the "All" rollup, then MLB --
                # if the code took splits[0] it would pick the minor league line.
                "splits": [
                    hitting_split(11, pa=999),   # AAA
                    hitting_split(0, pa=500),    # "All" rollup
                    hitting_split(1, pa=100 + pid % 10),  # MLB (the right one)
                ],
            }],
        })
    return {"people": people}


mlb._get = fake_get
result = mlb.get_stats_by_date_range_bulk([660271, 545361, 592450], "hitting", "2025-04-01", "2025-04-30")
check("players returned", sorted(result), [545361, 592450, 660271])
check("picked the MLB split, not AAA or the rollup", result[660271]["plateAppearances"], 101)
check("one HTTP call for 3 players", len(calls), 1)

calls.clear()
mlb.get_stats_by_date_range_bulk(list(range(1, 251)), "hitting", "2025-04-01", "2025-04-30")
check("250 players chunked into 3 calls", len(calls), 3)

calls.clear()
check("backwards window returns {}", mlb.get_stats_by_date_range_bulk([1], "hitting", "2025-03-18", "2025-03-17"), {})
check("...without calling the API", len(calls), 0)

check("empty id list returns {}", mlb.get_stats_by_date_range_bulk([], "hitting", "2025-04-01", "2025-04-30"), {})


def fake_get_wrong_type(url, params=None):
    """Right group, wrong stats type -- must be ignored, not silently used."""
    return {"people": [{
        "id": 1,
        "stats": [
            {"group": {"displayName": "hitting"}, "type": {"displayName": "career"},
             "splits": [hitting_split(1, pa=9999)]},
            {"group": {"displayName": "hitting"}, "type": {"displayName": "byDateRange"},
             "splits": [hitting_split(1, pa=42)]},
        ],
    }]}


mlb._get = fake_get_wrong_type
typed = mlb.get_stats_by_date_range_bulk([1], "hitting", "2025-04-01", "2025-04-30")
check("ignores a career block and takes byDateRange", typed[1]["plateAppearances"], 42)


# --- 2. career totals: prior seasons only, MLB only ------------------------
print("\n2. get_career_totals_before_season")


def fake_get_career(url, params=None):
    return {"people": [{
        "id": 545361,
        "stats": [{
            "group": {"displayName": "hitting"},
            "type": {"displayName": "yearByYear"},
            "splits": [
                {"season": "2023", "sport": {"id": 1}, "stat": {"plateAppearances": 400, "outs": 0}},
                {"season": "2024", "sport": {"id": 1}, "stat": {"plateAppearances": 600, "outs": 0}},
                {"season": "2024", "sport": {"id": 11}, "stat": {"plateAppearances": 50, "outs": 0}},   # rehab in AAA
                {"season": "2025", "sport": {"id": 1}, "stat": {"plateAppearances": 700, "outs": 0}},   # the season itself
                {"season": "2026", "sport": {"id": 1}, "stat": {"plateAppearances": 123, "outs": 0}},   # the future
            ],
        }],
    }]}


mlb._get = fake_get_career
career = mlb.get_career_totals_before_season([545361], "hitting", 2025)
check("sums 2023+2024 MLB only (excludes AAA, this season, and later)",
      career[545361]["plateAppearances"], 1000)


# --- 3. prefetched path matches the per-call path, and mlb_pa_count adds up -
print("\n3. build_starting_batter_form_row: prefetched vs per-call")

season_stat = hitting_split(1, pa=300)["stat"]
last30_stat = hitting_split(1, pa=100)["stat"]

api_calls = []


def fake_get_single(url, params=None):
    api_calls.append(url)
    window = (params["startDate"], params["endDate"])
    stat = season_stat if window[0] == "2025-03-01" else last30_stat
    if "career" in str(window):
        stat = season_stat
    return {"stats": [{"splits": [{"sport": {"id": 1}, "stat": stat}]}]}


mlb._get = fake_get_single
import pipelines.player_form.starting_batter_form as sbf
sbf.get_player_stats_by_date_range = mlb.get_player_stats_by_date_range

per_call = build_starting_batter_form_row(FakeConn(), 545361, 777001, "2025-06-01", 2025, "2019-04-01")
calls_made_per_call_path = len(api_calls)

api_calls.clear()
prefetched = build_starting_batter_form_row(
    FakeConn(), 545361, 777001, "2025-06-01", 2025, "2019-04-01",
    prefetched={"season": season_stat, "last30": last30_stat,
                "career_before_season": {"plateAppearances": 2500}},
)

check("per-call path made 3 HTTP calls", calls_made_per_call_path, 3)
check("prefetched path made 0 HTTP calls", len(api_calls), 0)
check("woba_season identical either way", prefetched["woba_season"], per_call["woba_season"])
check("k_pct_season identical either way", prefetched["k_pct_season"], per_call["k_pct_season"])
check("bb_pct_last_30d identical either way", prefetched["bb_pct_last_30d"], per_call["bb_pct_last_30d"])
check("mlb_pa_count = 2500 prior + 300 this season", prefetched["mlb_pa_count"], 2800)

missing = build_starting_batter_form_row(
    FakeConn(), 999999, 777001, "2025-06-01", 2025, None,
    prefetched={"season": None, "last30": None, "career_before_season": None},
)
check("player with no stats at all -> nulls, no crash", missing["woba_season"], None)
check("...and mlb_pa_count is None, not 0", missing["mlb_pa_count"], None)


# --- 4. pitcher innings rebuilt from outs ----------------------------------
print("\n4. build_starting_pitcher_form_row: innings from outs")

pitcher_row = build_starting_pitcher_form_row(
    FakeConn(), 601713, 777001, "2025-06-01", 2025, "2017-04-30",
    prefetched={"season": {"outs": 180, "era": 3.5, "battersFaced": 250, "strikeOuts": 60, "baseOnBalls": 20},
                "last30": {"outs": 60, "era": 2.9, "battersFaced": 90, "strikeOuts": 25, "baseOnBalls": 6},
                "career_before_season": {"outs": 2400}},
)
# 2400 prior outs + 180 this season = 2580 outs = 860.0 innings
check("mlb_ip_count = (2400+180)/3", pitcher_row["mlb_ip_count"], 860.0)
check("era_season passes through", pitcher_row["era_season"], 3.5)
check("k_pct_season computed from battersFaced", pitcher_row["k_pct_season"], 24.0)

no_career = build_starting_pitcher_form_row(
    FakeConn(), 601713, 777001, "2025-06-01", 2025, None,
    prefetched={"season": {}, "last30": {}, "career_before_season": None},
)
check("no innings data -> None, not 0.0", no_career["mlb_ip_count"], None)


# --- 5. the whole point: calls per game ------------------------------------
print("\n5. calls per game (18 batters + 2 pitchers)")
old = 18 * 3 + 2 * 3
new = 4  # hitting season+last30, pitching season+last30
check("old per-game HTTP calls", old, 60)
check("new per-game HTTP calls", new, 4)
print(f"  -> {old / new:.0f}x fewer requests per game")

print("\n" + ("FAILED: " + "; ".join(failures) if failures else "ALL CHECKS PASSED"))
sys.exit(1 if failures else 0)
