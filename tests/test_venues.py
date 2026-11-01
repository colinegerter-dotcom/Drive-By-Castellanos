"""Offline tests for the one venue table (pipelines/reference/venues.py).

Added 1 Nov 2026 with the 2027 fixes (fixes-2027.md section 3). The venue
table replaced copies in config.py, features/inputs.py, features/travel.py
and features/wind.py. These tests pin that the table gives the same values
those copies had, apart from the listed, intended additions, and that an
unknown venue stops the run.

Run: python tests/test_venues.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.reference import venues as V  # noqa: E402

failures = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + ("" if ok else f": {detail}"))
    if not ok:
        failures.append(label)


# every venue id with a game in mlb.games, 2021-2026 (queried 1 Nov 2026)
USED_2021_2026 = {1, 2, 3, 4, 5, 7, 10, 12, 14, 15, 17, 19, 22, 31, 32, 680, 2392, 2394, 2395, 2397, 2523,
                  2529, 2536, 2602, 2680, 2681, 2735, 2756, 2889, 3289, 3309, 3312, 3313, 3949, 4169, 4705,
                  5150, 5325, 5340, 5355, 5381, 5445, 6130}

# the copies the table replaced, as they were at commit 66b2818
ET, CT, MT, PT = -4, -5, -6, -7
OLD_TZ = {
    1: PT, 2: ET, 3: ET, 4: CT, 5: ET, 7: CT, 10: PT, 12: ET, 14: ET, 15: PT, 17: CT, 19: MT,
    22: PT, 31: ET, 32: CT, 680: PT, 2392: CT, 2394: ET, 2395: PT, 2536: ET, 2602: ET,
    2680: PT, 2681: ET, 2735: ET, 2756: ET, 2889: CT, 3289: ET, 3309: ET, 3312: CT,
    3313: ET, 3949: CT, 4169: ET, 4705: ET, 5150: 9, 5325: CT, 5340: -6, 5381: 1, 5445: CT,
}
OLD_INTERNATIONAL = {5150, 5340, 5381}
OLD_ROOFED = {32, 15, 2392, 5325, 4169, 14, 680, 12}
OLD_DOMES = {12}
OLD_BEARING = {1: 30, 2: 358, 3: 42, 4: 186, 5: 2, 7: 13, 17: 30, 19: 342, 22: 32, 31: 117, 2394: 141,
               2395: 60, 2602: 10, 2680: 350, 2681: 340, 2889: 37, 3289: 337, 3309: 2, 3312: 344,
               3313: 33, 4705: 152}

print("coverage")
missing = USED_2021_2026 - set(V.venues())
check("every 2021-2026 venue is in the table", not missing, f"missing {sorted(missing)}")

print("same values as the old copies, apart from the intended additions")
tz = V.utc_offsets()
check("time zones unchanged for every old id", all(tz[v] == o for v, o in OLD_TZ.items()),
      {v: (o, tz.get(v)) for v, o in OLD_TZ.items() if tz.get(v) != o})
added_tz = set(tz) - set(OLD_TZ)
check("time zones added only for the 5 missing ids", added_tz == {2397, 2523, 2529, 5355, 6130}, sorted(added_tz))
check("international = old set plus Tokyo", V.international_ids() == OLD_INTERNATIONAL | {2397},
      sorted(V.international_ids()))
check("roofed = old set plus the two international domes", V.roofed_ids() == OLD_ROOFED | {2397, 5150},
      sorted(V.roofed_ids()))
check("domes = Tropicana plus the two international domes", V.dome_ids() == OLD_DOMES | {2397, 5150},
      sorted(V.dome_ids()))
b = V.open_air_bearings()
check("wind bearings unchanged for every old id", all(b.get(v) == o for v, o in OLD_BEARING.items()),
      {v: (o, b.get(v)) for v, o in OLD_BEARING.items() if b.get(v) != o})
check("wind bearings added only for Sutter Health Park", set(b) - set(OLD_BEARING) == {2529},
      sorted(set(b) - set(OLD_BEARING)))

print("the modules read the table")
from pipelines.features import inputs, travel, wind  # noqa: E402
check("inputs.ROOFED_VENUE_IDS", inputs.ROOFED_VENUE_IDS == V.roofed_ids())
check("inputs.DOME_VENUE_IDS", inputs.DOME_VENUE_IDS == V.dome_ids())
check("travel.TZ_BY_VENUE", travel.TZ_BY_VENUE == V.utc_offsets())
check("travel.INTERNATIONAL", travel.INTERNATIONAL == V.international_ids())
check("wind.BEARING", wind.BEARING == V.open_air_bearings())

print("unknown venue stops the run")
try:
    V.require_known([17, 999999])
    check("unknown id raises", False, "no error")
except V.UnknownVenue as e:
    check("unknown id raises", "999999" in str(e))
try:
    V.require_known(USED_2021_2026)
    check("known ids pass", True)
except V.UnknownVenue as e:
    check("known ids pass", False, str(e))

print("table sanity")
rows = V.venues().values()
check("offsets between -12 and +14", all(-12 <= r["utc_offset_summer"] <= 14 for r in rows))
check("lat and lon in range", all(-90 <= r["lat"] <= 90 and -180 <= r["lon"] <= 180 for r in rows))
check("bearings between 0 and 360", all(r["bearing"] is None or 0 <= r["bearing"] < 360 for r in rows))
names = [n for r in rows for n in [r["name"], *r["other_names"]]]
check("no name used twice", len(names) == len(set(names)))

if failures:
    print(f"\n{len(failures)} venue check(s) failed: {failures}")
    sys.exit(1)
print("\nall venue checks passed")
