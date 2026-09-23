"""Offline tests for the pitch-row build path.

No network, no DB. These exist because of a live failure on 21 Sep 2026:
Savant's CSV came back with pandas *nullable* dtypes, whose missing value is
pd.NA rather than float('nan'). The old _clean() only tested for NaN, so pd.NA
flowed straight through to psycopg2, which raised "can't adapt type 'NAType'"
on every single pitch batch. The pipeline logged a successful run and wrote
about 1% of the season.

The lesson worth encoding: anything handed to psycopg2 must be a plain Python
value, and "is it missing" has three different answers in pandas.

Run: python tests/test_pitch_rows.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from pipelines.pitches.pitches import (
    EXTRA_FIELD_SOURCES,
    REQUIRED_SAVANT_COLUMNS,
    SKIP_GAME_TYPES,
    _clean,
    build_pitch_rows_for_range,
)
import pipelines.pitches.pitches as pitches_mod

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


def check_type(label, value, want_type):
    if type(value) is not want_type:
        failures.append(f"{label}: got {type(value).__name__}, want {want_type.__name__}")
        print(f"  FAIL {label}: got {type(value).__name__}, want {want_type.__name__}")
    else:
        print(f"  ok   {label} is {want_type.__name__}")


# --- 1. the three flavours of missing -------------------------------------
print("\n1. _clean: every pandas way of saying 'missing' becomes None")

check("None", _clean(None), None)
check("float nan", _clean(float("nan")), None)
check("np.nan", _clean(np.nan), None)
check("pd.NA (this is the one that broke production)", _clean(pd.NA), None)
check("pd.NaT", _clean(pd.NaT), None)


# --- 2. numpy scalars unwrap to plain Python ------------------------------
print("\n2. _clean: numpy scalars become values psycopg2 can bind")

check_type("np.int64 -> int", _clean(np.int64(778564)), int)
check("np.int64 keeps its value", _clean(np.int64(778564)), 778564)
check_type("np.float64 -> float", _clean(np.float64(95.3)), float)
check_type("np.bool_ -> bool", _clean(np.bool_(True)), bool)

# pandas nullable Int64 columns yield np.int64 for present values and pd.NA
# for missing ones -- both paths matter.
series = pd.Series([1, None, 3], dtype="Int64")
check("nullable Int64, present value", _clean(series[0]), 1)
check_type("...as a plain int", _clean(series[0]), int)
check("nullable Int64, missing value", _clean(series[1]), None)


# --- 3. real values survive untouched -------------------------------------
print("\n3. _clean: real values pass through unchanged")

check("str", _clean("ball"), "ball")
check("python int", _clean(5), 5)
check("python float", _clean(1.5), 1.5)
check("zero is not missing", _clean(0), 0)
check("empty string is not missing", _clean(""), "")
check("False is not missing", _clean(False), False)


# --- 4. end to end: no pandas objects escape into a row dict --------------
print("\n4. build_pitch_rows_for_range: nothing pandas-shaped reaches the DB")

fake = pd.DataFrame({
    "game_pk":        pd.array([778564, 778564, 999001, 778565], dtype="Int64"),
    "at_bat_number":  pd.array([1, 1, 1, 2], dtype="Int64"),
    "pitch_number":   pd.array([1, 2, 1, 1], dtype="Int64"),
    "game_type":      ["R", "R", "S", "R"],          # one spring training row
    "pitcher":        pd.array([601713, 601713, 1, 601713], dtype="Int64"),
    "batter":         pd.array([660271, None, 2, 660271], dtype="Int64"),  # a missing one
    "release_speed":  [95.3, np.nan, 88.0, 92.1],
    "release_spin_rate": pd.array([2400, None, 2000, 2350], dtype="Int64"),
    "description":    ["ball", "called_strike", "ball", "foul"],
    "bb_type":        [None, None, None, "fly_ball"],
    "inning":         pd.array([1, 1, 1, 3], dtype="Int64"),
})

# The build refuses a Savant table with any expected column absent (see
# REQUIRED_SAVANT_COLUMNS), so pad this minimal fixture with empty columns.
for col in REQUIRED_SAVANT_COLUMNS - set(fake.columns):
    fake[col] = np.nan

pitches_mod.pull_statcast_range = lambda s, e: fake
rows = build_pitch_rows_for_range("2025-04-01", "2025-04-07")

check("spring training row dropped", len(rows), 3)
check("game_types kept are all non-skip", SKIP_GAME_TYPES & {"R"}, set())

bad = []
for row in rows:
    for key, value in row.items():
        if value is None:
            continue
        if type(value) not in (int, float, str, bool):
            bad.append(f"{key}={value!r} ({type(value).__name__})")
check("no non-primitive values in any row", bad, [])

check("missing batter_id became None", rows[1]["batter_id"], None)
check("missing release_speed became None", rows[1]["release_speed"], None)
check("present spin_rate survived", rows[0]["spin_rate"], 2400)
check("present release_speed survived", rows[0]["release_speed"], 95.3)
check("natural key is ints", (rows[0]["game_id"], rows[0]["at_bat_id"], rows[0]["pitch_number"]), (778564, 1, 1))

# The specific crash, reproduced: pd.NA must never reach a row dict.
na_leaks = [k for row in rows for k, v in row.items() if v is pd.NA]
check("no pd.NA anywhere in the output", na_leaks, [])



# --- 5. the 23 Sep 2026 fields --------------------------------------------
print("\n5. movement, expected stats and game situation map from the right Savant columns")

fake2 = fake.copy()
fake2["pfx_x"] = [-0.5, 0.2, 0.0, 1.1]
fake2["estimated_woba_using_speedangle"] = [np.nan, np.nan, np.nan, 0.412]
fake2["outs_when_up"] = pd.array([0, 1, 0, 2], dtype="Int64")
fake2["on_2b"] = pd.array([None, None, None, 660271], dtype="Int64")
fake2["inning_topbot"] = ["Top", "Top", "Top", "Bot"]
fake2["delta_run_exp"] = [-0.02, 0.05, 0.0, 0.31]
pitches_mod.pull_statcast_range = lambda s, e: fake2
rows2 = build_pitch_rows_for_range("2025-04-01", "2025-04-07")
check("every extra field present in every row",
      all(set(EXTRA_FIELD_SOURCES) <= set(r) for r in rows2), True)
check("pfx_x", rows2[0]["pfx_x"], -0.5)
check("xwoba from estimated_woba_using_speedangle", rows2[2]["xwoba"], 0.412)
check("xwoba empty on a non-batted ball", rows2[0]["xwoba"], None)
check("outs from outs_when_up", rows2[2]["outs"], 2)
check_type("...as a plain int", rows2[2]["outs"], int)
check("runner on second is a player id", rows2[2]["on_2b"], 660271)
check("empty base is None", rows2[0]["on_2b"], None)
check("half inning", rows2[2]["inning_topbot"], "Bot")
check("run value", rows2[2]["delta_run_exp"], 0.31)

print("\n6. a column Savant stopped sending stops the pull")
pitches_mod.pull_statcast_range = lambda s, e: fake2.drop(columns=["pfx_z", "delta_run_exp"])
try:
    build_pitch_rows_for_range("2025-04-01", "2025-04-07")
    check("raised", False, True)
except RuntimeError as e:
    check("raised, naming the columns", "delta_run_exp, pfx_z" in str(e), True)


print("\n" + ("FAILED: " + "; ".join(failures) if failures else "ALL CHECKS PASSED"))
sys.exit(1 if failures else 0)
