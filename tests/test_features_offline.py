"""Offline tests for the feature builder's small pieces (25 Sep 2026).

The heavy checks (future-perturbation leak test, determinism) run in the
build-features workflow against the real inputs; see
pipelines/features/leak_test.py. These cover the formulas.

Run: python tests/test_features_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipelines.features import pitching, priors  # noqa: E402

failures = []


def check(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label}")


con = duckdb.connect()

print("\nblend formula")
# One season so far (10 K in 40 PA) + last season (50 K in 200 PA, weight
# 0.8, from the pitch files so src_w = 1) + k=60 PA of league 0.22.
con.execute("""create table t as select 10 as k, 40 as pa,
                1.0 as s1w, 50 as s1k, 200 as s1pa,
                null::double as s2w, null::int as s2k, null::int as s2pa,
                null::double as s3w, null::int as s3k, null::int as s3pa, 0.22 as mu""")
sql = priors._blend_cols("k", "pa", 60, "mu", 0.8, 0.64, 0.48)
got = con.execute(f"select {sql} from t").fetchone()[0]
want = (10 + 0.8 * 50 + 60 * 0.22) / (40 + 0.8 * 200 + 60)
check("current + one prior season + league shrinkage", got, want)

con.execute("""create table t2 as select null::int as k, null::int as pa,
                null::double as s1w, null::int as s1k, null::int as s1pa,
                null::double as s2w, null::int as s2k, null::int as s2pa,
                null::double as s3w, null::int as s3k, null::int as s3pa, 0.22 as mu""")
got = con.execute(f"select {sql} from t2").fetchone()[0]
check("no history at all -> exactly the league average", got, 0.22)

print("\nage factor")
got = con.execute("select " + priors._age_factor("date '1995-01-01'", "date '2024-01-01'")).fetchone()[0]
check("age 29 -> 1.0", round(got, 3), 1.0)
got = con.execute("select " + priors._age_factor("date '2001-01-01'", "date '2024-01-01'")).fetchone()[0]
check("age 23 -> above 1", got > 1, True)
got = con.execute("select " + priors._age_factor("null::date", "date '2024-01-01'")).fetchone()[0]
check("unknown birth date -> 1.0", got, 1.0)

print("\nrest buckets")
for days, first, want in [(4, False, "r4"), (5, False, "r5_6"), (8, False, "r7_10"),
                          (15, False, "r11_20"), (40, False, "r21p"), (99, True, "first")]:
    got = con.execute(f"select {pitching.rest_bucket_sql(str(days), str(first).lower())}").fetchone()[0]
    check(f"{days} days, first={first} -> {want}", got, want)

print("\nexpected batters faced is clipped to a sane range")
df = pd.DataFrame({"usual_bf": [60.0, 0.0], "last_pitches": [200, 0], "no_last": [False, False],
                   "hook": [50.0, -50.0], "rest_bucket": ["r5_6", "r4"]})
coef = {c: 1.0 for c in ["const", "usual_bf", "last_pitches", "no_last", "hook"] + [f"rest_{b}" for b in pitching.REST_BUCKETS[1:]]}
out = pitching.apply_exp_bf(df, coef)
check("high end clipped to 30", float(out[0]), 30.0)
check("low end clipped to 3", float(out[1]), 3.0)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    sys.exit(1)
print("all feature formula checks passed")
