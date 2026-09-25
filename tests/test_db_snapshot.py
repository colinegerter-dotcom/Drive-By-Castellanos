"""Offline tests for the snapshot checks (25 Sep 2026).

No database: builds a tiny snapshot folder by hand and checks that
verification passes on a good copy and fails on a tampered or truncated
one. The export itself was rehearsed against a local Postgres, including a
pg_dump restore and the workflow's encrypt/decrypt steps.

Run: python tests/test_db_snapshot.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from scripts.db_snapshot import arrow_type, sha256, verify  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


def fails(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


print("\ntype mapping covers every column type in the live mlb schema (25 Sep 2026)")
for t in ["bigint", "boolean", "date", "integer", "jsonb", "numeric", "smallint", "text", "timestamp with time zone"]:
    check(t, arrow_type(t, pa) is not None, True)
check("unknown type is refused, not guessed", arrow_type("tsvector", pa), None)

tmp = Path(tempfile.mkdtemp())
try:
    snap = tmp / "snap"
    snap.mkdir()
    pq.write_table(pa.table({"game_id": [1, 2, 3], "runs": [0, 2, 5]}), snap / "inning_scores.parquet")
    manifest = {"tables": {"inning_scores": {"rows": 3, "file": "inning_scores.parquet",
                                             "sha256": sha256(snap / "inning_scores.parquet")}}}
    (snap / "manifest.json").write_text(json.dumps(manifest))

    print("\nverification")
    check("good snapshot passes", fails(lambda: verify(snap)), False)

    bad = tmp / "bad"
    shutil.copytree(snap, bad)
    pq.write_table(pa.table({"game_id": [1, 2], "runs": [0, 2]}), bad / "inning_scores.parquet")
    check("truncated table fails", fails(lambda: verify(bad)), True)

    missing = tmp / "missing"
    shutil.copytree(snap, missing)
    (missing / "inning_scores.parquet").unlink()
    check("missing file fails", fails(lambda: verify(missing)), True)
finally:
    shutil.rmtree(tmp)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all snapshot checks passed")
