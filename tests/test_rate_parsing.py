"""Offline tests for MLB rate-stat placeholder parsing (22 Sep 2026).

No network, no DB. Reproduces the 2022 backfill crash: MLB returned
era "-.--" for pitcher 605483 (game 661624), a pitcher with no innings in
the season-to-date window, and the raw string reached Postgres as a numeric.

Run: python tests/test_rate_parsing.py
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.player_form.starting_pitcher_form import (  # noqa: E402
    _rate_or_none,
    build_starting_pitcher_form_row,
)

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


print("\nplaceholders become None")
for raw in ["-.--", ".---", "*.**", "", "  ", "∞", "inf", "nan", None]:
    check(f"{raw!r}", _rate_or_none(raw), None)

print("\nreal values are unchanged (rows written before the fix stay valid)")
check('"3.45"', _rate_or_none("3.45"), 3.45)
check('"0.00"', _rate_or_none("0.00"), 0.0)
check('"27.00"', _rate_or_none("27.00"), 27.0)
check("float 2.9", _rate_or_none(2.9), 2.9)
check("int 0", _rate_or_none(0), 0.0)


class FakeCursor:
    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return []


class FakeConn:
    @contextmanager
    def cursor(self):
        yield FakeCursor()


print("\nthe exact crash case builds a clean row")
row = build_starting_pitcher_form_row(
    FakeConn(),
    605483,
    661624,
    "2022-05-15",
    2022,
    None,
    prefetched={
        "season": {"outs": 0, "era": "-.--", "battersFaced": 2, "strikeOuts": 0, "baseOnBalls": 2},
        "last30": {"outs": 0, "era": "-.--", "battersFaced": 2, "strikeOuts": 0, "baseOnBalls": 2},
        "career_before_season": {"outs": 30},
    },
)
check("era_season is None, not '-.--'", row["era_season"], None)
check("era_last_30d is None, not '-.--'", row["era_last_30d"], None)
check("other fields still computed", row["bb_pct_season"], 100.0)
check("no string reaches a numeric column",
      any(isinstance(v, str) for k, v in row.items()), False)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all rate parsing tests passed")
