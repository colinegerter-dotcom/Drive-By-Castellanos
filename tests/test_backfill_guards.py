"""Offline tests for the two guards added 22 Sep 2026.

No network, no DB.

1. --skip-postgame must REFUSE to run on a season whose postgame data
   isn't loaded. Skipping it there would hand backfill_form_tables an empty
   game_results table, and the form build would produce a full set of rows
   computed from nothing and log a clean finish. That is the exact failure
   shape behind bugs 5, 6 and 7 in the build log, so the guard matters more
   than the 28 minutes the flag saves.

2. get_team_outs_above_average() must REFUSE a through_date. Verified live
   22 Sep 2026: Savant silently ignores startDate/endDate on that
   leaderboard and returns season-final numbers for any range asked of it.
   Accepting a through_date would mean every team_form row carried a
   defensive rating computed partly from the game being predicted.

Run: python tests/test_backfill_guards.py
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


class FakeCursor:
    """Returns a single count, which is all _completed_result_count reads."""

    def __init__(self, count):
        self._count = count

    def execute(self, *a, **k):
        pass

    def fetchone(self):
        return (self._count,)

    def fetchall(self):
        return [(self._count,)]


class FakeConn:
    def __init__(self, count):
        self._count = count

    @contextmanager
    def cursor(self):
        yield FakeCursor(self._count)


# ---------------------------------------------------------------- guard 1
# Loaded by path rather than `import scripts.backfill`, so this test doesn't
# require scripts/ to be a package (it isn't one, and making it one just to
# satisfy a test would be the test changing the shipping layout).
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_backfill_under_test", Path(__file__).resolve().parent.parent / "scripts" / "backfill.py"
)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

print("\n--skip-postgame guard")

games = [{"game_id": i} for i in range(2430)]

# A season that was never loaded: 0 of 2430 games have results.
try:
    bf._require_postgame_loaded(FakeConn(0), 2021, games)
    check("empty season is refused", "did not raise", "SystemExit")
except SystemExit as exc:
    check("empty season is refused", True, True)
    msg = str(exc)
    check("refusal names the season", "2021" in msg, True)
    check("refusal gives the actual counts", "0 of 2430" in msg, True)
    check(
        "refusal says what to do instead",
        "WITHOUT --skip-postgame" in msg,
        True,
    )

# A half-loaded season -- the dangerous middle case, where a partial write
# left real rows behind and a naive "are there any rows?" check would pass.
try:
    bf._require_postgame_loaded(FakeConn(1215), 2022, games)  # 50%
    check("half-loaded season is refused", "did not raise", "SystemExit")
except SystemExit:
    check("half-loaded season is refused", True, True)

# Just under the floor.
try:
    bf._require_postgame_loaded(FakeConn(2186), 2023, games)  # 89.96%
    check("just under the floor is refused", "did not raise", "SystemExit")
except SystemExit:
    check("just under the floor is refused", True, True)

# A properly loaded season passes. Not 100%: postponements never made up
# mean an exact match is unreachable, which is why the floor exists.
try:
    bf._require_postgame_loaded(FakeConn(2400), 2024, games)  # 98.8%
    check("fully loaded season passes", True, True)
except SystemExit as exc:
    check("fully loaded season passes", f"raised: {exc}", True)

# Degenerate input: no schedule at all should abort, not divide by zero.
try:
    bf._require_postgame_loaded(FakeConn(0), 2025, [])
    check("empty schedule aborts", "did not raise", "SystemExit")
except SystemExit:
    check("empty schedule aborts", True, True)
except ZeroDivisionError:
    check("empty schedule aborts", "ZeroDivisionError", "SystemExit")

print("\n--skip-postgame flag is wired into the CLI")
parser = bf.build_parser()
args = parser.parse_args(["--seasons", "2021", "2022", "--skip-postgame"])
check("flag parses", args.skip_postgame, True)
check("flag defaults to off", parser.parse_args([]).skip_postgame, False)
check("other flags still work", parser.parse_args(["--skip-pitches"]).skip_pitches, True)
check("seasons still parse", args.seasons, [2021, 2022])

# ---------------------------------------------------------------- guard 2
print("\nOAA lookahead guard")

import pipelines.savant_client as sc  # noqa: E402

try:
    sc.get_team_outs_above_average(2025, through_date="2025-05-01")
    check("through_date is refused", "did not raise", "ValueError")
except ValueError as exc:
    check("through_date is refused", True, True)
    msg = str(exc)
    check("error explains it is ignored by Savant", "ignores" in msg, True)
    check("error names the consequence", "lookahead" in msg.lower(), True)

# The unbounded call must still work -- it is legitimate season-final data,
# just not usable as an as-of feature. Stubbed so this stays offline.
import pandas as pd  # noqa: E402

calls = []


def fake_read(url, params, what):
    calls.append(params)
    return pd.DataFrame([{"display_team_name": "Cubs", "outs_above_average": 5}])


sc._read_savant_csv_optional = fake_read
sc.clear_caches()
df = sc.get_team_outs_above_average(2025)
check("unbounded call still returns data", len(df), 1)
check("uses the Fielder grouping, not the dead Team one", calls[0]["type"], "Fielder")
check("sends no date params", "startDate" in calls[0] or "endDate" in calls[0], False)

# ---------------------------------------------------------------- guard 3
print("\nteam_form no longer calls the OAA endpoint at all")
import pipelines.games.team_form as tf  # noqa: E402

src = Path(tf.__file__).read_text(encoding="utf-8")
import_lines = [
    ln for ln in src.splitlines() if ln.startswith(("import ", "from ")) and "savant" in ln
]
check("team_form imports nothing from savant_client", import_lines, [])
check(
    "and the symbol is not called anywhere in it",
    any(
        "get_team_outs_above_average(" in ln
        for ln in src.splitlines()
        if not ln.lstrip().startswith("#")
    ),
    False,
)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all guard tests passed")
