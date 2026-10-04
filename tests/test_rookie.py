"""Offline tests for rookie translations (design E19, 4 Oct 2026).

No network, no data files. Checks the season window (2020 skipped, prior
seasons only), the frozen constants' shape, and that without minor-league
inputs the rookie priors are left alone.

Run: python tests/test_rookie.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402

from pipelines.features import rookie  # noqa: E402

failures = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}" + ("" if ok else f", want {want!r}"))
    if not ok:
        failures.append(label)


print("test_rookie")
check("2021 looks back to 2019, 2018, 2017", rookie._prior_seasons(2021), [(2019, 0.8), (2018, 0.64), (2017, 0.48)])
check("2022 skips 2020", rookie._prior_seasons(2022), [(2021, 0.8), (2019, 0.64), (2018, 0.48)])
check("2024 uses the three seasons before", rookie._prior_seasons(2024), [(2023, 0.8), (2022, 0.64), (2021, 0.48)])
check("never the season itself", all(s < S for S in range(2018, 2027) for s, _ in rookie._prior_seasons(S)), True)
rookie.CURRENT_SEASON_LINE = True
check("mutation switch adds the season itself", rookie._prior_seasons(2022)[0], (2022, 1.0))
rookie.CURRENT_SEASON_LINE = False
check("frozen constants for 6 rates x 2 levels", len(rookie.FROZEN), 12)
check("frozen keys", sorted({k[0] for k in rookie.FROZEN}), ["hbb", "hk", "hwoba", "pbb", "pgb", "pk"])
check("fit seasons have no 2022 or later", max(rookie.FIT_SEASONS) <= 2021, True)
check("Mexican League list has 16 teams", len(set(rookie.MEXICAN_LEAGUE_TEAM_IDS)), 16)
check("switches at defaults", (rookie.CURRENT_SEASON_LINE, rookie.POOL_LEAGUE, rookie.REFIT_ALL), (False, False, False))
check("off by default (E19 not adopted)", rookie.ENABLED, False)
con = duckdb.connect()
note = rookie.build_rookie_adj(con)
check("off: empty table", con.execute("select count(*) from rookie_adj").fetchone()[0], 0)
rookie.ENABLED = True
con = duckdb.connect()
note = rookie.build_rookie_adj(con)
rookie.ENABLED = False
check("no minor-league inputs: empty table", con.execute("select count(*) from rookie_adj").fetchone()[0], 0)
check("no minor-league inputs: note says unchanged", "unchanged" in note["note"], True)

print("FAILED: " + ", ".join(failures) if failures else "all passed")
sys.exit(1 if failures else 0)
