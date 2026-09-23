"""Offline tests for the nightly pitch merge (pitch_store.merge_day) and the
daily pull's pitch-coverage guard.

Written 23 Sep 2026 after the first real nightly run failed. merge_day had
only ever been exercised against files it wrote itself, never against a
season file the backfill built, and those carry a stray 22nd `season`
column (DuckDB hive partitioning), so the merge's UNION failed. Test 1 is
that exact case, built with the backfill's own write_chunk and
consolidate_season rather than a hand-made fixture.

No network, no Postgres. Run: python tests/test_merge_day.py
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from pipelines import pitch_store
from pipelines.player_form.sql_helpers import pitcher_events_query

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


def row(game_id, ab, pn, **overrides):
    r = {c: None for c in pitch_store.PITCH_COLUMNS}
    r.update(
        game_id=game_id, at_bat_id=ab, pitch_number=pn, pitcher_id=500, batter_id=600,
        inning=1, balls=0, strikes=0, pitch_type="FF", release_speed=95.0,
        spin_rate=2300, plate_x=0.1, plate_z=2.5, sz_top=3.4, sz_bot=1.6,
        pitch_result="ball", exit_velocity=None, launch_angle=None,
        events=None, bb_type=None, hit_location=None,
    )
    r.update(overrides)
    return r


def schema_of(path):
    return [(f.name, str(f.type)) for f in pq.read_schema(path)]


CANONICAL = [
    (c, {"BIGINT": "int64", "DOUBLE": "double", "VARCHAR": "string"}[t])
    for c, t in pitch_store.PITCH_TYPES.items()
]


def count(path):
    return duckdb.sql(f"select count(*) from read_parquet('{path}', hive_partitioning=false)").fetchone()[0]


def backfill_style_file(tmp, season, rows):
    """A season file built exactly the way scripts/backfill.py builds one."""
    pitch_store.write_chunk(rows, season, 1, root=tmp)
    return pitch_store.consolidate_season(season, root=tmp)


def legacy_hive_file(tmp, season, rows):
    """A file shaped like the 2021-2026 release files already on GitHub:
    written before the consolidate fix, so it carries the extra `season`
    column. Built directly since consolidate_season no longer produces it."""
    out = pitch_store.season_file(season, tmp)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = {c: [r.get(c) for r in rows] for c in pitch_store.PITCH_COLUMNS}
    cols["season"] = [season] * len(rows)
    pq.write_table(pa.table(cols), out, compression="zstd")
    return out


def main():
    base = Path(tempfile.mkdtemp())
    try:
        print("1. merge into a release-shaped file with the stray `season` column (the 23 Sep failure)")
        tmp = base / "t1"
        f = legacy_hive_file(tmp, 2026, [row(1, 1, 1), row(1, 1, 2)])
        check("legacy file has 22 columns", len(pq.read_schema(f)), 22)
        pitch_store.merge_day(2026, [row(2, 1, 1)], root=tmp)
        check("merged row count", count(f), 3)
        check("merged schema is exactly the canonical 21 columns", schema_of(f), CANONICAL)

        print("2. consolidate_season no longer adds the `season` column")
        tmp = base / "t2"
        f = backfill_style_file(tmp, 2026, [row(1, 1, 1)])
        check("consolidated schema", schema_of(f), CANONICAL)
        pitch_store.merge_day(2026, [row(2, 1, 1)], root=tmp)
        check("merge into freshly consolidated file", count(f), 2)

        print("3. re-pulled pitches replace the old version, no duplicates")
        tmp = base / "t3"
        f = legacy_hive_file(tmp, 2026, [row(1, 1, 1, pitch_result="ball"), row(1, 1, 2)])
        pitch_store.merge_day(2026, [row(1, 1, 1, pitch_result="called_strike")], root=tmp)
        check("row count unchanged by a re-pull", count(f), 2)
        got = duckdb.sql(
            f"select pitch_result from read_parquet('{f}') where game_id=1 and at_bat_id=1 and pitch_number=1"
        ).fetchall()
        check("new version wins", got, [("called_strike",)])

        print("4. a day where a column is entirely empty (pyarrow would infer type NULL)")
        tmp = base / "t4"
        f = legacy_hive_file(tmp, 2026, [row(1, 1, 1, exit_velocity=101.2, launch_angle=25, hit_location=8)])
        pitch_store.merge_day(2026, [row(2, 1, 1), row(2, 1, 2)], root=tmp)
        check("schema still canonical", schema_of(f), CANONICAL)
        check("existing non-null value preserved",
              duckdb.sql(f"select exit_velocity from read_parquet('{f}') where game_id=1").fetchone()[0], 101.2)

        print("5. spin rate / launch angle arriving as floats")
        tmp = base / "t5"
        f = legacy_hive_file(tmp, 2026, [row(1, 1, 1)])
        pitch_store.merge_day(2026, [row(2, 1, 1, spin_rate=2412.0, launch_angle=18.0)], root=tmp)
        check("schema still canonical", schema_of(f), CANONICAL)
        check("float spin rate stored as integer",
              duckdb.sql(f"select spin_rate from read_parquet('{f}') where game_id=2").fetchone()[0], 2412)

        print("6. no new rows leaves the file untouched")
        tmp = base / "t6"
        f = legacy_hive_file(tmp, 2026, [row(1, 1, 1)])
        before = f.stat().st_mtime_ns
        check("returns existing path", pitch_store.merge_day(2026, [], root=tmp), f)
        check("file not rewritten", f.stat().st_mtime_ns, before)

        print("7. no existing file (first run of a season) creates a canonical one")
        tmp = base / "t7"
        f = pitch_store.merge_day(2026, [row(1, 1, 2), row(1, 1, 1)], root=tmp)
        check("created", f.exists(), True)
        check("schema", schema_of(f), CANONICAL)
        check("sorted by natural key",
              duckdb.sql(f"select pitch_number from read_parquet('{f}')").fetchall(), [(1,), (2,)])

        print("8. shrink guard refuses to replace the file")
        tmp = base / "t8"
        # An existing file with a duplicated natural key, plus an incoming
        # re-pull of a pitch it already has: the merge nets out to fewer rows
        # than it started with, which the guard must treat as "something is
        # wrong". (A brand-new incoming row would add one back and mask it.)
        f = legacy_hive_file(tmp, 2026, [row(1, 1, 1), row(1, 1, 1), row(1, 1, 2)])
        before = f.read_bytes()
        try:
            pitch_store.merge_day(2026, [row(1, 1, 2)], root=tmp)
            check("raised", False, True)
        except RuntimeError as e:
            check("raised", "shrink" in str(e), True)
        check("original file untouched", f.read_bytes() == before, True)
        check("no leftover staged file", f.with_suffix(".parquet.new").exists(), False)
        check("no leftover incoming file",
              (f.parent / f"{f.stem}_incoming.parquet").exists(), False)

        print("9. PitchSource over a merged file answers the production pitcher query")
        tmp = base / "t9"
        legacy_hive_file(tmp, 2026, [row(1, 1, 1, release_speed=94.0), row(1, 1, 2, release_speed=96.0)])
        pitch_store.merge_day(2026, [row(2, 1, 1, release_speed=99.0)], root=tmp)
        games = [
            {"game_id": 1, "date": date(2026, 9, 21), "season": 2026, "umpire_id": 7},
            {"game_id": 2, "date": date(2026, 9, 22), "season": 2026, "umpire_id": 7},
        ]
        src = pitch_store.open_pitch_source(2026, games, root=tmp)
        try:
            with src.cursor() as cur:
                cur.execute(pitcher_events_query(days=None),
                            {"player_id": 500, "season": 2026, "as_of_date": "2026-09-23"})
                check("sees both days before 23 Sep", len(cur.fetchall()), 3)
                cur.execute(pitcher_events_query(days=None),
                            {"player_id": 500, "season": 2026, "as_of_date": "2026-09-22"})
                check("no lookahead: 22 Sep excluded when scoring 22 Sep", len(cur.fetchall()), 2)
        finally:
            src.close()

        print("10. daily_pull's pitch-coverage guard")
        spec = importlib.util.spec_from_file_location(
            "daily_pull", Path(__file__).resolve().parent.parent / "scripts" / "daily_pull.py")
        daily_pull = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(daily_pull)

        class FakeCursor:
            def __init__(self, n): self.n = n
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a, **k): pass
            def fetchone(self): return (self.n,)

        class FakeConn:
            def __init__(self, n): self.n = n
            def cursor(self): return FakeCursor(self.n)

        src = pitch_store.open_pitch_source(2026, games, root=tmp)
        try:
            # 1 game with pitches before 22 Sep. 1 completed game: passes.
            daily_pull._check_pitch_coverage(FakeConn(1), src, 2026, "2026-09-22")
            check("passes when coverage is complete", True, True)
            # 1 game with pitches, 50 completed: a one-day file. Must stop.
            try:
                daily_pull._check_pitch_coverage(FakeConn(50), src, 2026, "2026-09-22")
                check("raises on thin pitch file", False, True)
            except RuntimeError as e:
                check("raises on thin pitch file", "covers only 1 of 50" in str(e), True)
            daily_pull._check_pitch_coverage(FakeConn(0), src, 2026, "2026-03-26")
            check("opening day (no prior games) passes", True, True)
        finally:
            src.close()
    finally:
        shutil.rmtree(base, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("all merge_day checks passed")


if __name__ == "__main__":
    main()
