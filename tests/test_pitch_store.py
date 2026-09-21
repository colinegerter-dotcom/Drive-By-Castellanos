"""Offline tests for the Parquet + DuckDB pitch path.

No network, no Postgres. Builds a tiny real Parquet file on disk, opens a
real DuckDB over it, and runs the actual production queries.

The test that matters most is the lookahead one. Moving these queries from
Postgres to DuckDB meant rewriting the `g.date < as_of_date` guard, and that
guard is the only thing preventing a game's own result from feeding the
prediction of that same game. A dialect port is exactly the kind of change
that breaks it silently, so it gets checked explicitly rather than assumed.

Run: python tests/test_pitch_store.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines import pitch_store
from pipelines.player_form.sql_helpers import (
    batter_events_query,
    pitcher_events_query,
    umpire_pitches_query,
    window_start,
)

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


BATTER, PITCHER, UMP = 660271, 601713, 900001

# Three games a week apart. The batter and pitcher appear in all three.
GAMES = [
    {"game_id": 1001, "date": date(2025, 5, 1), "season": 2025, "umpire_id": UMP},
    {"game_id": 1002, "date": date(2025, 5, 8), "season": 2025, "umpire_id": UMP},
    {"game_id": 1003, "date": date(2025, 6, 20), "season": 2025, "umpire_id": UMP},
    # A different season, to prove the season filter bites.
    {"game_id": 2001, "date": date(2024, 5, 1), "season": 2024, "umpire_id": UMP},
]


def pitch(game_id, n, *, events=None, ev=None, la=None, speed=None, result="ball"):
    return {
        "game_id": game_id, "at_bat_id": n, "pitch_number": 1,
        "pitcher_id": PITCHER, "batter_id": BATTER,
        "inning": 1, "balls": 0, "strikes": 0,
        "pitch_type": "FF", "release_speed": speed, "spin_rate": 2200,
        "plate_x": 0.1, "plate_z": 2.5, "sz_top": 3.4, "sz_bot": 1.6,
        "pitch_result": result, "exit_velocity": ev, "launch_angle": la,
        "events": events, "bb_type": None, "hit_location": None,
    }


ROWS = (
    [pitch(1001, i, events="strikeout" if i == 1 else None, ev=100.0, la=28, speed=95.0) for i in range(1, 4)]
    + [pitch(1002, i, events="walk" if i == 1 else None, ev=90.0, la=10, speed=93.0) for i in range(1, 3)]
    + [pitch(1003, i, events="single", ev=105.0, la=27, speed=91.0) for i in range(1, 5)]
    + [pitch(2001, i, events="strikeout", ev=80.0, la=5, speed=89.0) for i in range(1, 3)]
)

tmp = Path(tempfile.mkdtemp(prefix="pitchstore-"))
try:
    print("\n1. write + consolidate")
    pitch_store.write_chunk(ROWS[:5], 2025, 1, root=tmp)
    pitch_store.write_chunk(ROWS[5:9], 2025, 2, root=tmp)
    check("empty chunk writes nothing", pitch_store.write_chunk([], 2025, 3, root=tmp), None)
    pitch_store.write_chunk([r for r in ROWS if r["game_id"] == 2001], 2024, 1, root=tmp)

    out = pitch_store.consolidate_season(2025, root=tmp)
    check("consolidated file exists", out.exists(), True)

    src = pitch_store.open_pitch_source(2025, GAMES, root=tmp)
    check("all 2025 rows present", src.query("select count(*) from pitches").fetchone()[0], 9)

    print("\n2. THE LOOKAHEAD GUARD (g.date < as_of_date)")
    with src.cursor() as cur:
        # Scoring game 1002 on 2025-05-08: game 1001 (May 1) counts,
        # 1002 itself must NOT, and 1003 (June) is in the future.
        cur.execute(batter_events_query(days=None),
                    {"player_id": BATTER, "season": 2025, "as_of_date": "2025-05-08"})
        rows = cur.fetchall()
    check("only the PRIOR game's pitches are visible", len(rows), 3)
    check("...and none of them are from the game being scored",
          sorted({r[4] for r in rows}), [1001])

    with src.cursor() as cur:
        cur.execute(batter_events_query(days=None),
                    {"player_id": BATTER, "season": 2025, "as_of_date": "2025-05-01"})
        check("opening game sees nothing before it", len(cur.fetchall()), 0)

    with src.cursor() as cur:
        cur.execute(batter_events_query(days=None),
                    {"player_id": BATTER, "season": 2025, "as_of_date": "2025-12-31"})
        check("end of season sees every 2025 pitch", len(cur.fetchall()), 9)
        check("...and no 2024 pitches leak across the season filter",
              src.query("select count(*) from pitches p join games g using (game_id) "
                        "where g.season = 2024").fetchone()[0], 0)

    print("\n3. rolling 30-day window")
    check("window_start is inclusive and 30 days back",
          window_start("2025-06-20", 30), "2025-05-21")
    with src.cursor() as cur:
        # As of 2025-06-20, a 30-day window starts 2025-05-21. Games 1001
        # (May 1) and 1002 (May 8) are both older than that, so the window
        # is empty even though the season-to-date query would return 5 rows.
        cur.execute(batter_events_query(days=30),
                    {"player_id": BATTER, "season": 2025, "as_of_date": "2025-06-20",
                     "since_date": window_start("2025-06-20", 30)})
        check("30d window excludes games older than the window", len(cur.fetchall()), 0)
        cur.execute(batter_events_query(days=None),
                    {"player_id": BATTER, "season": 2025, "as_of_date": "2025-06-20"})
        check("...while season-to-date still sees them", len(cur.fetchall()), 5)

    print("\n4. column order matches what the summarize() functions unpack")
    with src.cursor() as cur:
        cur.execute(batter_events_query(days=None),
                    {"player_id": BATTER, "season": 2025, "as_of_date": "2025-05-08"})
        r = cur.fetchall()[0]
    # (pitch_result, events, exit_velocity, launch_angle, game_id, date)
    check("batter row is 6 wide", len(r), 6)
    check("batter[2] is exit_velocity", r[2], 100.0)
    check("batter[4] is game_id", r[4], 1001)
    check("batter[5] is a date", isinstance(r[5], date), True)

    with src.cursor() as cur:
        cur.execute(pitcher_events_query(days=None),
                    {"player_id": PITCHER, "season": 2025, "as_of_date": "2025-05-08"})
        r = cur.fetchall()[0]
    # (pitch_result, events, bb_type, release_speed, game_id, date)
    check("pitcher row is 6 wide", len(r), 6)
    check("pitcher[3] is release_speed", r[3], 95.0)
    check("pitcher[4] is game_id", r[4], 1001)

    with src.cursor() as cur:
        cur.execute(umpire_pitches_query(),
                    {"umpire_id": UMP, "season": 2025, "as_of_date": "2025-12-31"})
        r = cur.fetchall()[0]
    # (pitch_result, plate_x, plate_z, sz_top, sz_bot, game_id)
    check("umpire row is 6 wide", len(r), 6)
    check("umpire[5] is game_id", r[5] in (1001, 1002, 1003), True)

    print("\n5. SQL-side K/BB counting matches counting in Python")
    events_sql = src.query(
        "select count(*), count(*) filter (where p.events='strikeout'), "
        "count(*) filter (where p.events='walk') "
        "from pitches p join games g using (game_id) "
        "where g.season=2025 and p.events is not null"
    ).fetchone()
    py_events = [r["events"] for r in ROWS if r["game_id"] != 2001 and r["events"]]
    check("total events agree", events_sql[0], len(py_events))
    check("strikeouts agree", events_sql[1], sum(1 for e in py_events if e == "strikeout"))
    check("walks agree", events_sql[2], sum(1 for e in py_events if e == "walk"))

    print("\n6. missing season fails loudly rather than returning empty")
    try:
        pitch_store.open_pitch_source(2099, GAMES, root=tmp)
        check("missing season raises", False, True)
    except FileNotFoundError:
        check("missing season raises FileNotFoundError", True, True)

    src.close()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("FAILED: " + "; ".join(failures) if failures else "ALL CHECKS PASSED"))
sys.exit(1 if failures else 0)
