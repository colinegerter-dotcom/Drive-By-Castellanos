"""End-to-end check that the form builders work against the DuckDB pitch source.

test_pitch_store.py proves the queries are right. This proves the *callers*
are right: that build_starting_batter_form_row / build_starting_pitcher_form_row
/ the umpire functions can be handed a PitchSource instead of a psycopg2
connection and produce the same shape of output they always did.

That's the risk this port introduces. The form modules were written against
psycopg2 and are full of working, reviewed aggregation logic; the change
swapped the handle underneath them. If the cursor shim or the column order
were even slightly off, these functions would return plausible-looking
numbers computed from the wrong columns -- the worst kind of failure, because
nothing raises.

No network, no Postgres. Run: python tests/test_form_on_parquet.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines import pitch_store
from pipelines.player_form.starting_batter_form import build_starting_batter_form_row
from pipelines.player_form.starting_pitcher_form import build_starting_pitcher_form_row
from pipelines.player_form.umpire_stats import (
    build_umpire_stats_row,
    league_k_bb_rate,
    umpire_k_bb_rate,
)

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


BATTER, PITCHER, UMP = 660271, 601713, 900001
GAMES = [
    {"game_id": 1001, "date": date(2025, 5, 1), "season": 2025, "umpire_id": UMP},
    {"game_id": 1002, "date": date(2025, 5, 3), "season": 2025, "umpire_id": UMP},
    {"game_id": 1003, "date": date(2025, 5, 5), "season": 2025, "umpire_id": UMP},
]


def p(game_id, n, **kw):
    row = {
        "game_id": game_id, "at_bat_id": n, "pitch_number": 1,
        "pitcher_id": PITCHER, "batter_id": BATTER,
        "inning": 1, "balls": 0, "strikes": 0,
        "pitch_type": "FF", "release_speed": 94.0, "spin_rate": 2200,
        "plate_x": 0.1, "plate_z": 2.5, "sz_top": 3.4, "sz_bot": 1.6,
        "pitch_result": "hit_into_play", "exit_velocity": None, "launch_angle": None,
        "events": None, "bb_type": None, "hit_location": None,
    }
    row.update(kw)
    return row


# Game 1001: two barrels (EV>=98, 26<=LA<=30) out of four balls in play.
# Game 1002: no barrels. Game 1003 is the game being SCORED, and is loaded
# with extreme values -- if any of them show up in the output, the lookahead
# guard failed.
ROWS = (
    [p(1001, 1, exit_velocity=100.0, launch_angle=28, events="home_run"),
     p(1001, 2, exit_velocity=99.0, launch_angle=27, events="double"),
     p(1001, 3, exit_velocity=80.0, launch_angle=5, events="groundout"),
     p(1001, 4, exit_velocity=85.0, launch_angle=45, events="flyout"),
     p(1001, 5, pitch_result="swinging_strike", events="strikeout"),
     p(1001, 6, pitch_result="ball", events="walk")]
    + [p(1002, 1, exit_velocity=70.0, launch_angle=2, events="groundout"),
       p(1002, 2, exit_velocity=72.0, launch_angle=4, events="groundout"),
       p(1002, 3, pitch_result="swinging_strike", events="strikeout")]
    + [p(1003, i, exit_velocity=115.0, launch_angle=28, events="home_run",
         release_speed=105.0) for i in range(1, 6)]
)

tmp = Path(tempfile.mkdtemp(prefix="formparquet-"))
try:
    pitch_store.write_chunk(ROWS, 2025, 1, root=tmp)
    pitch_store.consolidate_season(2025, root=tmp)
    src = pitch_store.open_pitch_source(2025, GAMES, root=tmp)

    print("\n1. batter form row, scored for game 1003 on 2025-05-05")
    row = build_starting_batter_form_row(
        src, BATTER, 1003, "2025-05-05", 2025, "2016-08-13",
        prefetched={"season": {"atBats": 100, "hits": 30, "doubles": 6, "triples": 1,
                               "homeRuns": 8, "baseOnBalls": 12, "intentionalWalks": 1,
                               "hitByPitch": 2, "sacFlies": 1, "strikeOuts": 25,
                               "plateAppearances": 115},
                    "last30": {"atBats": 40, "hits": 12, "doubles": 2, "triples": 0,
                               "homeRuns": 3, "baseOnBalls": 5, "intentionalWalks": 0,
                               "hitByPitch": 1, "sacFlies": 0, "strikeOuts": 10,
                               "plateAppearances": 46},
                    "career_before_season": {"plateAppearances": 3000}},
    )

    # Balls in play before 2025-05-05: four from 1001 + two from 1002 = six.
    # Mean EV = (100+99+80+85+70+72)/6 = 84.333 -> 84.3
    check("avg_exit_velo_season over prior games only", row["avg_exit_velo_season"], 84.3)
    # Two barrels out of six balls in play = 33.3%
    check("barrel_pct_season", row["barrel_pct_season"], 33.3)
    check("mlb_pa_count = 3000 prior + 115 this season", row["mlb_pa_count"], 3115)

    # THE CRITICAL ONE: game 1003's 115 mph home runs must be invisible.
    if row["avg_exit_velo_season"] is not None and row["avg_exit_velo_season"] > 100:
        failures.append("LOOKAHEAD: the scored game's own pitches leaked into its form row")
        print("  FAIL LOOKAHEAD: the scored game leaked into its own form row")
    else:
        print("  ok   LOOKAHEAD: the scored game's own 115mph pitches are excluded")

    check("days_since_last_game (1002 was 2 days earlier)", row["days_since_last_game"], 2)

    print("\n2. pitcher form row, same game")
    prow = build_starting_pitcher_form_row(
        src, PITCHER, 1003, "2025-05-05", 2025, "2017-04-30",
        prefetched={"season": {"outs": 180, "era": 3.5, "battersFaced": 250,
                               "strikeOuts": 60, "baseOnBalls": 20},
                    "last30": {"outs": 60, "era": 2.9, "battersFaced": 90,
                               "strikeOuts": 25, "baseOnBalls": 6},
                    "career_before_season": {"outs": 2400}},
    )
    # Every prior pitch was thrown at 94.0; game 1003's 105s must not count.
    check("avg_velo_season from prior games only", prow["avg_velo_season"], 94.0)
    check("mlb_ip_count = (2400+180)/3", prow["mlb_ip_count"], 860.0)
    check("last start was game 1002", prow["pitch_count_last_start"], 3)
    check("days_rest since 1002", prow["days_rest"], 2)

    print("\n3. umpire stats")
    u_k, u_bb = umpire_k_bb_rate(src, UMP, 2025, "2025-05-05")
    lg_k, lg_bb = league_k_bb_rate(src, 2025, "2025-05-05")
    # Events before 05-05: HR, 2B, groundout, flyout, K, BB, groundout,
    # groundout, K = 9 events, 2 strikeouts, 1 walk.
    check("umpire K% = 2/9", u_k, 22.2)
    check("umpire BB% = 1/9", u_bb, 11.1)
    check("league matches umpire here (only one umpire in the fixture)", (lg_k, lg_bb), (u_k, u_bb))

    urow = build_umpire_stats_row(src, UMP, 2025, "2025-05-05")
    check("umpire row built", urow is not None, True)
    check("games_umpired counts prior games only", urow["games_umpired"], 2)

    print("\n4. an umpire with no prior games returns None, not a crash")
    check("unknown umpire", build_umpire_stats_row(src, 999999, 2025, "2025-05-05"), None)
    check("...and rates are None", umpire_k_bb_rate(src, 999999, 2025, "2025-05-05"), (None, None))

    src.close()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("FAILED: " + "; ".join(failures) if failures else "ALL CHECKS PASSED"))
sys.exit(1 if failures else 0)
