"""Offline tests for the catcher framing feature (design E18, 3 Oct 2026).

No network, no DB, no data files: tiny hand-built tables with a flat expected
call of 0.5, so a called strike is +0.5 and a ball -0.5. Catcher 11 plays for
team 1 and gets every borderline call (all strikes); catcher 22 plays for
team 2 and gets none (all balls), so the league average residual is 0.

Run: python tests/test_framing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from pipelines.features import framing  # noqa: E402

failures = []


def check(label, got, want, tol=1e-9):
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}" + ("" if ok else f", want {want!r}"))
    if not ok:
        failures.append(label)


def make(days=(1, 2, 3), cs_later=1, with_prior=False, c_missing=False, resume_game1=False):
    """Team 1 (catcher 11) hosts team 2 (catcher 22) on each day of 2022-06-<day>;
    100 taken pitches a game, all credited to the fielding catcher of that half
    (team 1 fields when team 2 bats). Catcher 11 gets strikes, 22 gets balls."""
    con = duckdb.connect()
    games, taken, lu, tg = [], [], [], []
    seasons = [(2021, "2021-06-%02d")] if with_prior else []
    seasons.append((2022, "2022-06-%02d"))
    gid = 0
    for season, fmt in seasons:
        for d in days:
            gid += 1
            date = fmt % d
            games.append((gid, season, date))
            for team, c in ((1, 11), (2, 22)):
                if not (c_missing and season == 2022 and d == days[-1] and team == 1):
                    lu.append((gid, team, 2, c, "C"))
                lu.append((gid, team, 1, 900 + team, "SS"))
            for fld, cs in ((1, 1), (2, 0)):
                # a later day's outcomes are flipped by cs_later=0 to test look-ahead
                v = cs if not (season == 2022 and d == days[-1]) else (cs if cs_later else 1 - cs)
                taken.append((gid, season, date, fld, 0, 0, 0, 0, v, 100))
            tg.append((gid, season, date, 2, 1))   # team 2 bats, team 1 fields
            tg.append((gid, season, date, 1, 2))
    con.register("g_df", pd.DataFrame(games, columns=["game_id", "season", "date"]))
    con.execute("create table games as select * from g_df")
    con.register("lu_df", pd.DataFrame(lu, columns=["game_id", "team_id", "slot", "player_id", "pos"]))
    con.execute("create table lineup as select * from lu_df")
    t = pd.DataFrame(taken, columns=["game_id", "season", "data_date", "fld_team", "sd", "cg", "ix", "iz", "cs", "n"])
    con.register("t_df", t)
    # one row per pitch group: expand n pitches as a weight-free summary by repeating n times
    con.execute("""create table ump_taken as
        select game_id, season, data_date::date as data_date, null::int as umpire_id, fld_team, sd, cg, cs, ix, iz
        from t_df, range(100) r(i)""")
    if resume_game1:   # game 1 (scheduled day 1) is finished on day 3: its pitches are thrown then
        con.execute("update ump_taken set data_date = date '2022-06-03' where game_id = 1")
    con.execute("create table ump_grid as select 0 as sd, 0 as cg, 0 as ix, 0 as iz, 0.5 as p_cs")
    con.execute("""create table ump_game as select t.game_id, t.season, t.data_date, null::int as umpire_id,
        sum(t.cs - 0.5) as r, count(*) as n from ump_taken t group by all""")
    con.execute("create table ump_lg_day as select season, data_date, sum(r) as r, sum(n) as n from ump_game group by all")
    con.execute("create table ump_lg_season as select season, sum(r) / sum(n) as l from ump_game group by all")
    con.register("tg_df", pd.DataFrame(tg, columns=["game_id", "season", "date", "bat_team", "fld_team"]))
    con.execute("create table tg as select game_id, season, date::date as date, bat_team, fld_team from tg_df")
    return con


def feat(con, game_id, bat_team):
    r = con.execute("select frm_runs_opp, frm_missing from frm_feat where game_id=? and bat_team=?",
                    [game_id, bat_team]).fetchone()
    return r


print("test_framing")
con = make()
framing.framing_features(con, "tg", "P2")
# game 3 (day 3), team 2 bats against catcher 11: days 1 and 2 count, 200 pitches, residual +50 each
want = framing.RUNS_PER_POINT * 100.0 * 100.0 / (200 + 1000)
check("day 3, bat team 2 faces catcher 11: two prior days", feat(con, 3, 2)[0], want)
check("day 3, bat team 2 faces catcher 11: not missing", feat(con, 3, 2)[1], False)
check("day 3, bat team 1 faces catcher 22 (sees balls): negative mirror", feat(con, 3, 1)[0], -want)
check("day 1: no history, so 0", feat(con, 1, 2)[0], 0.0)
check("day 2: one prior day", feat(con, 2, 2)[0], framing.RUNS_PER_POINT * 100.0 * 50.0 / (100 + 1000))

# no look-ahead: flip the outcomes of day 3 itself, day-3 features must not move
con2 = make(cs_later=0)
framing.framing_features(con2, "tg", "P2")
check("no look-ahead: day 3 unchanged when day 3 outcomes flip", feat(con2, 3, 2)[0], want)

# mutation: counting the game's own day changes day 3, so the leak test can catch it
framing.SAME_DAY = True
framing.framing_features(con, "tg", "P2")
mut = feat(con, 3, 2)[0]
framing.SAME_DAY = False
check("mutation SAME_DAY changes the feature", mut != want, True)
framing.framing_features(con, "tg", "P2")

# prior season: 2021 days carry weight 0.8 and are centred on 2021's league average (0)
con3 = make(with_prior=True)
framing.framing_features(con3, "tg", "P2")
gid_2022_d1 = 4
w = 0.8
want_p = framing.RUNS_PER_POINT * 100.0 * (w * 150.0) / (w * 300 + 1000)
check("prior season weighted 0.8 on day 1 of next season", feat(con3, gid_2022_d1, 2)[0], want_p)

# missing C: feature 0 and flagged
con4 = make(c_missing=True)
framing.framing_features(con4, "tg", "P2")
r = feat(con4, 3, 2)   # team 1 (fielding) has no C in game 3
check("no catcher in the lineup: 0", r[0], 0.0)
check("no catcher in the lineup: flagged", r[1], True)

# a resumed game: scheduled day 1, pitches thrown day 3. Game 2 (day 2) must not see them
con5 = make(resume_game1=True)
framing.framing_features(con5, "tg", "P2")
check("resumed game's pitches are not used before they were thrown", feat(con5, 2, 2)[0], 0.0)
framing.DATE_CUTOFF = True
framing.framing_features(con5, "tg", "P2")
mut = feat(con5, 2, 2)[0]
framing.DATE_CUTOFF = False
check("mutation DATE_CUTOFF would use them", mut != 0.0, True)

# P1: zeros and missing
framing.framing_features(con, "tg", "P1")
rows = con.execute("select count(*), sum((frm_runs_opp <> 0)::int), sum(frm_missing::int) from frm_feat").fetchone()
check("P1: all rows 0 and missing", (rows[1], rows[2] == rows[0]), (0, True))

# fixed conversion
check("conversion is 76 x 0.14 / 100", framing.RUNS_PER_POINT, 0.1064)

print("FAILED: " + ", ".join(failures) if failures else "all passed")
sys.exit(1 if failures else 0)
