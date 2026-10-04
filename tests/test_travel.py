"""Offline tests for the travel feature (design E17, 3 Oct 2026).

No network, no DB, no data files: tiny hand-built schedules whose lags are
worked out by hand. Venue ids used: 680 Seattle (Pacific, -7), 3313 New York
(Eastern, -4), 17 Chicago (Central, -5), 5381 London (international),
99999 (not in the table).

Run: python tests/test_travel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from pipelines.features import travel  # noqa: E402

failures = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}" + ("" if ok else f", want {want!r}"))
    if not ok:
        failures.append(label)


def sched(rows):
    """rows: (game_id, season, 'YYYY-MM-DD', fp_hour, venue_id) for ONE team."""
    return pd.DataFrame([{"game_id": g, "season": s, "date": pd.Timestamp(d), "fp_sched": h * 3600.0 + pd.Timestamp(d).value // 10**9,
                          "venue_id": v, "team": 1} for g, s, d, h, v in rows])


def lags(rows):
    out = travel.team_lags(sched(rows))
    return dict(zip(out.game_id, out.lag_e))


SEA, NYY, CHI, LON, UNK = 680, 3313, 17, 5381, 99999

print("Seattle to New York, no off day: 3, 2, 1, 0")
r = lags([(1, 2022, "2022-06-01", 19, SEA), (2, 2022, "2022-06-02", 19, NYY), (3, 2022, "2022-06-03", 19, NYY),
          (4, 2022, "2022-06-04", 19, NYY), (5, 2022, "2022-06-05", 19, NYY)])
check("opening game at Seattle", r[1], 0.0)
check("day after the flight", r[2], 3.0)
check("second day in New York", r[3], 2.0)
check("third day", r[4], 1.0)
check("fourth day", r[5], 0.0)

print("Seattle to New York with an off day after the flight")
r = lags([(1, 2022, "2022-06-01", 19, SEA), (2, 2022, "2022-06-03", 19, NYY), (3, 2022, "2022-06-04", 19, NYY)])
check("arrival game after an off day", r[2], 2.0)
check("next day", r[3], 1.0)

print("Round trip: west is never counted as east lag")
r = lags([(1, 2022, "2022-06-01", 19, NYY), (2, 2022, "2022-06-02", 19, SEA), (3, 2022, "2022-06-03", 19, SEA),
          (4, 2022, "2022-06-04", 19, NYY)])
check("flying west", r[2], 0.0)
# arrival day moves nothing, the second day on the coast moves the clock 1 hour toward Pacific (-4 to -5);
# back in New York (-4) the clock is 1 hour behind local time
check("back east after 2 days on the coast", r[4], 1.0)

print("Central to Eastern is a 1 hour shift")
r = lags([(1, 2022, "2022-06-01", 19, CHI), (2, 2022, "2022-06-02", 19, NYY), (3, 2022, "2022-06-03", 19, NYY)])
check("one zone east", r[2], 1.0)
check("recovered a day later", r[3], 0.0)

print("Doubleheader: game ids run against first-pitch order, the later first pitch moves nothing")
r = lags([(1, 2022, "2022-06-01", 19, SEA), (20, 2022, "2022-06-02", 19, NYY),   # game id 20 played second
          (10, 2022, "2022-06-02", 23, NYY)])
# game 20 has the EARLIER first pitch (19) so it is first; game 10 (23) second; both at New York
check("first game of the doubleheader", r[20], 3.0)
check("second game: same day, no movement", r[10], 3.0)

print("New season resets the body clock")
r = lags([(1, 2021, "2021-09-30", 19, SEA), (2, 2022, "2022-04-08", 19, NYY)])
check("first game of the next season", r[2], 0.0)

print("Venue not in the table is skipped as if the game had not happened")
r = lags([(1, 2022, "2022-06-01", 19, SEA), (2, 2022, "2022-06-02", 19, UNK), (3, 2022, "2022-06-03", 19, NYY)])
check("unknown venue game is not in the output", 2 in r, False)
check("lag across the gap as if Seattle then two days later New York", r[3], 2.0)

print("International game: lag 0, and the recursion still runs through it")
r = lags([(1, 2022, "2022-06-01", 19, NYY), (2, 2022, "2022-06-02", 19, LON), (3, 2022, "2022-06-03", 19, LON),
          (4, 2022, "2022-06-04", 19, NYY)])
check("London game", r[2], 0.0)
check("back in New York (west of the London clock)", r[4], 0.0)

print("No look-ahead: dropping every later game leaves earlier lags alone")
rows = [(1, 2022, "2022-06-01", 19, SEA), (2, 2022, "2022-06-02", 19, NYY), (3, 2022, "2022-06-03", 19, NYY),
        (4, 2022, "2022-06-04", 19, SEA), (5, 2022, "2022-06-05", 19, NYY)]
full, cut = lags(rows), lags(rows[:3])
check("lags of games 1 to 3 with and without the future", [full[i] for i in (1, 2, 3)], [cut[i] for i in (1, 2, 3)])
travel.USE_NEXT_GAME = True
mut = lags(rows)
travel.USE_NEXT_GAME = False
check("the next-venue mutation changes them (so the leak test can catch it)", [mut[i] for i in (1, 2, 3)] != [full[i] for i in (1, 2, 3)], True)

print("travel_features: own and opposing lag, cancelled game, unknown venue, international")
con = duckdb.connect()
G = [  # game_id, season, date, fp (epoch hours added), venue, home, away, status
    (1, 2022, "2022-06-01", 19, SEA, 10, 20, "completed"),     # team 20 hosts nobody here: team 20 is away at Seattle
    (2, 2022, "2022-06-02", 19, NYY, 30, 20, "completed"),     # team 20 is in New York the next day: lag 3
    (3, 2022, "2022-06-03", 19, NYY, 30, 20, None),            # cancelled: not a game
    (4, 2022, "2022-06-04", 19, UNK, 40, 20, "completed"),     # unknown venue
    (5, 2022, "2022-06-05", 19, LON, 50, 20, "completed"),     # international
]
g = pd.DataFrame([{"game_id": a, "season": b, "date": pd.Timestamp(c).date(), "fp_sched": pd.Timestamp(c) + pd.Timedelta(hours=d),
                   "venue_id": e, "home_team": f, "away_team": h, "status": s} for a, b, c, d, e, f, h, s in G])
con.register("g_df", g)
con.execute("create table games as select * from g_df")
con.execute("create table tg as select game_id, bat_team, fld_team from (select game_id, home_team bat_team, away_team fld_team from games union all select game_id, away_team, home_team from games)")
travel.travel_features(con, "tg", "P2")
res = con.execute("select * from trv_feat order by game_id, bat_team").fetchdf().set_index(["game_id", "bat_team"])
check("game 2, away team 20 batting: own lag 3, opposing 0", (res.loc[(2, 20), "lag_e_own"], res.loc[(2, 20), "lag_e_opp"]), (3.0, 0.0))
check("game 2, home team 30 batting: own 0, opposing (team 20) 3", (res.loc[(2, 30), "lag_e_own"], res.loc[(2, 30), "lag_e_opp"]), (0.0, 3.0))
check("game 2 is not flagged missing", bool(res.loc[(2, 20), "trv_missing"]), False)
check("cancelled game is flagged missing, lags 0", (bool(res.loc[(3, 20), "trv_missing"]), res.loc[(3, 20), "lag_e_own"]), (True, 0.0))
check("unknown venue is flagged missing, lags 0", (bool(res.loc[(4, 20), "trv_missing"]), res.loc[(4, 20), "lag_e_own"]), (True, 0.0))
check("international game is flagged missing, lags 0", (bool(res.loc[(5, 20), "trv_missing"]), res.loc[(5, 20), "lag_e_own"]), (True, 0.0))
check("row count: two rows per game", len(res), 10)

print("Live use: a game later than every completed one is kept and gets its lag")
con2 = duckdb.connect()
G2 = [(1, 2022, "2022-06-01", 19, SEA, 10, 20, "completed"), (2, 2022, "2022-06-02", 19, NYY, 30, 20, None)]  # game 2 is today, no result yet
g2 = pd.DataFrame([{"game_id": a, "season": b, "date": pd.Timestamp(c).date(), "fp_sched": pd.Timestamp(c) + pd.Timedelta(hours=d),
                    "venue_id": e, "home_team": f, "away_team": h, "status": s_} for a, b, c, d, e, f, h, s_ in G2])
con2.register("g2_df", g2)
con2.execute("create table games as select * from g2_df")
con2.execute("create table tg as select game_id, home_team bat_team, away_team fld_team from games union all select game_id, away_team, home_team from games")
travel.travel_features(con2, "tg", "P2")
r2 = con2.execute("select * from trv_feat where bat_team = 30").fetchdf().iloc[0]
check("today's game with no result yet gets the fielding team's lag 3", (r2.lag_e_opp, bool(r2.trv_missing)), (3.0, False))

print("A nominal home game played at another venue uses the venue (Blue Jays at Angel Stadium, 2021-08-10)")
r = lags([(1, 2021, "2021-08-08", 19, NYY), (2, 2021, "2021-08-10", 13, 1), (3, 2021, "2021-08-10", 17, 1)])
check("Eastern to Pacific is westward: no east lag", (r[2], r[3]), (0.0, 0.0))

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all travel tests passed")
