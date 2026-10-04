"""
Round 2 group 6: travel and rest, as built (design E17, drafted 3 Oct 2026
and revised after an independent pre-build review, before the build).

Only one piece of 5.7 item 6 is tested: eastward jet lag, as a body-clock lag
measured in hours. (Days since the last game and lineup rest days were left
out on purpose, design E17.) Everything here comes from the schedule: dates
and venues of games already played, plus the current game's own venue.

For each team, its completed games of every type, in date order (same date:
scheduled first pitch, then game id):

  0. Games counted: status completed, plus any game dated after the latest
     completed one (the game being predicted and the days after it, live).
  1. Each venue has a summer UTC offset (TZ_BY_VENUE; ET -4, CT -5, MT -6,
     PT and Arizona -7). A venue missing from the table is skipped and
     warned about; the recursion goes on as if that game had not happened.
  2. The team's body clock B starts each season on its first game's offset.
  3. Before the next game, g calendar days later, B moves toward the NEW
     venue's offset by up to ADAPT_PER_DAY hours for each day spent there:
     g days if the offset is unchanged, g - 1 if it changed (the team spends
     the day of arrival travelling). Same-date games (doubleheaders) move
     nothing.
  4. lag = venue offset - B, clipped to -3..+3. lag_e = max(lag, 0): hours
     of EASTWARD lag. Example: Seattle to New York with no off day gives 3,
     then 2, 1, 0 on the next days; with an off day after the flight, 2.
  5. Games at international sites (Seoul, Mexico City, London) get
     lag_e = 0 and trv_missing true (a 9- or 12-hour change is not a
     3-hour clip); the recursion still runs through them.

Columns written (table trv_feat, one row per key row):
  lag_e_own    hours of eastward lag of the batting team
  lag_e_opp    the same for the fielding team (the model feature)
  trv_missing  the game's venue has no offset, is international, or the
               game isn't a completed game; both lags are 0

Same values at both prediction points (schedule only). The module-level
switches below exist for the leak test's mutation check and must stay at
their defaults.
"""
from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd

ADAPT_PER_DAY = 1.0     # hours of body-clock shift per day in the new zone (Song et al. 2017)
CLIP = 3.0              # hours
COMPLETED = "completed"
USE_NEXT_GAME = False   # True reads the team's NEXT game as its previous one (mutation only)

ET, CT, MT, PT = -4, -5, -6, -7
# summer UTC offset by MLB venue id (every regular-season date falls in daylight
# time; Arizona has none and so matches Pacific summer time)
TZ_BY_VENUE = {
    1: PT, 2: ET, 3: ET, 4: CT, 5: ET, 7: CT, 10: PT, 12: ET, 14: ET, 15: PT, 17: CT, 19: MT,
    22: PT, 31: ET, 32: CT, 680: PT, 2392: CT, 2394: ET, 2395: PT, 2536: ET, 2602: ET,
    2680: PT, 2681: ET, 2735: ET, 2756: ET, 2889: CT, 3289: ET, 3309: ET, 3312: CT,
    3313: ET, 3949: CT, 4169: ET, 4705: ET, 5150: 9, 5325: CT, 5340: -6, 5381: 1, 5445: CT,
}
INTERNATIONAL = {5150, 5340, 5381}     # Seoul, Mexico City, London


def team_lags(sched: pd.DataFrame) -> pd.DataFrame:
    """sched: game_id, season, date, fp_sched, venue_id, team (one row per team per
    completed game). Returns game_id, team, lag_e, usable (bool)."""
    s = sched[sched.venue_id.isin(TZ_BY_VENUE)].copy()
    s["z"] = s.venue_id.map(TZ_BY_VENUE).astype(float)
    s = s.sort_values(["team", "date", "fp_sched", "game_id"], kind="mergesort").reset_index(drop=True)
    out = np.zeros(len(s))
    team, season, date, z = s.team.to_numpy(), s.season.to_numpy(), s.date.to_numpy(), s.z.to_numpy()
    n = len(s)
    B = 0.0
    for i in range(n):
        if USE_NEXT_GAME:
            # mutation: look at the team's NEXT game instead of its last (a leak)
            j = i + 1 if i + 1 < n and team[i + 1] == team[i] and season[i + 1] == season[i] else None
            target = z[j] if j is not None else z[i]
        else:
            j = i - 1 if i > 0 and team[i - 1] == team[i] and season[i - 1] == season[i] else None
            target = z[i]
        if j is None:
            B = z[i]                                  # season opener: body clock on local time
        else:
            g = abs((date[i] - date[j]).astype("timedelta64[D]").astype(int))
            days_there = g if z[i] == z[j] else max(g - 1, 0)
            B = B + np.clip(target - B, -days_there * ADAPT_PER_DAY, days_there * ADAPT_PER_DAY)
        out[i] = max(min(z[i] - B, CLIP), 0.0)
    s["lag_e"] = out
    s.loc[s.venue_id.isin(INTERNATIONAL), "lag_e"] = 0.0
    s["usable"] = ~s.venue_id.isin(INTERNATIONAL)
    return s[["game_id", "team", "lag_e", "usable"]]


def travel_features(con: duckdb.DuckDBPyConnection, keys: str = "tg", point: str = "P2") -> dict:
    """Writes table trv_feat (game_id, bat_team, lag_e_own, lag_e_opp, trv_missing)."""
    g = con.execute(f"""
        select game_id, season, date, epoch(fp_sched) as fp_sched, venue_id, home_team, away_team
        from games
        where status = '{COMPLETED}'
           -- live use (reviewer, 3 Oct 2026): the game being predicted has no result yet, so a
           -- game later than every completed one is kept. An older game that never finished
           -- (cancelled, status null) is still not a game.
           or date > (select max(date) from games where status = '{COMPLETED}')
    """).fetchdf()
    g["date"] = pd.to_datetime(g["date"])
    unknown = sorted(set(g.venue_id) - set(TZ_BY_VENUE))
    if unknown:
        print(f"warning: venue id(s) {unknown} have no time zone in travel.py; those games are skipped "
              f"and their travel features are 0 (add them to TZ_BY_VENUE)", file=sys.stderr)
    sched = pd.concat([g.assign(team=g.home_team), g.assign(team=g.away_team)], ignore_index=True)
    lags = team_lags(sched[["game_id", "season", "date", "fp_sched", "venue_id", "team"]])
    con.register("trv_lag_df", lags)
    con.execute(f"""
        create or replace table trv_feat as
        select q.game_id, q.bat_team,
               coalesce(a.lag_e, 0.0)::double as lag_e_own,
               coalesce(b.lag_e, 0.0)::double as lag_e_opp,
               not (coalesce(a.usable, false) and coalesce(b.usable, false)) as trv_missing
        from {keys} q
        left join trv_lag_df a on a.game_id = q.game_id and a.team = q.bat_team
        left join trv_lag_df b on b.game_id = q.game_id and b.team = q.fld_team
    """)
    con.unregister("trv_lag_df")
    notes = con.execute("""
        select count(*) as rows, sum(trv_missing::int) as missing,
               avg((lag_e_opp > 0)::int) as share_lagged, avg((lag_e_opp >= 2)::int) as share_2plus,
               avg(lag_e_opp) as mean, var_pop(lag_e_opp) as var
        from trv_feat
    """).fetchdf().iloc[0].to_dict()
    notes["point"] = point
    return {k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer,)) else v))
            for k, v in notes.items()}
