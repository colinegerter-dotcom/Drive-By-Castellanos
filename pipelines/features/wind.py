"""
Round 2 group 3: wind (design 5.7 item 3, as built: design E12, written
1 Oct 2026 before the build; second look, park-centred: design E13, written
2 Oct 2026 before the build).

Wind changes how far fly balls carry. What matters is the part of the wind
blowing out toward center field (helps fly balls) or in from it (hurts),
which depends on the park's direction:

  out = speed x cos(bearing the wind blows TOWARD - park bearing)

speed in mph and direction (the bearing the wind blows FROM) are the
observed game-time conditions (table `conditions`, Open-Meteo, the hour of
first pitch); the park bearing is home plate to center field, from
pipelines/reference/park_orientation.csv (copied below by venue id).

E12 features, the same for both team rows of a game:
  wind_out        outward part, 0 to 12 mph (capped: moderate wind out adds
                  runs, strong wind out doesn't add more, design C10)
  wind_in         inward part, 0 to 12 mph
  wind_out_wrig   wind_out at Wrigley Field only (its own pair, design 5.7)
  wind_in_wrig    wind_in at Wrigley Field only
  wind_missing    flag: roofed park, no reliable bearing, or no reading;
                  all four features are 0 then

E13 features (park-centred). The runs park factor already carries each
park's usual wind (Oracle Park sits at the 12 mph cap in most games), so
only wind that differs from a park's normal can add anything. Each E12
feature minus that park's average of the same feature over its
regular-season games with a reading in the 3 prior seasons, the same window
as the runs park factor (2018-2020 for 2021 games, 2019-2021 for 2022, ...):
  wind_out_c, wind_in_c, wind_out_wrig_c, wind_in_wrig_c
  wind_c_missing  flag: no reading, roofed or no bearing, or fewer than
                  MIN_PRIOR_GAMES prior games at the park; all four are 0
Positive wind_out_c means more wind blowing out than usual at that park.
2018-2020 readings come from table `wind_hist` (wind_hist.csv), so 2021
games have a full window. Only prior seasons are used, so nothing from the
season being predicted (or later) enters a game's features.

Prediction points: P2 uses the observed game-time wind (the same assumption
P2 already makes for temperature). P1 has no fair wind input for 2021-2023
(the forecast archive has no wind before 2024), so at P1 every wind feature
is 0 and both missing flags are true (designs E12 and E13: wind is judged
at P2).

Game-time weather of game D itself is legitimately known for game D; the
park averages use only earlier seasons.
"""
from __future__ import annotations

import duckdb

from .inputs import ROOFED_VENUE_IDS

CAP_MPH = 12.0
WRIGLEY = 17
LOOKBACK = 3          # prior seasons in a park's average (E13)
LAG = 1               # most recent season used = target season - LAG. Must be
                      # 1; the leak test's mutation check sets 0 (same season)
MIN_PRIOR_GAMES = 40  # games with a reading needed for a park average (E13)

# Home plate to center field bearing (degrees), from park_orientation.csv,
# by MLB venue id, for open-air parks used 2021-2026. Left out on purpose:
# roofed parks (wind is forced to 0), Oakland Coliseum (venue 10; not in the
# orientation file), and spring, temporary and one-off neutral sites.
BEARING = {
    1: 30,      # Angel Stadium
    2: 358,     # Oriole Park at Camden Yards
    3: 42,      # Fenway Park
    4: 186,     # Guaranteed Rate Field / Rate Field (flagged unverified)
    5: 2,       # Progressive Field
    7: 13,      # Kauffman Stadium
    17: 30,     # Wrigley Field
    19: 342,    # Coors Field
    22: 32,     # Dodger Stadium
    31: 117,    # PNC Park
    2394: 141,  # Comerica Park (flagged unverified)
    2395: 60,   # Oracle Park
    2602: 10,   # Great American Ball Park
    2680: 350,  # Petco Park
    2681: 340,  # Citizens Bank Park
    2889: 37,   # Busch Stadium
    3289: 337,  # Citi Field
    3309: 2,    # Nationals Park
    3312: 344,  # Target Field
    3313: 33,   # Yankee Stadium
    4705: 152,  # Truist Park
}


def _capped(out: str) -> str:
    """The four E12 columns from a signed outward-wind expression (mph)."""
    o = f"coalesce(least(greatest({out}, 0), {CAP_MPH}), 0.0)"
    i = f"coalesce(least(greatest(-({out}), 0), {CAP_MPH}), 0.0)"
    return (f"{o} as wind_out, {i} as wind_in, "
            f"case when venue_id = {WRIGLEY} then {o} else 0.0 end as wind_out_wrig, "
            f"case when venue_id = {WRIGLEY} then {i} else 0.0 end as wind_in_wrig")


def park_wind_means(con: duckdb.DuckDBPyConnection, keys: str = "tg") -> None:
    """Writes table wind_park_mean (venue_id, season, n, m_out, m_in,
    m_out_wrig, m_in_wrig): each park's average of the four E12 features
    over regular-season games with a reading in seasons
    season-LAG-LOOKBACK+1 .. season-LAG, for every season in `keys`."""
    has_hist = con.execute(
        "select count(*) from information_schema.tables where table_name = 'wind_hist'").fetchone()[0] > 0
    hist = ("union all select game_id, season, venue_id, game_type, wind_speed, wind_dir from wind_hist "
            "where season < (select min(season) from games)") if has_hist else ""
    con.execute(f"""
        create or replace table wind_park_mean as
        with allg as (
            select g.game_id, g.season, g.venue_id, g.game_type, c.wind_speed, c.wind_dir
            from games g left join conditions c using (game_id)
            {hist}
        ),
        pg as (
            select a.season, a.venue_id,
                   a.wind_speed * cos(radians(((a.wind_dir + 180) % 360) - b.bearing)) as out_mph
            from allg a join park_bearing b on b.venue_id = a.venue_id
            where a.game_type = 'R' and a.wind_speed is not null and a.wind_dir is not null
        ),
        f as (select season, venue_id, {_capped('out_mph')} from pg),
        targets as (select distinct season as target from {keys})
        select t.target as season, f.venue_id, count(*) as n,
               avg(f.wind_out) as m_out, avg(f.wind_in) as m_in,
               avg(f.wind_out_wrig) as m_out_wrig, avg(f.wind_in_wrig) as m_in_wrig
        from targets t
        join f on f.season between t.target - {LAG} - {LOOKBACK} + 1 and t.target - {LAG}
        group by 1, 2
    """)


def wind_features(con: duckdb.DuckDBPyConnection, keys: str = "tg", point: str = "P2") -> None:
    """Writes table wind_feat (game_id, bat_team, the E12 columns and the E13
    park-centred columns)."""
    rows = ",".join(f"({v}, {b})" for v, b in BEARING.items() if v not in ROOFED_VENUE_IDS)
    con.execute(f"create or replace temp table park_bearing as select * from (values {rows}) t(venue_id, bearing)")
    park_wind_means(con, keys)
    if point == "P2":
        out_sql = "c.wind_speed * cos(radians(((c.wind_dir + 180) % 360) - b.bearing))"
        usable = "c.wind_speed is not null and c.wind_dir is not null and b.bearing is not null"
    else:
        out_sql = "0.0"
        usable = "false"
    con.execute(f"""
        create or replace table wind_feat as
        with w as (
            select q.game_id, q.bat_team, q.venue_id, q.season,
                   case when {usable} then {out_sql} end as out_mph
            from {keys} q
            left join conditions c on c.game_id = q.game_id
            left join park_bearing b on b.venue_id = q.venue_id
        ),
        f as (select game_id, bat_team, venue_id, season, out_mph, {_capped('out_mph')} from w),
        j as (
            select f.*, (f.out_mph is not null and coalesce(m.n, 0) >= {MIN_PRIOR_GAMES}) as ok_c,
                   m.m_out, m.m_in, m.m_out_wrig, m.m_in_wrig
            from f left join wind_park_mean m on m.venue_id = f.venue_id and m.season = f.season
        )
        select game_id, bat_team,
               wind_out, wind_in, wind_out_wrig, wind_in_wrig,
               (out_mph is null) as wind_missing,
               case when ok_c then wind_out - m_out else 0.0 end as wind_out_c,
               case when ok_c then wind_in - m_in else 0.0 end as wind_in_c,
               case when ok_c then wind_out_wrig - m_out_wrig else 0.0 end as wind_out_wrig_c,
               case when ok_c then wind_in_wrig - m_in_wrig else 0.0 end as wind_in_wrig_c,
               (not ok_c) as wind_c_missing
        from j
    """)
