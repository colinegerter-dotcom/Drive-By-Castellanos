"""
Round 2 group 3: wind (design 5.7 item 3, as built: design E12, written
1 Oct 2026 before the build).

Wind changes how far fly balls carry. What matters is the part of the wind
blowing out toward center field (helps fly balls) or in from it (hurts),
which depends on the park's direction:

  out = speed x cos(bearing the wind blows TOWARD - park bearing)

speed in mph and direction (the bearing the wind blows FROM) are the
observed game-time conditions (table `conditions`, Open-Meteo, the hour of
first pitch); the park bearing is home plate to center field, from
pipelines/reference/park_orientation.csv (copied below by venue id).

Features, the same for both team rows of a game:
  wind_out        outward part, 0 to 12 mph (capped: moderate wind out adds
                  runs, strong wind out doesn't add more, design C10)
  wind_in         inward part, 0 to 12 mph
  wind_out_wrig   wind_out at Wrigley Field only (its own pair, design 5.7)
  wind_in_wrig    wind_in at Wrigley Field only
  wind_missing    flag: roofed park, no reliable bearing, or no reading;
                  all four features are 0 then

Prediction points: P2 uses the observed game-time wind (the same assumption
P2 already makes for temperature). P1 has no fair wind input for 2021-2023
(the forecast archive has no wind before 2024), so at P1 the features are 0
and wind_missing is true (design E12: wind is judged at P2).

Game-time weather of game D itself is legitimately known for game D; nothing
from any other game is used.
"""
from __future__ import annotations

import duckdb

from .inputs import ROOFED_VENUE_IDS

CAP_MPH = 12.0
WRIGLEY = 17

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


def wind_features(con: duckdb.DuckDBPyConnection, keys: str = "tg", point: str = "P2") -> None:
    """Writes table wind_feat (game_id, bat_team, wind_out, wind_in,
    wind_out_wrig, wind_in_wrig, wind_missing)."""
    rows = ",".join(f"({v}, {b})" for v, b in BEARING.items() if v not in ROOFED_VENUE_IDS)
    con.execute(f"create or replace temp table park_bearing as select * from (values {rows}) t(venue_id, bearing)")
    if point == "P2":
        out_sql = "c.wind_speed * cos(radians(((c.wind_dir + 180) % 360) - b.bearing))"
        usable = "c.wind_speed is not null and c.wind_dir is not null and b.bearing is not null"
    else:
        out_sql = "0.0"
        usable = "false"
    con.execute(f"""
        create or replace table wind_feat as
        with w as (
            select q.game_id, q.bat_team, q.venue_id,
                   case when {usable} then {out_sql} end as out_mph
            from {keys} q
            left join conditions c on c.game_id = q.game_id
            left join park_bearing b on b.venue_id = q.venue_id
        )
        select game_id, bat_team,
               coalesce(least(greatest(out_mph, 0), {CAP_MPH}), 0.0) as wind_out,
               coalesce(least(greatest(-out_mph, 0), {CAP_MPH}), 0.0) as wind_in,
               case when venue_id = {WRIGLEY} then coalesce(least(greatest(out_mph, 0), {CAP_MPH}), 0.0) else 0.0 end as wind_out_wrig,
               case when venue_id = {WRIGLEY} then coalesce(least(greatest(-out_mph, 0), {CAP_MPH}), 0.0) else 0.0 end as wind_in_wrig,
               (out_mph is null) as wind_missing
        from w
    """)
