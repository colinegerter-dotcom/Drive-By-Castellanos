"""
game_conditions table -- Open-Meteo (free, no key), decided with Colin since
the schema doc left this source unspecified. Historical rows use the
archive endpoint (observed conditions); live/upcoming rows use the forecast
endpoint. is_forecast records which kind a row is, per the schema's
forecast-vs-observed requirement.

wind_effect is derived here, not fetched: Open-Meteo gives wind DIRECTION
(the compass bearing the wind is blowing FROM, meteorological convention).
Combined with park_factors.field_orientation_degrees (the home-plate-to-
center-field bearing), that tells us whether the wind is pushing fly balls
toward the fence (blowing_out), back in (blowing_in), or across
(crosswind). See _classify_wind_effect for the exact angular logic.

reference/park_orientation.csv is now filled in for every open-air park
(researched visually from Google Maps satellite imagery -- see that file's
notes column for per-park confidence). A handful of roofed parks are still
unresolved there because the roof was closed in the available imagery --
see config.ROOFED_PARKS.

Roofed parks (config.ROOFED_PARKS) are handled specially, per Colin: wind
direction and park orientation should never drive wind_effect when a roof
can enclose the field, since we don't have a per-game feed telling us
whether a retractable roof was actually open or closed. build_game_condition_row
forces wind_effect to "neutral" for every game at one of those venues,
regardless of what orientation_deg and Open-Meteo say -- see the check near
the bottom of that function. temp/wind/humidity/precip are still stored for
those venues (mild ambient-day context), just not used to derive wind_effect.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone

import requests

from pipelines.config import OPEN_METEO_ARCHIVE_URL, OPEN_METEO_FORECAST_URL, ROOFED_PARKS, ROOFED_VENUE_IDS
from pipelines.mlb_stats_client import get_venue_names_by_season, get_venues

log = logging.getLogger(__name__)

_session = requests.Session()

_HOURLY_VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m"


def venue_name_keys(name: str) -> list[str]:
    """Every name a venue might be referred to by, most specific first.

    Sponsor renames are the reason this exists. Confirmed live (18 Sep 2026):
    MLB's /venues endpoint now calls venue 22 "UNIQLO Field at Dodger
    Stadium", while the schedule endpoint still reports that game's venue as
    "Dodger Stadium" -- so a plain name-keyed lookup missed every Dodgers home
    game and silently skipped the weather for all of them ("no coordinates for
    venue 'Dodger Stadium'" in the backfill log, ~90 games a season). Keying
    on the part after " at " as well as the full name covers the whole
    "<Sponsor> Field/Park at <Real Stadium>" pattern rather than special-casing
    one park. The real fix is to key parks by venue id everywhere instead of
    name -- that's a schema change (games.venue is a name), noted in the README.
    """
    keys = [name]
    for separator in (" at ",):
        if separator in name:
            keys.append(name.split(separator, 1)[1].strip())
    return keys


def _venue_coords_by_name(seasons: list[int] | None = None) -> dict[str, tuple[float, float]]:
    """Venue name -> (lat, lon), including names the park had in `seasons`.

    `seasons` matters for any historical backfill. games.venue holds the
    name as the schedule feed reported it at the time, while the venues
    endpoint returns today's name, so a renamed park matches nothing and
    its weather is skipped silently. Passing the season being backfilled
    adds that season's names as extra keys pointing at the same
    coordinates. See mlb_stats_client.get_venue_names_by_season for the
    live evidence and the ~162 games it cost in 2021.

    Omitting `seasons` keeps the old current-names-only behaviour, which
    is correct for the daily pull (today's games use today's names).
    """
    out: dict[str, tuple[float, float]] = {}
    coords_by_id: dict[int, tuple[float, float]] = {}

    for v in get_venues():
        loc = (v.get("location") or {}).get("defaultCoordinates") or {}
        lat, lon = loc.get("latitude"), loc.get("longitude")
        name = v.get("name")
        if lat is None or lon is None:
            continue
        if v.get("id") is not None:
            coords_by_id[v["id"]] = (lat, lon)
        if not name:
            continue
        for key in venue_name_keys(name):
            # First writer wins so a real venue never gets clobbered by
            # another park's alias.
            out.setdefault(key, (lat, lon))

    for season in seasons or []:
        try:
            names = get_venue_names_by_season(season)
        except Exception:  # noqa: BLE001 -- an alias lookup must never break a backfill
            log.warning(
                "could not fetch %s venue names for historical aliases; parks renamed "
                "since then will fall back to skipping weather (see "
                "mlb_stats_client.get_venue_names_by_season)",
                season,
                exc_info=True,
            )
            continue

        added = 0
        for venue_id, name in names.items():
            latlon = coords_by_id.get(venue_id)
            if latlon is None:
                continue
            for key in venue_name_keys(name):
                if key not in out:
                    out[key] = latlon
                    added += 1
        log.info("[%s] venue name aliases added from historical names: %d", season, added)

    return out


def _classify_wind_effect(
    wind_speed_mph: float | None,
    wind_direction_from_deg: float | None,
    orientation_deg: float | None,
    calm_threshold_mph: float = 5.0,
) -> str | None:
    if wind_speed_mph is None or wind_direction_from_deg is None or orientation_deg is None:
        return None
    if wind_speed_mph < calm_threshold_mph:
        return "neutral"
    wind_toward_deg = (wind_direction_from_deg + 180) % 360
    diff = abs(wind_toward_deg - orientation_deg) % 360
    diff = min(diff, 360 - diff)  # fold to 0-180
    if diff <= 45:
        return "blowing_out"
    if diff >= 135:
        return "blowing_in"
    return "crosswind"


def _nearest_hour_index(hourly_times: list[str], target_utc_iso: str) -> int | None:
    target = datetime.fromisoformat(target_utc_iso.replace("Z", "+00:00"))
    best_idx, best_diff = None, None
    for i, t in enumerate(hourly_times):
        # Open-Meteo returns naive local-ish timestamps when timezone param
        # is set; request timezone=UTC explicitly (see below) so these are
        # directly comparable to target (also UTC).
        ts = datetime.fromisoformat(t).replace(tzinfo=dt_timezone.utc)
        diff = abs((ts - target).total_seconds())
        if best_diff is None or diff < best_diff:
            best_idx, best_diff = i, diff
    return best_idx


def is_roofed(venue_name: str | None, venue_id: int | None = None) -> bool:
    """True for a park with a roof. Checks the venue id first (survives
    sponsor renames), then falls back to the name list for callers that
    only have a name."""
    if venue_id is not None and int(venue_id) in ROOFED_VENUE_IDS:
        return True
    return venue_name in ROOFED_PARKS


def build_game_condition_row(
    game_id: int,
    venue_name: str,
    first_pitch_time_iso: str,
    game_date: str,
    is_forecast: bool,
    orientation_deg: float | None,
    coords_cache: dict[str, tuple[float, float]] | None = None,
    venue_id: int | None = None,
) -> dict | None:
    coords = coords_cache if coords_cache is not None else _venue_coords_by_name()
    latlon = coords.get(venue_name)
    if latlon is None:
        log.warning("no coordinates for venue %r (game %s), skipping weather", venue_name, game_id)
        return None
    lat, lon = latlon

    if is_forecast:
        url = OPEN_METEO_FORECAST_URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": _HOURLY_VARS,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
            "forecast_days": 10,
        }
    else:
        url = OPEN_METEO_ARCHIVE_URL
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": game_date,
            "end_date": game_date,
            "hourly": _HOURLY_VARS,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
        }

    resp = _session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        log.warning("no hourly weather data returned for game %s", game_id)
        return None

    idx = _nearest_hour_index(times, first_pitch_time_iso)
    if idx is None:
        return None

    def at(key):
        vals = hourly.get(key, [])
        return vals[idx] if idx < len(vals) else None

    wind_speed = at("wind_speed_10m")
    wind_direction = at("wind_direction_10m")
    precip = at("precipitation")

    # Roofed parks: never let wind/orientation drive wind_effect -- a closed
    # roof makes the physics moot, and we have no per-game roof-state feed to
    # tell open from closed. See config.ROOFED_PARKS and the module docstring.
    if is_roofed(venue_name, venue_id):
        wind_effect = "neutral"
    else:
        wind_effect = _classify_wind_effect(wind_speed, wind_direction, orientation_deg)

    return {
        "game_id": game_id,
        "temp_f": at("temperature_2m"),
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "humidity": at("relative_humidity_2m"),
        "precip_flag": (precip or 0) > 0,
        "wind_effect": wind_effect,
        "is_forecast": is_forecast,
    }
