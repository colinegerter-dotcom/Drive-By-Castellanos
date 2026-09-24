"""
Shared constants for the whole pipeline.

Why this file exists: several pipeline modules need the same handful of
decisions (which timezone to render in, which seasons count as "in scope",
which TV networks count as "national"). Keeping them here means a change
only has to happen in one place, and every ingestion script agrees on the
same rules.
"""
from __future__ import annotations

import os
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------
# Every timestamp we store is a Postgres `timestamptz`, which is always stored
# as an absolute instant (UTC under the hood). That already solves the "DST
# fall-back ambiguity" problem the schema doc calls out -- a timestamptz can
# never be ambiguous, because it isn't a wall-clock time, it's a point on the
# timeline. CENTRAL is only used when we need to *render* a value in Central
# time (e.g. turning "7:10 PM Central on 2026-09-17" into an absolute instant
# before we store it), never to store it.
CENTRAL = ZoneInfo("America/Chicago")

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
# Historical backfill depth, per Colin: last 4-5 completed seasons.
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024, 2025]
CURRENT_SEASON = 2026

# Generous month-day bounds for "a game could plausibly be happening today" --
# covers the earliest spring-training-adjacent date through the latest
# realistic World Series date. Shared so there's one definition instead of
# two that can drift apart: scripts/backfill.py's SEASON_DATE_RANGE builds
# from these, and daily_pull.py uses them directly to skip the off-season
# (23 Sep 2026) rather than running its API calls against an empty schedule
# 365 days a year.
SEASON_START_MD = "03-01"
SEASON_END_MD = "11-15"

# ---------------------------------------------------------------------------
# National TV networks
# ---------------------------------------------------------------------------
# MLB Stats API's schedule endpoint (hydrate=broadcasts) returns every
# broadcaster for a game, including regional sports networks (Bally Sports,
# NBC Sports Chicago, etc). There's no single field that means "this game is
# nationally televised" -- we have to classify it ourselves from the network
# name. This list is the classification. It will need occasional maintenance
# as networks rebrand (e.g. Bally -> FanDuel Sports Network happened in 2025)
# or MLB signs new national deals -- that's expected, not a bug.
NATIONAL_TV_NETWORKS = {
    "FOX", "FS1", "FS2",
    "ESPN", "ESPN2", "ESPN+",
    "TBS", "TNT",
    "Apple TV+", "Apple TV",
    "Peacock",
    "MLB Network",
    "ABC",
    "Roku",
}

# ---------------------------------------------------------------------------
# Roofed parks
# ---------------------------------------------------------------------------
# Venues with a roof (retractable or fixed). Decided with Colin: wind
# direction and park orientation should never drive wind_effect for these --
# a closed roof makes the physics moot, and we have no per-game feed telling
# us whether a retractable roof was actually open or closed on a given day.
# Rather than guess, game_conditions.py forces wind_effect to "neutral" for
# every game at one of these venues, full stop, regardless of what Open-Meteo
# reports for that day. temp_f/humidity/precip_flag are still stored (mild
# ambient-day context), but treat them as weak signal here too -- they
# reflect outside conditions, not the climate-controlled conditions batters
# and pitchers actually played in.
#
# This is also why several of these venues are the ones with an unresolved
# field_orientation_degrees in reference/park_orientation.csv: a closed roof
# in the satellite imagery hides the field, so orientation research and
# "has a roof" tend to go hand in hand -- but they're not the same list
# (T-Mobile Park's roof happened to be open in the imagery, so its
# orientation IS resolved, and it's still in this set).
ROOFED_PARKS = {
    # 24 Sep 2026: historical names added. The Astros' park was "Minute Maid
    # Park" through 2024, so 2021-2024 games there never matched this set.
    # No stored data was affected (its orientation is unresolved, so wind
    # came out neutral anyway), but anything new keying on this set would
    # have missed it. Prefer ROOFED_VENUE_IDS below, which can't go stale.
    "Minute Maid Park", "Miller Park", "Marlins Park", "Safeco Field",
    "American Family Field",  # Brewers -- retractable
    "Chase Field",  # Diamondbacks -- retractable
    "Daikin Park",  # Astros -- retractable
    "Globe Life Field",  # Rangers -- retractable
    "loanDepot park",  # Marlins -- retractable
    "Rogers Centre",  # Blue Jays -- retractable
    "T-Mobile Park",  # Mariners -- retractable
    "Tropicana Field",  # Rays -- fixed dome
}

# Same parks by MLB venue id (verified against the schedule feed 24 Sep 2026:
# each id carried both its old and new name across 2018-2026). Ids survive
# renames, so this is the set to check whenever a venue id is available.
ROOFED_VENUE_IDS = {
    32,    # American Family Field / Miller Park -- Brewers
    15,    # Chase Field -- Diamondbacks
    2392,  # Daikin Park / Minute Maid Park -- Astros
    5325,  # Globe Life Field -- Rangers
    4169,  # loanDepot park / Marlins Park -- Marlins
    14,    # Rogers Centre -- Blue Jays
    680,   # T-Mobile Park / Safeco Field -- Mariners
    12,    # Tropicana Field -- Rays (fixed dome)
}

# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------
def db_dsn() -> str:
    """Build a psycopg2 connection string from environment variables.

    Reads from the process environment (populated by a .env file via
    python-dotenv in scripts that need it) rather than hardcoding
    credentials anywhere in this repo.
    """
    host = os.environ["SUPABASE_DB_HOST"]
    port = os.environ.get("SUPABASE_DB_PORT", "5432")
    dbname = os.environ.get("SUPABASE_DB_NAME", "postgres")
    user = os.environ.get("SUPABASE_DB_USER", "postgres")
    password = os.environ["SUPABASE_DB_PASSWORD"]
    # sslmode=require: Supabase requires TLS; this isn't optional hardening,
    # connections without it are rejected.
    return (
        f"host={host} port={port} dbname={dbname} "
        f"user={user} password={password} sslmode=require"
    )


# MLB Stats API base URL. No API key required -- it's a public, unauthenticated API.
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
MLB_STATS_API_BASE_V1_1 = "https://statsapi.mlb.com/api/v1.1"

# Open-Meteo: free, no key required. Historical archive + forecast, both.
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
