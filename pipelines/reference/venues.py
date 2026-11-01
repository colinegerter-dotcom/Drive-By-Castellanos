"""
The one venue table (2027 fixes plan, fixes-2027.md section 3).

Every module that needs something about a ballpark reads it from
venues.csv, keyed by MLB venue id. Ids survive renames, so the table never
goes stale when a park changes sponsor.

Columns of venues.csv:
  venue_id           MLB Stats API venue id
  name               current name; other_names holds earlier names, "|"-separated
  lat, lon           location (weather pulls)
  utc_offset_summer  UTC offset in daylight time, hours (every regular-season
                     date falls in daylight time; Arizona and Mexico City have none)
  roof               open, retractable or dome (dome = always closed)
  bearing            home plate to center field, degrees; blank if not researched
  international      1 for sites outside the US and Canada
  seasons            seasons with games there (documentation only)
  notes              confidence and history

Rules:
  * A venue id that isn't in the table stops the run (UnknownVenue). No
    silent neutral default
  * A known venue with too little history gets a neutral park factor that is
    set on purpose and flagged (environment.py: new_park)
  * In season, a new venue (a neutral site, a temporary home) is added as one
    row, logged as a data fix with a before and after on the affected games.
    The model is not refitted
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

VENUES_CSV = Path(__file__).with_name("venues.csv")
ROOF_TYPES = {"open", "retractable", "dome"}


class UnknownVenue(ValueError):
    """A game is at a venue id that isn't in venues.csv."""


@lru_cache(maxsize=1)
def venues() -> dict[int, dict]:
    out: dict[int, dict] = {}
    with VENUES_CSV.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vid = int(r["venue_id"])
            if vid in out:
                raise ValueError(f"venues.csv: venue id {vid} appears twice")
            if r["roof"] not in ROOF_TYPES:
                raise ValueError(f"venues.csv: venue {vid} has roof {r['roof']!r}, expected one of {sorted(ROOF_TYPES)}")
            out[vid] = {
                "venue_id": vid,
                "name": r["name"],
                "other_names": [n for n in r["other_names"].split("|") if n],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "utc_offset_summer": float(r["utc_offset_summer"]),
                "roof": r["roof"],
                "bearing": float(r["bearing"]) if r["bearing"].strip() else None,
                "international": r["international"].strip() == "1",
            }
    return out


def require_known(venue_ids) -> None:
    """Raise UnknownVenue if any id is missing from venues.csv."""
    missing = sorted({int(v) for v in venue_ids if v is not None} - set(venues()))
    if missing:
        raise UnknownVenue(
            f"venue id(s) {missing} are not in pipelines/reference/venues.csv. Add a row (see the in-season "
            f"procedure in venues.py) before running; no neutral default is used")


def roofed_ids() -> set[int]:
    """Venues with any roof, retractable or fixed."""
    return {v for v, r in venues().items() if r["roof"] in ("retractable", "dome")}


def dome_ids() -> set[int]:
    """Fixed domes: the roof is always closed, so weather never matters."""
    return {v for v, r in venues().items() if r["roof"] == "dome"}


def utc_offsets() -> dict[int, float]:
    return {v: r["utc_offset_summer"] for v, r in venues().items()}


def international_ids() -> set[int]:
    return {v for v, r in venues().items() if r["international"]}


def open_air_bearings() -> dict[int, float]:
    """Home-to-center bearing for open-air parks where it was researched (wind features)."""
    return {v: r["bearing"] for v, r in venues().items() if r["roof"] == "open" and r["bearing"] is not None}


def all_names() -> dict[str, int]:
    """Every current and earlier name to its id, for sources keyed by name."""
    out: dict[str, int] = {}
    for v, r in venues().items():
        for n in [r["name"], *r["other_names"]]:
            out[n] = v
    return out
