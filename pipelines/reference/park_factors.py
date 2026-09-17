"""
park_factors table -- three different sources stitched together:

1. park_id: MLB Stats API venue id (stable identifier even when a park's
   sponsor name changes, unlike using the name itself as the key).
2. park_factor_hr / park_factor_runs: Baseball Savant's Statcast park
   factors leaderboard (savant_client.get_park_factors).
3. field_orientation_degrees: NOT available from any free API -- this is
   static geographic data. Filled in by hand, once, in
   reference/park_orientation.csv, keyed by venue NAME (readable, but
   fragile across stadium-naming-rights changes -- see that file's header).

reference/park_orientation.csv was originally a template with every row
left as TODO -- no clean machine-readable source exists (Baseball Almanac's
AL/NL orientation pages, and everything else found, are diagrams, not
text/tables). It's now filled in via one-time visual research against
Google Maps satellite imagery (home-plate-to-center-field bearing,
estimated to roughly +/-15-20 degrees -- plenty for the 4-bucket,
45-degree-wide wind_effect classification this feeds). See that file's
notes column for per-park confidence, and for which parks are still
unresolved (roofed parks whose roof was closed in the available imagery).

Roofed parks (config.ROOFED_PARKS): field_orientation_degrees is forced to
None here regardless of what's in the CSV, even for the couple of roofed
parks where the roof happened to be open in the imagery (T-Mobile Park) or
a value was estimated anyway (American Family Field). Decided with Colin --
orientation is physically moot once a field can be enclosed, and we have no
per-game feed telling us whether a retractable roof was actually open, so
nulling it here means nothing downstream (not just game_conditions.py's
wind_effect) can silently treat a roofed park's orientation as meaningful.
"""
from __future__ import annotations

import csv
import logging
import os

from pipelines.config import ROOFED_PARKS
from pipelines.mlb_stats_client import get_venues
from pipelines.savant_client import get_park_factors

log = logging.getLogger(__name__)

_ORIENTATION_CSV = os.path.join(os.path.dirname(__file__), "park_orientation.csv")


def _load_orientation_by_name() -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    with open(_ORIENTATION_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = row["orientation_degrees_azimuth"].strip()
            out[row["venue_name"]] = float(val) if val else None
    return out


def build_park_factor_rows(year: int) -> list[dict]:
    venues = get_venues()
    orientation_by_name = _load_orientation_by_name()

    savant_df = get_park_factors(year)
    # Savant's park factor CSV column names have moved before across site
    # redesigns; this list covers what's been observed. If a pull starts
    # returning zero matches, log the actual columns (savant_client logs
    # them at DEBUG level) and add the new name here.
    name_col_candidates = ["venue_name", "name_display_club", "home_team", "team_name"]
    hr_col_candidates = ["index_hr", "hr_index", "park_factor_hr"]
    runs_col_candidates = ["index_wOBA", "index_runs", "park_factor_runs", "index_woba"]

    def pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    name_col = pick_col(savant_df, name_col_candidates)
    hr_col = pick_col(savant_df, hr_col_candidates)
    runs_col = pick_col(savant_df, runs_col_candidates)
    if name_col is None:
        log.warning(
            "could not find a venue-name column in Savant park factors CSV; "
            "got columns=%s -- park_factor_hr/runs will be null this run",
            list(savant_df.columns),
        )

    savant_by_name = {}
    if name_col:
        for _, r in savant_df.iterrows():
            savant_by_name[r[name_col]] = r

    rows = []
    unmatched_orientation = []
    for v in venues:
        # Only active MLB venues -- the /venues endpoint returns a lot of
        # spring training / minor league facilities too, which we don't want.
        if not v.get("active", True):
            continue
        name = v.get("name")
        park_id = v.get("id")
        if park_id is None or name is None:
            continue

        savant_row = savant_by_name.get(name)
        orientation = None if name in ROOFED_PARKS else orientation_by_name.get(name)
        if name not in orientation_by_name:
            unmatched_orientation.append(name)

        rows.append(
            {
                "park_id": str(park_id),
                "year": year,
                "park_factor_hr": float(savant_row[hr_col]) if savant_row is not None and hr_col else None,
                "park_factor_runs": float(savant_row[runs_col]) if savant_row is not None and runs_col else None,
                "field_orientation_degrees": orientation,
            }
        )

    if unmatched_orientation:
        log.warning(
            "%d venues have no row in park_orientation.csv (add them): %s",
            len(unmatched_orientation),
            unmatched_orientation,
        )
    return rows
