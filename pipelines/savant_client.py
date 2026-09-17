"""
Baseball Savant access: pybaseball for the well-trodden paths (raw Statcast
pitch data), plus direct CSV-export pulls for two leaderboards pybaseball
doesn't wrap at the granularity we need (team-level OAA, park factors).

Savant's leaderboard pages all support a `&csv=true` query param that
returns the same data backing the page as a plain CSV -- no auth, no key.
The exact column names have shifted between Savant redesigns before, so
`_read_savant_csv` logs the columns it got back on first use; if a pull
starts returning empty/wrong data, check that first.
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({"User-Agent": "drive-from-castellanos/0.1 (personal research project)"})


def _read_savant_csv(url: str, params: dict) -> pd.DataFrame:
    resp = _session.get(url, params={**params, "csv": "true"}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    log.debug("savant csv %s columns: %s", url, list(df.columns))
    return df


def get_team_outs_above_average(year: int, through_date: str | None = None) -> pd.DataFrame:
    """Team-level Outs Above Average for a season, optionally as-of a date.

    pybaseball's statcast_outs_above_average() is player-level only; this
    hits the same leaderboard Savant serves at the "Team" grouping directly.

    NEEDS LIVE VALIDATION: `through_date` is intended to bound the
    leaderboard to games through that date (so team_form.py can compute a
    no-lookahead, as-of-yesterday defense number for historical backfill
    rather than reusing the full season's final number on every game).
    I could not confirm live whether Savant's leaderboard endpoint actually
    honors a date-bounded query in this "Team" mode versus only whole-year
    aggregates -- this repo's cloud environment can't reach baseballsavant.mlb.com
    to check. If through_date turns out to be ignored, def_oaa_season will
    silently be season-end data reused for every game in that season, which
    IS a lookahead-bias violation -- verify this before trusting that column
    for anything backtest-critical, and see team_form.py's docstring.
    """
    params = {"type": "Team", "startYear": year, "endYear": year, "split": "no", "team": ""}
    if through_date:
        params["startDate"] = f"{year}-03-01"
        params["endDate"] = through_date
    return _read_savant_csv("https://baseballsavant.mlb.com/leaderboard/outs_above_average", params)


def get_park_factors(year: int) -> pd.DataFrame:
    """Savant's Statcast park factors leaderboard (HR factor, runs factor) for a season."""
    return _read_savant_csv(
        "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors",
        {"type": "year", "year": year, "batSide": "", "stat": "index_wOBA", "condition": "All", "rolling": "no"},
    )


def pull_statcast_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Raw Statcast pitch-level rows for a date range, via pybaseball.

    Callers should chunk large ranges (e.g. week by week) rather than
    passing a full season here -- large single pulls are a known pybaseball
    pain point (timeouts / partial results). See pipelines/pitches/pitches.py.
    """
    import pybaseball

    pybaseball.cache.enable()  # avoid re-downloading identical date ranges within a run
    return pybaseball.statcast(start_dt=start_date, end_dt=end_date, verbose=False)
