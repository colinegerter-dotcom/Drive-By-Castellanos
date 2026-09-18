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

# Consecutive-failure counts per URL, for the circuit breaker in
# _read_savant_csv_optional.
_failure_counts: dict[str, int] = {}
_MAX_FAILURES_PER_URL = 3

# get_team_outs_above_average is called once per team per game (~5,000 times
# for one season) but only ever varies by as-of DATE -- roughly 190 distinct
# values in a season. Caching by (year, through_date) turns those 5,000
# network calls into ~190.
_oaa_cache: dict[tuple[int, str | None], pd.DataFrame] = {}


def clear_caches() -> None:
    """Reset the OAA cache and the circuit breaker. Call between seasons."""
    _oaa_cache.clear()
    _failure_counts.clear()


def _read_savant_csv(url: str, params: dict) -> pd.DataFrame:
    resp = _session.get(url, params={**params, "csv": "true"}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    log.debug("savant csv %s columns: %s", url, list(df.columns))
    return df


def _read_savant_csv_optional(url: str, params: dict, what: str) -> pd.DataFrame:
    """Same as _read_savant_csv, but never raises.

    Confirmed live (17 Sep 2026, first real backfill run): the
    statcast-park-factors leaderboard no longer honors `csv=true` the way
    the other Savant leaderboards below still do -- it now always returns
    the full interactive HTML page, which breaks pandas' CSV parser
    (`Expected 1 fields ... saw 4`, from HTML lines sneaking into what
    pandas expects to be comma-separated data). Rather than crash an
    entire season's backfill over one enrichment source, log it clearly
    and hand back an empty frame. build_park_factor_rows() already treats
    "no matching venue-name column" as "leave park_factor_hr/runs null for
    this park" -- see its warning -- so this degrades exactly the same way
    a genuinely-missing row would, instead of stopping the run.
    """
    # Circuit breaker: once an endpoint has failed this many times in a run,
    # stop calling it at all. Added after a live run burned a 60-second read
    # timeout against baseballsavant.mlb.com -- with a caller that runs once
    # per team per game, that is thousands of 60-second stalls waiting on a
    # service that plainly isn't answering.
    if _failure_counts.get(url, 0) >= _MAX_FAILURES_PER_URL:
        return pd.DataFrame()

    try:
        df = _read_savant_csv(url, params)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure here should degrade, not crash the caller
        _failure_counts[url] = _failure_counts.get(url, 0) + 1
        gave_up = _failure_counts[url] >= _MAX_FAILURES_PER_URL
        log.warning(
            "could not fetch %s from Savant (%s: %s) -- leaving it null this run%s; "
            "see _read_savant_csv_optional's docstring if this URL needs re-checking",
            what,
            type(exc).__name__,
            exc,
            " (giving up on this endpoint for the rest of the run)" if gave_up else "",
        )
        return pd.DataFrame()

    _failure_counts.pop(url, None)
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

    PERFORMANCE: team_form.py calls this once per team per game, so a season
    backfill asks for the same ~190 distinct as-of dates about 5,000 times.
    The result is cached by (year, through_date), and failures degrade to an
    empty frame rather than raising (team_form leaves def_oaa_season null and
    moves on) -- which also means a Savant outage no longer produces one
    60-second stall and one full traceback per team per game.
    """
    cache_key = (year, through_date)
    if cache_key in _oaa_cache:
        return _oaa_cache[cache_key]

    params = {"type": "Team", "startYear": year, "endYear": year, "split": "no", "team": ""}
    if through_date:
        params["startDate"] = f"{year}-03-01"
        params["endDate"] = through_date
    df = _read_savant_csv_optional(
        "https://baseballsavant.mlb.com/leaderboard/outs_above_average",
        params,
        what=f"team outs above average ({year} through {through_date or 'season end'})",
    )
    _oaa_cache[cache_key] = df
    return df


def get_park_factors(year: int) -> pd.DataFrame:
    """Savant's Statcast park factors leaderboard (HR factor, runs factor) for a season.

    Uses _read_savant_csv_optional, not _read_savant_csv -- see that
    function's docstring for why (this specific leaderboard page has been
    confirmed, live, to no longer return CSV via `csv=true`).
    """
    return _read_savant_csv_optional(
        "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors",
        {"type": "year", "year": year, "batSide": "", "stat": "index_wOBA", "condition": "All", "rolling": "no"},
        what="park factors",
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
