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
    """Season-final Outs Above Average from Savant. NOT usable as an as-of feature.

    VERIFIED LIVE 22 Sep 2026 (browser, from Colin's machine, against
    baseballsavant.mlb.com -- the check the previous docstring asked for).
    Three findings, all reproducible:

    1. `type=Team` RETURNS ZERO DATA ROWS. Every variant tried
       (`type=Team` with/without startYear+endYear, `split=no`, `team=`,
       `min=q`, `playerType=Team`, lowercase `type=team`) returns HTTP 200
       with a valid 365-byte CSV header and no rows at all. The team
       grouping is simply not served by this CSV export any more. This
       alone means def_oaa_season was never going to populate, which
       matches what's in the database: the column is null everywhere.

    2. `type=Fielder` DOES work -- 256 player rows, 21,515 bytes, with a
       `display_team_name` column. Team OAA is by definition the sum of its
       fielders' OAA, so a team number can be rebuilt from this if wanted.

    3. **`startDate` / `endDate` ARE SILENTLY IGNORED.** This is the
       important one. Fetched `type=Fielder` three ways -- no date params,
       bounded 2025-03-01..2025-05-01, and bounded 2025-03-01..2025-09-28 --
       and all three responses were BYTE-IDENTICAL (21,515 bytes each,
       same leading rows). A query asking for "through May 1" hands back
       full-season numbers without any error or warning.

    So the lookahead risk flagged in the old docstring is REAL and
    CONFIRMED, not hypothetical. Any as-of call against this endpoint gets
    season-final data -- data that includes the game being predicted and
    every game after it -- while looking like it was correctly bounded.
    Backfilling five seasons on top of that would have put a silent leak
    under the entire training set, and the model would have looked better
    for it.

    Because of (3), passing `through_date` now RAISES rather than returning
    quietly wrong data. That is deliberate: this repo's standing lesson is
    that a silent partial/wrong write is worse than a crash, and an
    as-of-bounded defensive metric is exactly the kind of thing that would
    be wired back in months from now by someone who didn't read this. Fail
    loudly instead.

    If an as-of team-defense feature is wanted later, build it from the
    pitch-level Parquet already in this repo (which IS correctly
    date-bounded because we bound it ourselves), not from this endpoint.
    """
    if through_date is not None:
        raise ValueError(
            "get_team_outs_above_average(through_date=...) is not supported: Savant "
            "silently ignores startDate/endDate on this leaderboard and returns "
            "season-final numbers (verified live 22 Sep 2026 -- three date ranges, "
            "byte-identical responses). Using it as an as-of feature would inject "
            "lookahead bias into every row. See this function's docstring."
        )

    cache_key = (year, None)
    if cache_key in _oaa_cache:
        return _oaa_cache[cache_key]

    # type=Fielder, not type=Team: the Team grouping returns a header and no
    # rows (finding 1 above). Callers wanting a team number should aggregate
    # by display_team_name.
    params = {"type": "Fielder", "startYear": year, "endYear": year, "split": "no", "team": ""}
    df = _read_savant_csv_optional(
        "https://baseballsavant.mlb.com/leaderboard/outs_above_average",
        params,
        what=f"player outs above average ({year}, season-final)",
    )
    _oaa_cache[cache_key] = df
    return df


def get_park_factors(year: int) -> pd.DataFrame:
    """DEAD ENDPOINT -- kept only so nothing silently re-wires itself to it.

    VERIFIED LIVE 22 Sep 2026 (browser, from Colin's machine). This
    leaderboard cannot be scraped any more, by any of the routes tried:

    - `&csv=true` returns `content-type: text/html`, a 98KB page, not CSV.
      That is the ParserError ("Expected 1 fields ... saw 4") in the build
      log -- pandas choking on HTML.
    - The returned HTML contains NO park data at all. Searched it for
      "Coors" and for "venue_id": neither appears. It is a pure shell.
    - No `var data = [...]` blob in the page, no data-bearing array on
      `window`, and no table in the rendered DOM containing any park name.
    - The page's own JS bundle (statcast-park-factors.js) contains no data
      fetch -- only navigation URLs to /leaderboard/statcast-venue.
    - `/leaderboard/statcast-park-factors/api`, `/api/leaderboard/...`,
      and `/leaderboard/park-factors` all 404.

    park_factors.py no longer calls this. Park factors are now computed
    in-house from our own games/game_results tables -- see
    pipelines/reference/park_factors.py. That is a better source anyway:
    no external dependency that can rot silently, and we control the
    date-bounding so it cannot leak.
    """
    log.warning(
        "get_park_factors() is a confirmed-dead endpoint (verified 22 Sep 2026) and "
        "should not be used; park factors are computed in-house from our own game "
        "results. See this function's docstring and pipelines/reference/park_factors.py."
    )
    return pd.DataFrame()


def pull_statcast_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Raw Statcast pitch-level rows for a date range, via pybaseball.

    Callers should chunk large ranges (e.g. week by week) rather than
    passing a full season here -- large single pulls are a known pybaseball
    pain point (timeouts / partial results). See pipelines/pitches/pitches.py.
    """
    import pybaseball

    pybaseball.cache.enable()  # avoid re-downloading identical date ranges within a run
    return pybaseball.statcast(start_dt=start_date, end_dt=end_date, verbose=False)
