"""
starting_batter_form table. Same split-source pattern as
starting_pitcher_form.py: official byDateRange stats for anything that's a
standard box-score counting stat (wOBA inputs, K%, BB%), our own
mlb.pitches table for Statcast-specific metrics (exit velo, barrel rate).

wOBA weights: MLB Stats API doesn't return wOBA directly, so it's computed
here from official counting stats (AB, BB, HBP, singles, doubles, triples,
HR, SF) using FanGraphs' published linear weight formula. The weights
themselves ("the guts") are recalculated by FanGraphs every season and
drift slightly year to year -- this uses the well-established ~2023-2025
constants as a stable approximation rather than a look-up table of exact
annual weights. Good enough for relative player comparison within a season;
flag this if the model ever needs precise cross-season wOBA comparability.

NOT YET BUILT (flagged, not faked):
- vs_pitcher_hand_split: needs to know the OPPOSING starter's throwing hand
  for the specific game being scored, which this function doesn't receive
  (it's a per-game context input, not a property of the batter alone) --
  wire this up at the call site once starting_pitcher_form and
  starting_batter_form are being built together for the same game.
- days_since_trade: same gap as the pitcher table -- no transactions table
  in scope for this build.
"""
from __future__ import annotations

from datetime import date, timedelta

from pipelines.mlb_stats_client import get_player_stats_by_date_range
from pipelines.player_form.sql_helpers import batter_events_query, window_start

SEASON_START = "{season}-03-01"

# FanGraphs-style wOBA linear weights, ~2023-2025 seasons (see module docstring).
W_BB, W_HBP, W_1B, W_2B, W_3B, W_HR = 0.690, 0.722, 0.888, 1.271, 1.616, 2.101


def _woba(stat: dict) -> float | None:
    ab = stat.get("atBats")
    bb = stat.get("baseOnBalls", 0)
    ibb = stat.get("intentionalWalks", 0)
    hbp = stat.get("hitByPitch", 0)
    sf = stat.get("sacFlies", 0)
    hits = stat.get("hits", 0)
    doubles = stat.get("doubles", 0)
    triples = stat.get("triples", 0)
    hr = stat.get("homeRuns", 0)
    singles = hits - doubles - triples - hr
    unintentional_bb = bb - ibb
    denom = (ab or 0) + unintentional_bb + sf + hbp
    if not denom:
        return None
    numerator = (
        W_BB * unintentional_bb
        + W_HBP * hbp
        + W_1B * singles
        + W_2B * doubles
        + W_3B * triples
        + W_HR * hr
    )
    return round(numerator / denom, 3)


def _k_bb_pct(stat: dict) -> tuple[float | None, float | None]:
    pa = stat.get("plateAppearances")
    if not pa:
        return None, None
    k = stat.get("strikeOuts", 0)
    bb = stat.get("baseOnBalls", 0)
    return round(100 * k / pa, 1), round(100 * bb / pa, 1)


def build_starting_batter_form_row(
    conn,
    batter_id: int,
    game_id: int,
    game_date: str,
    season: int,
    debut_date: str | None,
    prefetched: dict | None = None,
) -> dict:
    """prefetched: optional {"season": stat, "last30": stat,
    "career_before_season": totals} for THIS batter, already fetched in bulk by
    the caller (see scripts/backfill.py). Supplying it replaces the three
    per-player HTTP calls this function would otherwise make.

    Why it exists: those three calls, times ~18 batters, times every game, were
    ~135,000 requests for one season and measured at 20+ minutes per 100 games
    on 18 Sep 2026 -- too slow to finish inside GitHub's 6-hour cap. Fetching a
    whole lineup's window in one request cuts that by more than an order of
    magnitude. Left optional so daily_pull.py, which only ever touches a
    handful of games, keeps working unchanged.
    """
    season_start = SEASON_START.format(season=season)
    career_start = debut_date or season_start
    last30_start = (date.fromisoformat(game_date) - timedelta(days=30)).isoformat()

    # LOOKAHEAD FIX (18 Sep 2026): these used to pass game_date itself as the
    # end of the range. MLB's byDateRange endpoint is INCLUSIVE of endDate, so
    # every one of these "form coming into the game" numbers silently included
    # the game being predicted -- a batter's 4-for-4 showed up in the wOBA the
    # model would have used to predict that same game. The SQL-sourced metrics
    # in this module were always correct (sql_helpers filters `g.date <
    # as_of_date`); it was only the official-stats calls that leaked. Ending
    # the day BEFORE the game makes both sources agree on the same cutoff.
    as_of_end = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()

    if prefetched is None:
        season_stat = get_player_stats_by_date_range(batter_id, "hitting", season_start, as_of_end)
        last30_stat = get_player_stats_by_date_range(batter_id, "hitting", last30_start, as_of_end)
        career_stat = get_player_stats_by_date_range(batter_id, "hitting", career_start, as_of_end)
        mlb_pa_count = career_stat.get("plateAppearances")
    else:
        season_stat = prefetched.get("season") or {}
        last30_stat = prefetched.get("last30") or {}
        # Career-to-date = everything before this season (a constant, fetched
        # once per backfill) + this season so far. Equivalent to the old
        # debut-date-to-yesterday window, without a call per batter per game --
        # and it can't drift into lookahead, because both halves are bounded
        # by the same as-of cutoff.
        prior_totals = prefetched.get("career_before_season") or {}
        pa_before = prior_totals.get("plateAppearances")
        pa_this_season = season_stat.get("plateAppearances")
        mlb_pa_count = (
            (pa_before or 0) + (pa_this_season or 0)
            if (pa_before is not None or pa_this_season is not None)
            else None
        )

    k_pct_season, bb_pct_season = _k_bb_pct(season_stat)
    k_pct_30d, bb_pct_30d = _k_bb_pct(last30_stat)

    # `conn` here is the pitch source (DuckDB over Parquet), not Postgres --
    # see pipelines/pitch_store.py. It deliberately presents the same
    # cursor/execute/fetchall surface, so this block is unchanged apart from
    # the rolling window's lower bound now being computed in Python rather
    # than with engine-specific interval SQL.
    with conn.cursor() as cur:
        cur.execute(
            batter_events_query(days=None),
            {"player_id": batter_id, "season": season, "as_of_date": game_date},
        )
        season_rows = cur.fetchall()
        cur.execute(
            batter_events_query(days=30),
            {
                "player_id": batter_id, "season": season, "as_of_date": game_date,
                "since_date": window_start(game_date, 30),
            },
        )
        last30_rows = cur.fetchall()

    def summarize(rows):
        # rows: (pitch_result, events, exit_velocity, launch_angle, game_id, date)
        bip = [r for r in rows if r[2] is not None]  # balls actually put in play
        exit_velos = [r[2] for r in bip]
        # Barrel: Statcast's real definition is a launch-angle/exit-velo
        # matrix (not a single cutoff); this uses the commonly-cited
        # simplified approximation (EV >= 98 mph AND 26 <= LA <= 30) rather
        # than the full matrix, which is close but not exact at the edges.
        barrels = sum(1 for r in bip if r[2] >= 98 and r[3] is not None and 26 <= r[3] <= 30)
        avg_exit_velo = round(sum(exit_velos) / len(exit_velos), 1) if exit_velos else None
        barrel_pct = round(100 * barrels / len(bip), 1) if bip else None
        last_game_date = max((r[5] for r in rows), default=None)
        return avg_exit_velo, barrel_pct, last_game_date

    avg_exit_velo_season, barrel_pct_season, last_game_date = summarize(season_rows)
    avg_exit_velo_30d, barrel_pct_30d, _ = summarize(last30_rows)

    days_since_last_game = (date.fromisoformat(game_date) - last_game_date).days if last_game_date else None

    return {
        "batter_id": batter_id,
        "game_id": game_id,
        "woba_season": _woba(season_stat),
        "woba_last_30d": _woba(last30_stat),
        "k_pct_season": k_pct_season,
        "k_pct_last_30d": k_pct_30d,
        "bb_pct_season": bb_pct_season,
        "bb_pct_last_30d": bb_pct_30d,
        "avg_exit_velo_season": avg_exit_velo_season,
        "avg_exit_velo_last_30d": avg_exit_velo_30d,
        "barrel_pct_season": barrel_pct_season,
        "barrel_pct_last_30d": barrel_pct_30d,
        "vs_pitcher_hand_split": None,  # not yet built -- see module docstring
        "days_since_last_game": days_since_last_game,
        "mlb_pa_count": mlb_pa_count,
        "days_since_trade": None,  # no transactions table in scope for this build
    }
