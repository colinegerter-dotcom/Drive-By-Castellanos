"""
starting_pitcher_form table.

Splits its fields across two different sources deliberately:

- era_*, fip inputs (BB/HBP/K/HR/IP), k_pct/bb_pct: pulled from the MLB
  Stats API's official byDateRange stats (mlb_stats_client), NOT
  reconstructed from raw Statcast pitch events. Reason: ERA depends on
  "earned" vs unearned runs, which is an official scorer's judgment call
  (errors, etc) that simply isn't present in Statcast pitch-by-pitch data.
  Trying to derive ERA from mlb.pitches would silently produce a different
  (wrong) number from the one everyone else means by "ERA".
- avg_velo, velo_trend, ground_ball_pct, whiff_pct, pitch_count_last_start:
  these ARE Statcast-specific and come from our own mlb.pitches table,
  which is the only place that has pitch-level release speed / batted-ball
  type / swing-and-miss detail.

Every date-range call uses [season-start, as_of_date) or [debut_date,
as_of_date) -- never "current"/"career" stats as MLB's API would return
them live, which would leak future games during backfill.

NOT YET BUILT (flagged, not faked):
- fip / fip_season, xfip / xfip_season: FIP needs a per-season constant
  that can be computed correctly from our own league-wide byDateRange
  totals (FIP_constant = lgERA - ((13*lgHR + 3*(lgBB+lgHBP) - 2*lgK) / lgIP)),
  which requires a league-wide aggregation step not built in this pass.
  The per-pitcher HR/BB/HBP/K/IP inputs below ARE already being pulled, so
  wiring this up later is straightforward -- it just isn't done yet.
- spin_rate_percentile: needs a league-wide percentile ranking over
  mlb.pitches, not just this pitcher's rows. Same story, not yet built.
- days_since_trade: no source table for transactions is in scope for this
  build (schema doc doesn't define one) -- left null pending a decision on
  whether to add one.
"""
from __future__ import annotations

import statistics

from pipelines.mlb_stats_client import get_player_stats_by_date_range
from pipelines.player_form.sql_helpers import pitcher_events_query

SEASON_START = "{season}-03-01"  # generous lower bound covering earliest spring/opener dates


def _k_bb_pct(stat: dict) -> tuple[float | None, float | None]:
    batters_faced = stat.get("battersFaced")
    if not batters_faced:
        return None, None
    k = stat.get("strikeOuts", 0)
    bb = stat.get("baseOnBalls", 0)
    return round(100 * k / batters_faced, 1), round(100 * bb / batters_faced, 1)


def build_starting_pitcher_form_row(
    conn, pitcher_id: int, game_id: int, game_date: str, season: int, debut_date: str | None
) -> dict:
    season_start = SEASON_START.format(season=season)
    career_start = debut_date or season_start

    from datetime import date, timedelta

    # LOOKAHEAD FIX (18 Sep 2026): same bug as starting_batter_form.py -- MLB's
    # byDateRange endpoint is INCLUSIVE of endDate, so passing game_date meant
    # the start being scored was counted in the "form coming into the start"
    # numbers (a 7-inning shutout improved the ERA/K% the model would have used
    # to predict that very start). Ends the day before now, matching what
    # sql_helpers already enforced on the Statcast side.
    as_of_end = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()

    season_stat = get_player_stats_by_date_range(pitcher_id, "pitching", season_start, as_of_end)
    # last-30-days window
    last30_start = (date.fromisoformat(game_date) - timedelta(days=30)).isoformat()
    last30_stat = get_player_stats_by_date_range(pitcher_id, "pitching", last30_start, as_of_end)
    career_stat = get_player_stats_by_date_range(pitcher_id, "pitching", career_start, as_of_end)

    k_pct_season, bb_pct_season = _k_bb_pct(season_stat)
    k_pct_30d, bb_pct_30d = _k_bb_pct(last30_stat)

    # Statcast-specific fields from our own pitches table. Season-to-date
    # rows cover everything we need here (season ground_ball_pct/whiff_pct,
    # plus the most recent start's velo/pitch-count/days-rest) -- no
    # separate last-30d pull needed on this side, unlike the official-stats
    # calls above where k_pct_last_30d/bb_pct_last_30d are genuinely
    # different windows.
    with conn.cursor() as cur:
        cur.execute(
            pitcher_events_query(days=None),
            {"player_id": pitcher_id, "season": season, "as_of_date": game_date},
        )
        season_rows = cur.fetchall()

    def summarize(rows):
        # rows: (pitch_result, events, bb_type, release_speed, game_id, date)
        speeds = [r[3] for r in rows if r[3] is not None]
        ground_balls = sum(1 for r in rows if r[2] == "ground_ball")
        batted_balls = sum(1 for r in rows if r[2] is not None)
        swings_results = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
        whiffs = sum(1 for r in rows if r[0] in swings_results)
        # Approximate swing count: whiffs + fouls + balls_in_play (description
        # doesn't cleanly separate "swing" from "take" beyond these categories).
        swing_like = {"foul", "hit_into_play", "foul_tip"} | swings_results
        swings = sum(1 for r in rows if r[0] in swing_like)
        avg_velo = round(statistics.mean(speeds), 1) if speeds else None
        ground_ball_pct = round(100 * ground_balls / batted_balls, 1) if batted_balls else None
        whiff_pct = round(100 * whiffs / swings, 1) if swings else None
        last_game_id = rows[0][4] if rows else None
        last_game_date = rows[0][5] if rows else None
        return avg_velo, ground_ball_pct, whiff_pct, last_game_id, last_game_date

    avg_velo_season, ground_ball_pct, whiff_pct, _, _ = summarize(season_rows)
    # Most recent game this pitcher threw in, found from the season rows
    # (sorted desc by date isn't guaranteed by the query, so take max explicitly).
    last_game_id, last_game_date, avg_velo_last_start = None, None, None
    if season_rows:
        last_game_date = max(r[5] for r in season_rows)
        last_game_id = next(r[4] for r in season_rows if r[5] == last_game_date)
        last_start_speeds = [r[3] for r in season_rows if r[4] == last_game_id and r[3] is not None]
        if last_start_speeds:
            avg_velo_last_start = round(statistics.mean(last_start_speeds), 1)

    days_rest = None
    pitch_count_last_start = None
    velo_trend = None
    if last_game_id is not None:
        days_rest = (date.fromisoformat(game_date) - last_game_date).days
        pitch_count_last_start = sum(1 for r in season_rows if r[4] == last_game_id)
        if avg_velo_last_start is not None and avg_velo_season is not None:
            velo_trend = round(avg_velo_last_start - avg_velo_season, 1)

    return {
        "pitcher_id": pitcher_id,
        "game_id": game_id,
        "era_last_30d": last30_stat.get("era"),
        "era_season": season_stat.get("era"),
        "fip": None,  # not yet built -- see module docstring
        "fip_season": None,
        "xfip": None,
        "xfip_season": None,
        "k_pct_last_30d": k_pct_30d,
        "k_pct_season": k_pct_season,
        "bb_pct_last_30d": bb_pct_30d,
        "bb_pct_season": bb_pct_season,
        "avg_velo_last_start": avg_velo_last_start,
        "avg_velo_season": avg_velo_season,
        "velo_trend": velo_trend,
        "days_rest": days_rest,
        "pitch_count_last_start": pitch_count_last_start,
        "spin_rate_percentile": None,  # not yet built -- see module docstring
        "ground_ball_pct": ground_ball_pct,
        "whiff_pct": whiff_pct,
        "mlb_ip_count": career_stat.get("inningsPitched"),
        "days_since_trade": None,  # no transactions table in scope for this build
    }
