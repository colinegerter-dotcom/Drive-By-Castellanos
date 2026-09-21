"""
umpire_stats table -- computed entirely from mlb.pitches + mlb.games
(games.umpire_id, filled in by game_results.py from box score officials).

Strike-zone edge approximation: a pitch is judged "in the rulebook zone"
when sz_bot <= plate_z <= sz_top (the batter's actual measured zone top/
bottom for that at-bat) and -0.83 <= plate_x <= 0.83 (home plate is 17in
wide = 1.417ft = +/-0.708ft from center; +/-0.83ft is the commonly-used
Statcast approximation that accounts for the ball's own radius touching
the edge of the zone). This is the standard public approximation used in
umpire-scorecard-style analyses, not an exact rulebook-radius calculation.

zone_favor_score: net edge in the PITCHER's favor, in percentage points --
(extra called strikes on pitches outside the zone) minus (missed strikes,
i.e. in-zone pitches called balls), as a share of all called pitches.
Positive = umpire's zone runs pitcher-favorable relative to a literal
rulebook zone; negative = hitter-favorable. This is this umpire's own
absolute rate, not relative to league average -- unlike k_rate_boost /
bb_rate_boost below, which ARE relative to league average for the same
season (since a raw K-rate says more about the pitchers/batters in that
ump's games than about the umpire).

Only regular season, completed games count toward games_umpired -- an ump
who worked one inning of a suspended game before it was resumed under a
different plate umpire shouldn't be double-counted; this uses whichever
umpire_id games.py/game_results.py recorded as the plate umpire of record.
"""
from __future__ import annotations

from pipelines.player_form.sql_helpers import umpire_pitches_query


def _in_zone(plate_x, plate_z, sz_top, sz_bot) -> bool | None:
    if None in (plate_x, plate_z, sz_top, sz_bot):
        return None
    return (sz_bot <= plate_z <= sz_top) and (-0.83 <= plate_x <= 0.83)


def build_umpire_stats_row(conn, umpire_id: int, season: int, as_of_date: str) -> dict | None:
    # Still returns raw rows, deliberately: _in_zone() applies a strike-zone
    # geometry test per pitch that isn't worth expressing in SQL, and this
    # query is bounded by one umpire's ~30 games rather than the whole league.
    query = umpire_pitches_query()
    with conn.cursor() as cur:
        cur.execute(query, {"umpire_id": umpire_id, "season": season, "as_of_date": as_of_date})
        rows = cur.fetchall()

    if not rows:
        return None

    games_umpired = len({r[5] for r in rows})

    called_total = 0
    correct = 0
    strikes_on_outofzone = 0
    balls_on_inzone = 0
    outofzone_total = 0
    inzone_total = 0

    for pitch_result, plate_x, plate_z, sz_top, sz_bot, _ in rows:
        if pitch_result not in ("ball", "called_strike"):
            continue  # only umpire judgment calls, not swings
        in_zone = _in_zone(plate_x, plate_z, sz_top, sz_bot)
        if in_zone is None:
            continue
        called_total += 1
        called_strike = pitch_result == "called_strike"
        if in_zone:
            inzone_total += 1
            if called_strike:
                correct += 1
            else:
                balls_on_inzone += 1
        else:
            outofzone_total += 1
            if called_strike:
                strikes_on_outofzone += 1
            else:
                correct += 1

    ball_strike_accuracy_pct = round(100 * correct / called_total, 1) if called_total else None

    zone_favor_score = None
    if called_total:
        extra_strikes_pct = 100 * strikes_on_outofzone / called_total
        missed_strikes_pct = 100 * balls_on_inzone / called_total
        zone_favor_score = round(extra_strikes_pct - missed_strikes_pct, 2)

    return {
        "umpire_id": umpire_id,
        "season": season,
        "games_umpired": games_umpired,
        "ball_strike_accuracy_pct": ball_strike_accuracy_pct,
        "zone_favor_score": zone_favor_score,
        # k_rate_boost / bb_rate_boost filled in by the caller: combine
        # umpire_k_bb_rate() and league_k_bb_rate() (both below) and take
        # the difference -- this function only has this one umpire's rows
        # in scope, not league-wide context.
        "k_rate_boost": None,
        "bb_rate_boost": None,
    }


def umpire_k_bb_rate(conn, umpire_id: int, season: int, as_of_date: str) -> tuple[float | None, float | None]:
    """This umpire's own K%/BB% (share of plate appearances they officiated
    that ended in a strikeout/walk), for comparison against league_k_bb_rate.
    """
    # Counted in SQL rather than fetched row-by-row (21 Sep 2026). The old
    # version pulled every plate-appearance-ending event into Python just to
    # run three len()/sum() calls over it. Same arithmetic, but the engine
    # returns one row instead of tens of thousands.
    query = """
        select
            count(*) as total,
            count(*) filter (where p.events = 'strikeout') as ks,
            count(*) filter (where p.events = 'walk') as bbs
        from pitches p
        join games g on g.game_id = p.game_id
        where g.umpire_id = $umpire_id
          and g.season = $season
          and g.date < CAST($as_of_date AS DATE)
          and p.events is not null
    """
    with conn.cursor() as cur:
        cur.execute(query, {"umpire_id": umpire_id, "season": season, "as_of_date": as_of_date})
        total, ks, bbs = cur.fetchone()
    if not total:
        return None, None
    return round(100 * ks / total, 1), round(100 * bbs / total, 1)


def league_k_bb_rate(conn, season: int, as_of_date: str) -> tuple[float | None, float | None]:
    """League-wide K%/BB% over the same season/date-range, as the baseline
    k_rate_boost / bb_rate_boost are measured against.
    """
    # This is the single heaviest query in the whole form build: it covers
    # EVERY pitch in the season to date and runs once per game (~2,477 times).
    # Fetching the raw events to Python, as this used to, meant moving
    # hundreds of millions of rows across a season. Counting in SQL turns
    # each call into one returned row.
    query = """
        select
            count(*) as total,
            count(*) filter (where p.events = 'strikeout') as ks,
            count(*) filter (where p.events = 'walk') as bbs
        from pitches p
        join games g on g.game_id = p.game_id
        where g.season = $season
          and g.date < CAST($as_of_date AS DATE)
          and p.events is not null
    """
    with conn.cursor() as cur:
        cur.execute(query, {"season": season, "as_of_date": as_of_date})
        total, ks, bbs = cur.fetchone()
    if not total:
        return None, None
    return round(100 * ks / total, 1), round(100 * bbs / total, 1)
