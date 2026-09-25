#!/usr/bin/env python3
"""
One-off data repair for the phase A fixes (24 Sep 2026). Safe to re-run:
every step replaces or upserts, and checks its own result before it commits.

    python scripts/repair_data.py --seasons 2025 --steps venues resumed lineups batter_form bullpen

Steps, in the order they run for each season:

  venues       fill mlb.games.venue_id from the schedule feed (one light call
               per season). Ids survive sponsor renames; names don't
  resumed      record suspended-and-resumed games in mlb.resumed_games
  lineups      rebuild mlb.lineup from each box score with the fixed parser
               (true starters, not end-of-game occupants), cross-checked
               against the pitch file, then delete-and-reinsert per game
  batter_form  starting_batter_form follows the lineup: rows for players who
               didn't start are removed, starters without a row get one
  bullpen      recompute every bullpen_status row with the per-reliever rules
  innings      (added 25 Sep) runs per half inning from the pitch file into
               mlb.inning_scores, checked against game_results' first-five
               score before anything is written

ALL-OR-NOTHING (after an independent review, 24 Sep 2026): each group
computes everything first, runs its checks, and only then writes, in one
transaction that is committed only after the written result is re-checked
in the database. A failed check leaves the season exactly as it was.
lineups and batter_form commit together, so the lineup table and the batter
form table can never disagree. bullpen commits on its own.

The pitch-file checks need data/pitches_<season>.parquet (the workflow
downloads it). 2018-2020 have no pitch files and no form rows, so they get
venues, resumed and lineups (without the cross-check) only.
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2.extras  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from pipelines import pitch_store  # noqa: E402
from pipelines.config import SEASON_END_MD, SEASON_START_MD  # noqa: E402
from pipelines.db import get_conn, replace_game_rows, upsert_rows  # noqa: E402
from pipelines.games import bullpen_status  # noqa: E402
from pipelines.games.bullpen_status import build_bullpen_status_row  # noqa: E402
from pipelines.games.lineup import LINEUP_KEEP_COLS, LINEUP_KEY, build_lineup_rows, lineup_problems  # noqa: E402
from pipelines.games.lineup_check import compare_lineups, pitch_lineups  # noqa: E402
from pipelines.games.inning_scores import INNING_KEY, build_inning_rows, compare_first_five, half_starts  # noqa: E402
from pipelines.games.resumed_games import RESUMED_KEY, resumed_game_rows, venue_ids_by_game  # noqa: E402
from pipelines.mlb_stats_client import (  # noqa: E402
    get_career_totals_before_season,
    get_live_feed,
    get_schedule_light,
    get_stats_by_date_range_bulk,
)
from pipelines.player_form.starting_batter_form import SEASON_START, build_starting_batter_form_row  # noqa: E402
from pipelines.reference.players import ensure_players_exist  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("pipelines.db").setLevel(logging.WARNING)  # one "upserted 18 rows" line per game is noise
log = logging.getLogger("repair_data")

STEPS = ["venues", "resumed", "lineups", "batter_form", "bullpen", "innings"]
FETCH_WORKERS = 4
WRITE_CHUNK = 1000
# A box score whose starters aren't exactly slots 1-9 is skipped (its old
# rows are left alone) and listed. More than this share of a season's games
# means the parser is wrong, not the box scores, so nothing is written.
MAX_PROBLEM_SHARE = 0.002
# Share of team-games with at least one reliever on back-to-back days.
# 21.5% on a week of June 2025 in testing; the old bug gave 95.6%. Outside
# this band the rules or the box-score parsing are off, so nothing is written.
BULLPEN_B2B_BAND = (0.05, 0.60)


class RepairFailed(RuntimeError):
    pass


def bounded_map(fn, items, workers: int = FETCH_WORKERS, window: int = 48):
    """Like ThreadPoolExecutor.map, in order, but with at most `window` calls
    in flight. Plain pool.map submits everything at once and holds every
    finished result until it's consumed -- for live feeds (1-3 MB each) a
    whole season would sit in memory at once."""
    items = list(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(items), window):
            yield from pool.map(fn, items[start:start + window])


# ---------------------------------------------------------------------------
# shared lookups
# ---------------------------------------------------------------------------
_schedule_cache: dict[int, list[dict]] = {}


def season_schedule(season: int) -> list[dict]:
    if season not in _schedule_cache:
        _schedule_cache[season] = get_schedule_light(f"{season}-{SEASON_START_MD}", f"{season}-{SEASON_END_MD}")
        log.info("[%s] schedule: %d entries", season, len(_schedule_cache[season]))
    return _schedule_cache[season]


def season_games(conn, season: int) -> list[dict]:
    """Every game of the season in mlb.games. `completed` = has a completed
    result; only those have a box-score lineup worth rebuilding."""
    with conn.cursor() as cur:
        cur.execute(
            """
            select g.game_id, g.date, g.home_team, g.away_team, g.umpire_id,
                   coalesce(gr.game_status = 'completed', false) as completed
            from mlb.games g
            left join mlb.game_results gr on gr.game_id = g.game_id
            where g.season = %s
            order by g.date, g.game_id
            """,
            (season,),
        )
        rows = [
            {"game_id": r[0], "date": str(r[1]), "season": season, "home_team": r[2],
             "away_team": r[3], "umpire_id": r[4], "completed": r[5]}
            for r in cur.fetchall()
        ]
    conn.commit()  # end the read transaction; long API work follows
    return rows


def _fetch(game_id: int):
    return game_id, get_live_feed(game_id)


# ---------------------------------------------------------------------------
# venues, resumed games
# ---------------------------------------------------------------------------
def repair_venues(conn, season: int, games: list[dict]) -> None:
    ids = venue_ids_by_game(season_schedule(season))
    pairs = [(g["game_id"], ids[g["game_id"]]) for g in games if g["game_id"] in ids]
    missing = [g["game_id"] for g in games if g["game_id"] not in ids]
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "update mlb.games g set venue_id = v.venue_id "
                "from (values %s) as v(game_id, venue_id) where g.game_id = v.game_id",
                pairs,
                page_size=1000,
            )
            cur.execute("select count(*), count(venue_id) from mlb.games where season = %s", (season,))
            total, filled = cur.fetchone()
        if total and filled < 0.995 * total:
            raise RepairFailed(f"[{season}] venue_id would be filled for only {filled} of {total} games")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    log.info("[%s] venues: %d of %d games have a venue_id", season, filled, total)
    if missing:
        log.warning("[%s] venues: %d games not in the schedule feed, left null: %s", season, len(missing), missing[:20])


def repair_resumed(conn, season: int, games: list[dict]) -> None:
    rows = resumed_game_rows(season_schedule(season))
    known = {g["game_id"] for g in games}
    unknown = [r["game_id"] for r in rows if r["game_id"] not in known]
    rows = [r for r in rows if r["game_id"] in known]
    try:
        upsert_rows(conn, "resumed_games", rows, conflict_cols=RESUMED_KEY, strict=True)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    for r in rows:
        gap = (date.fromisoformat(r["resume_date"]) - date.fromisoformat(r["original_date"])).days
        log.info("[%s] resumed: game %s, %s -> %s (%d days), %s", season, r["game_id"],
                 r["original_date"], r["resume_date"], gap, r["status"])
    if unknown:
        log.warning("[%s] resumed: %d games not in mlb.games, skipped: %s", season, len(unknown), unknown)
    log.info("[%s] resumed: %d games recorded", season, len(rows))


# ---------------------------------------------------------------------------
# lineups + batter form: plan, check, then write together
# ---------------------------------------------------------------------------
def plan_lineups(season: int, games: list[dict], pitches) -> dict:
    """Fetch every completed game's box score and build its starting lineup.
    Nothing is written. Raises before any write if the lineups look wrong."""
    played = [g for g in games if g["completed"]]
    by_id = {g["game_id"]: g for g in played}
    rows_by_game: dict[int, list[dict]] = {}
    problems: list[tuple[int, list[str]]] = []
    for i, (game_id, feed) in enumerate(bounded_map(_fetch, [g["game_id"] for g in played]), 1):
        bullpen_status.prime_cache(game_id, feed)  # the bullpen step reuses this box score
        rows = build_lineup_rows(game_id, feed=feed)
        g = by_id[game_id]
        issues = lineup_problems(rows, [g["home_team"], g["away_team"]])
        if issues:
            problems.append((game_id, issues))
        else:
            rows_by_game[game_id] = rows
        if i % 250 == 0:
            log.info("[%s] lineups: fetched %d/%d box scores", season, i, len(played))

    for game_id, issues in problems[:30]:
        log.warning("[%s] lineups: game %s will be skipped: %s", season, game_id, "; ".join(issues))
    if len(problems) > MAX_PROBLEM_SHARE * len(played):
        raise RepairFailed(f"[{season}] {len(problems)} of {len(played)} games had malformed starting lineups; nothing written")

    if pitches is None:
        log.info("[%s] lineups: no pitch file, cross-check skipped", season)
    else:
        home_away = {g["game_id"]: (g["home_team"], g["away_team"]) for g in played}
        slots, batted = pitch_lineups(pitches, home_away)
        result = compare_lineups([r for rows in rows_by_game.values() for r in rows], slots, batted)
        log.info(
            "[%s] lineup cross-check: %d team-games, %d of %d slots agree (%.3f%%); "
            "%d starters never batted (expected), %d other disagreements, %d team-games had no pitch data",
            season, result["team_games_compared"], result["slots_agree"], result["slots_compared"],
            100 * result["agreement"], result["never_batted"], len(result["other"]), result["team_games_skipped"],
        )
        for m in result["other"][:30]:
            log.warning("[%s] lineup disagreement: %s", season, m)
        if not result["passed"]:
            raise RepairFailed(f"[{season}] lineup cross-check agreement {result['agreement']:.4f} is below 0.995; nothing written")

    log.info("[%s] lineups: %d games planned, %d skipped", season, len(rows_by_game), len(problems))
    return {"rows_by_game": rows_by_game, "problems": problems}


def plan_batter_form(conn, season: int, games: list[dict], pitches, rows_by_game: dict) -> dict | None:
    """Which form rows to remove and which to add so starting_batter_form
    matches the (planned) lineup table exactly. Computes the new rows."""
    with conn.cursor() as cur:
        cur.execute(
            "select f.game_id, f.batter_id from mlb.starting_batter_form f "
            "join mlb.games g on g.game_id = f.game_id where g.season = %s",
            (season,),
        )
        existing = {(r[0], r[1]) for r in cur.fetchall()}
        cur.execute(
            "select l.game_id, l.player_id from mlb.lineup l "
            "join mlb.games g on g.game_id = l.game_id where g.season = %s",
            (season,),
        )
        lineup_now = {(r[0], r[1]) for r in cur.fetchall()}
        cur.execute("select player_id, debut_date from mlb.players")
        debut_by_player = {r[0]: str(r[1]) if r[1] else None for r in cur.fetchall()}
    conn.commit()  # end the read transaction before the API calls below

    if pitches is None:
        if not existing:
            log.info("[%s] batter_form: no pitch file and no form rows (pre-2021 season), nothing to do", season)
            return None
        raise RepairFailed(f"[{season}] batter_form needs data/pitches_{season}.parquet")

    target = {(gid, r["player_id"]) for gid, rows in rows_by_game.items() for r in rows}
    target |= {(gid, pid) for gid, pid in lineup_now if gid not in rows_by_game}
    stale = existing - target
    missing = sorted(target - existing, key=lambda k: k[0])
    log.info("[%s] batter_form: %d rows to remove (non-starters), %d starters need a row", season, len(stale), len(missing))

    by_game: dict[int, list[int]] = {}
    for gid, pid in missing:
        by_game.setdefault(gid, []).append(pid)
    date_by_game = {g["game_id"]: g["date"] for g in games}
    career = get_career_totals_before_season(sorted({p for _, p in missing}), "hitting", season)
    season_start = SEASON_START.format(season=season)

    def prefetch(gid):
        game_date = date_by_game[gid]
        as_of_end = (date.fromisoformat(game_date) - timedelta(days=1)).isoformat()
        last30_start = (date.fromisoformat(game_date) - timedelta(days=30)).isoformat()
        ids = by_game[gid]
        return gid, (get_stats_by_date_range_bulk(ids, "hitting", season_start, as_of_end),
                     get_stats_by_date_range_bulk(ids, "hitting", last30_start, as_of_end))

    new_rows: list[dict] = []
    for i, (gid, (hit_season, hit_last30)) in enumerate(bounded_map(prefetch, list(by_game)), 1):
        for pid in by_game[gid]:
            new_rows.append(build_starting_batter_form_row(
                pitches, pid, gid, date_by_game[gid], season, debut_by_player.get(pid),
                prefetched={"season": hit_season.get(pid), "last30": hit_last30.get(pid),
                            "career_before_season": career.get(pid)},
            ))
        if i % 250 == 0:
            log.info("[%s] batter_form: computed %d/%d games", season, i, len(by_game))
    return {"stale": sorted(stale), "new_rows": new_rows, "target": len(target)}


def write_lineups_and_form(conn, season: int, lplan: dict | None, fplan: dict | None) -> None:
    """One transaction: lineups replaced, stale form rows removed, new form
    rows added, then re-checked in the database. Commits only if every check
    passes; otherwise rolls back and raises."""
    rows_by_game = lplan["rows_by_game"] if lplan else {}
    skipped = {gid for gid, _ in lplan["problems"]} if lplan else set()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from mlb.lineup l join mlb.games g on g.game_id = l.game_id where g.season = %s",
                (season,),
            )
            before = cur.fetchone()[0]
        deleted = inserted = 0
        if rows_by_game:
            ensure_players_exist(conn, {r["player_id"] for rows in rows_by_game.values() for r in rows})
            for i, (gid, rows) in enumerate(rows_by_game.items(), 1):
                d, n = replace_game_rows(conn, "lineup", gid, rows, LINEUP_KEY, keep_cols=LINEUP_KEEP_COLS)
                deleted += d
                inserted += n
                if i % 500 == 0:
                    log.info("[%s] lineups: wrote %d/%d games", season, i, len(rows_by_game))
            log.info("[%s] lineups: %d rows before, %d deleted, %d inserted", season, before, deleted, inserted)

        if fplan:
            with conn.cursor() as cur:
                if fplan["stale"]:
                    psycopg2.extras.execute_values(
                        cur,
                        "delete from mlb.starting_batter_form f using (values %s) as v(game_id, batter_id) "
                        "where f.game_id = v.game_id and f.batter_id = v.batter_id",
                        fplan["stale"],
                        page_size=WRITE_CHUNK,
                    )
            new_rows = fplan["new_rows"]
            for start in range(0, len(new_rows), WRITE_CHUNK):
                upsert_rows(conn, "starting_batter_form", new_rows[start:start + WRITE_CHUNK],
                            conflict_cols=["batter_id", "game_id"], strict=True)
            log.info("[%s] batter_form: removed %d, added %d", season, len(fplan["stale"]), len(new_rows))

        # Re-check what is now in the database, inside the same transaction.
        with conn.cursor() as cur:
            cur.execute(
                """
                select l.game_id from mlb.lineup l join mlb.games g on g.game_id = l.game_id
                where g.season = %s
                group by l.game_id, l.team_id
                having count(*) <> 9 or count(distinct batting_order_slot) <> 9
                """,
                (season,),
            )
            bad = {r[0] for r in cur.fetchall()} - skipped
            if bad:
                raise RepairFailed(f"[{season}] {len(bad)} rebuilt games don't have exactly 9 starters per team, e.g. {sorted(bad)[:5]}")
            if fplan:
                cur.execute(
                    """
                    select
                      (select count(distinct (l.game_id, l.player_id)) from mlb.lineup l
                         join mlb.games g on g.game_id = l.game_id where g.season = %(s)s),
                      (select count(*) from mlb.starting_batter_form f
                         join mlb.games g on g.game_id = f.game_id where g.season = %(s)s),
                      (select count(*) from mlb.lineup l join mlb.games g on g.game_id = l.game_id
                         where g.season = %(s)s and not exists (
                           select 1 from mlb.starting_batter_form f
                           where f.game_id = l.game_id and f.batter_id = l.player_id))
                    """,
                    {"s": season},
                )
                n_lineup, n_form, starters_without_form = cur.fetchone()
                log.info("[%s] check: %d starter keys, %d form rows, %d starters without a form row",
                         season, n_lineup, n_form, starters_without_form)
                if n_lineup != n_form or starters_without_form:
                    raise RepairFailed(
                        f"[{season}] batter form would have {n_form} rows for {n_lineup} starters "
                        f"({starters_without_form} starters without a row)"
                    )
        conn.commit()
    except Exception:
        conn.rollback()
        log.error("[%s] lineup/batter form write rolled back; the season is unchanged", season)
        raise
    log.info("[%s] lineups and batter form committed", season)


# ---------------------------------------------------------------------------
# bullpen
# ---------------------------------------------------------------------------
def repair_bullpen(conn, season: int, games: list[dict]) -> None:
    # Box scores the lineup step didn't already fetch (e.g. --steps bullpen alone)
    need = [g["game_id"] for g in games if g["completed"] and g["game_id"] not in bullpen_status._PITCHING_LINES_CACHE]
    for game_id, feed in bounded_map(_fetch, need):
        bullpen_status.prime_cache(game_id, feed)
    if need:
        log.info("[%s] bullpen: fetched %d box scores", season, len(need))

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                select gr.game_id, g.home_team, g.away_team, gr.actual_home_starter_id, gr.actual_away_starter_id
                from mlb.game_results gr join mlb.games g on g.game_id = gr.game_id
                where g.season = %s
                """,
                (season,),
            )
            starter_lookup = {}
            for gid, home, away, hs, as_ in cur.fetchall():
                starter_lookup[(gid, home)] = hs
                starter_lookup[(gid, away)] = as_
            cur.execute(
                """
                select b.team_id, b.game_id, g.date
                from mlb.bullpen_status b join mlb.games g on g.game_id = b.game_id
                where g.season = %s order by g.date, b.game_id
                """,
                (season,),
            )
            targets = [(r[0], r[1], str(r[2])) for r in cur.fetchall()]

        rows = []
        for i, (team_id, game_id, game_date) in enumerate(targets, 1):
            rows.append(build_bullpen_status_row(conn, team_id, game_id, game_date, season, starter_lookup))
            if i % 1000 == 0:
                log.info("[%s] bullpen: computed %d/%d team-games", season, i, len(targets))
        for start in range(0, len(rows), WRITE_CHUNK):
            upsert_rows(conn, "bullpen_status", rows[start:start + WRITE_CHUNK],
                        conflict_cols=["team_id", "game_id"], strict=True)

        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*), count(relievers_back_to_back), avg(back_to_back_appearances::int),
                       avg(relievers_back_to_back), avg(cardinality(unavailable_reliever_ids)),
                       avg((not closer_available_flag)::int)
                from mlb.bullpen_status b join mlb.games g on g.game_id = b.game_id
                where g.season = %s
                """,
                (season,),
            )
            n, filled, b2b, avg_b2b, avg_unavail, closer_out = cur.fetchone()
        b2b = float(b2b or 0)
        log.info(
            "[%s] bullpen: %d rows, %d with the new columns; any back-to-back %.1f%%, avg back-to-back "
            "relievers %.2f, avg unavailable %.2f, closer unavailable %.1f%%",
            season, n, filled, 100 * b2b, float(avg_b2b or 0), float(avg_unavail or 0), 100 * float(closer_out or 0),
        )
        if filled != n or n != len(targets):
            raise RepairFailed(f"[{season}] bullpen: {filled} of {n} rows rewritten, expected {len(targets)}")
        if n >= 500 and not (BULLPEN_B2B_BAND[0] <= b2b <= BULLPEN_B2B_BAND[1]):
            raise RepairFailed(f"[{season}] bullpen: back-to-back share {b2b:.1%} is outside {BULLPEN_B2B_BAND}")
        conn.commit()
    except Exception:
        conn.rollback()
        log.error("[%s] bullpen write rolled back; the season is unchanged", season)
        raise


# ---------------------------------------------------------------------------
# innings
# ---------------------------------------------------------------------------
MAX_INNING_PROBLEM_SHARE = 0.005
MIN_F5_AGREEMENT = 0.995


def completed_results(conn, season: int) -> dict[int, dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select g.game_id, g.home_team, g.away_team, r.home_score_final, r.away_score_final,
                   r.home_score_f5, r.away_score_f5, r.innings_played
            from mlb.games g join mlb.game_results r on r.game_id = g.game_id
            where g.season = %s and r.game_status = 'completed'
            """,
            (season,),
        )
        out = {
            r[0]: {"home_team": r[1], "away_team": r[2], "home_final": r[3], "away_final": r[4],
                   "home_f5": r[5], "away_f5": r[6], "innings_played": r[7]}
            for r in cur.fetchall()
        }
    conn.commit()
    return out


def repair_innings(conn, season: int, pitches) -> None:
    """Rebuild a season of mlb.inning_scores. Computes and checks everything
    first; writes in one transaction, re-checked before commit."""
    if pitches is None:
        log.info("[%s] innings: no pitch file (pre-2021 season), nothing to do", season)
        return
    games = completed_results(conn, season)
    rows, problems = build_inning_rows(half_starts(pitches), games)
    f5 = compare_first_five(rows, games)
    log.info(
        "[%s] innings: %d completed games, %d half-inning rows for %d games, %d games with problems; "
        "first five agrees on %d of %d games (%.3f%%)",
        season, len(games), len(rows), len({r["game_id"] for r in rows}), len(problems),
        f5["agree"], f5["compared"], 100 * f5["agreement"],
    )
    for game_id, why in problems[:30]:
        log.warning("[%s] innings: game %s skipped: %s", season, game_id, why)
    for m in f5["misses"][:30]:
        log.warning("[%s] innings: first-five mismatch %s", season, m)
    if len({g for g, _ in problems}) > MAX_INNING_PROBLEM_SHARE * max(len(games), 1):
        raise RepairFailed(f"[{season}] innings: {len(problems)} of {len(games)} games failed checks; nothing written")
    if f5["agreement"] < MIN_F5_AGREEMENT:
        raise RepairFailed(f"[{season}] innings: first-five agreement {f5['agreement']:.4f} below {MIN_F5_AGREEMENT}; nothing written")
    # A game whose first five doesn't match is known to be wrong: skip it too.
    bad_f5 = {m["game_id"] for m in f5["misses"]}
    rows = [r for r in rows if r["game_id"] not in bad_f5]
    write_ids = sorted({r["game_id"] for r in rows})

    try:
        with conn.cursor() as cur:
            # Only the games being rewritten; a game that fails the checks keeps
            # whatever rows it had rather than losing them.
            cur.execute("delete from mlb.inning_scores where game_id = any(%s)", (write_ids,))
            removed = cur.rowcount
        for start in range(0, len(rows), WRITE_CHUNK * 5):
            upsert_rows(conn, "inning_scores", rows[start:start + WRITE_CHUNK * 5], conflict_cols=INNING_KEY, strict=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*),
                       count(*) filter (where t.runs_total <> case when t.is_home then r.home_score_final
                                                                  else r.away_score_final end)
                from mlb.team_game_runs t
                join mlb.games g on g.game_id = t.game_id
                join mlb.game_results r on r.game_id = t.game_id
                where g.season = %s
                """,
                (season,),
            )
            team_games, bad_totals = cur.fetchone()
        if bad_totals:
            raise RepairFailed(f"[{season}] innings: {bad_totals} team-games don't add up to the final score")
        conn.commit()
    except Exception:
        conn.rollback()
        log.error("[%s] innings write rolled back; the season is unchanged", season)
        raise
    log.info("[%s] innings: removed %d old rows, wrote %d; %d team-games add up to their final score",
             season, removed, len(rows), team_games)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, required=True)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    args = parser.parse_args()
    load_dotenv()

    steps = [s for s in STEPS if s in args.steps]  # always run in dependency order
    for season in args.seasons:
        bullpen_status.clear_cache()
        pitch_path = pitch_store.season_file(season)
        with get_conn() as conn:
            games = season_games(conn, season)
            log.info("[%s] %d games in mlb.games, %d completed; steps %s", season, len(games),
                     sum(g["completed"] for g in games), steps)
            pitches = pitch_store.open_pitch_source(season, games) if pitch_path.exists() else None
            if pitches is None:
                log.info("[%s] no pitch file at %s", season, pitch_path)
            try:
                if "venues" in steps:
                    repair_venues(conn, season, games)
                if "resumed" in steps:
                    repair_resumed(conn, season, games)
                lplan = plan_lineups(season, games, pitches) if "lineups" in steps else None
                fplan = None
                if "batter_form" in steps:
                    fplan = plan_batter_form(conn, season, games, pitches, lplan["rows_by_game"] if lplan else {})
                if lplan or fplan:
                    write_lineups_and_form(conn, season, lplan, fplan)
                if "bullpen" in steps:
                    repair_bullpen(conn, season, games)
                if "innings" in steps:
                    repair_innings(conn, season, pitches)
            finally:
                if pitches is not None:
                    pitches.close()
        log.info("[%s] repair complete", season)


if __name__ == "__main__":
    main()
