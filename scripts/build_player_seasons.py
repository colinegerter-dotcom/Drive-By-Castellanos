#!/usr/bin/env python3
"""
Load official MLB season lines for every player in mlb.players into
mlb.player_season_stats, and fill players.birth_date (model design A5,
25 Sep 2026). Safe to re-run: the whole table is replaced in one
transaction, and only after every check below passes.

    python scripts/build_player_seasons.py

Checks, all before anything is committed:
  1. the API returned every player asked for (at most 0.5% missing)
  2. a birth date for at least 99.5% of players
  3. the design's validation: every 2022 starting hitter and starting pitcher
     with MLB time before 2022 has an official line from an earlier season
     in the right group (hitting or pitching), at least 99.5% of them
  4. after writing, the row counts in the database match what was built

See pipelines/reference/player_seasons.py for why a traded player's season
is MLB's combined line and never the team stints added up.
"""
from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2.extras  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from pipelines.db import get_conn, upsert_rows  # noqa: E402
from pipelines.mlb_stats_client import get_people_year_by_year  # noqa: E402
from pipelines.reference.player_seasons import PLAYER_SEASON_KEY, birth_date, player_season_rows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("pipelines.db").setLevel(logging.WARNING)
log = logging.getLogger("build_player_seasons")

CHUNK = 25  # two stat groups per player, veterans have 15+ seasons: keep responses small
WORKERS = 4
MIN_SHARE = 0.995
CHECK_SEASON = 2022


class BuildFailed(RuntimeError):
    pass


def fetch_all(ids: list[int]) -> list[dict]:
    chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
    people: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, part in enumerate(pool.map(get_people_year_by_year, chunks), 1):
            people.extend(part)
            if i % 10 == 0:
                log.info("fetched %d/%d chunks", i, len(chunks))
    return people


def coverage(conn, rows: list[dict], season: int) -> tuple[int, int, list]:
    """Starters in `season` who debuted before it, and how many have an
    official line from an earlier season in the right group."""
    have = {(r["player_id"], r["stat_group"]) for r in rows if r["season"] < season}
    with conn.cursor() as cur:
        cur.execute(
            """
            select distinct l.player_id, 'hitting'
            from mlb.lineup l
            join mlb.games g on g.game_id = l.game_id
            join mlb.players p on p.player_id = l.player_id
            where g.season = %(s)s and g.game_type = 'R' and p.debut_date < make_date(%(s)s, 1, 1)
              and coalesce(l.defensive_position, '') <> 'P'
            union
            select distinct x.pid, 'pitching'
            from (
                select r.actual_home_starter_id pid, g.season from mlb.game_results r join mlb.games g using (game_id)
                union all
                select r.actual_away_starter_id, g.season from mlb.game_results r join mlb.games g using (game_id)
            ) x
            join mlb.players p on p.player_id = x.pid
            where x.season = %(s)s and p.debut_date < make_date(%(s)s, 1, 1)
            """,
            {"s": season},
        )
        need = [(r[0], r[1]) for r in cur.fetchall()]
    conn.commit()
    missing = [k for k in need if k not in have]
    return len(need), len(need) - len(missing), missing


def main() -> None:
    load_dotenv()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select player_id from mlb.players order by player_id")
            ids = [r[0] for r in cur.fetchall()]
        conn.commit()
        log.info("%d players in mlb.players", len(ids))

        people = fetch_all(ids)
        got = {p.get("id") for p in people}
        missing_ids = sorted(set(ids) - got)
        rows = [r for p in people if p.get("id") in set(ids) for r in player_season_rows(p)]
        births = {p["id"]: birth_date(p) for p in people if birth_date(p)}
        seasons = sorted({r["season"] for r in rows})
        log.info(
            "API returned %d of %d players; %d season lines (%d hitting, %d pitching) across %s-%s; %d birth dates",
            len(got & set(ids)), len(ids), len(rows), sum(r["stat_group"] == "hitting" for r in rows),
            sum(r["stat_group"] == "pitching" for r in rows), seasons[0] if seasons else "-",
            seasons[-1] if seasons else "-", len(births),
        )

        if len(missing_ids) > (1 - MIN_SHARE) * len(ids):
            raise BuildFailed(f"API returned no entry for {len(missing_ids)} players, e.g. {missing_ids[:10]}; nothing written")
        if len(births) < MIN_SHARE * len(ids):
            raise BuildFailed(f"birth dates for only {len(births)} of {len(ids)} players; nothing written")
        for season in (CHECK_SEASON, CHECK_SEASON - 1):
            n_need, n_have, missing = coverage(conn, rows, season)
            share = n_have / n_need if n_need else 1.0
            log.info("%s starters with MLB time before %s: %d, with an earlier official line: %d (%.2f%%)",
                     season, season, n_need, n_have, 100 * share)
            if missing:
                log.warning("%s starters without an earlier line (first 20): %s", season, missing[:20])
            if season == CHECK_SEASON and share < MIN_SHARE:
                raise BuildFailed(f"only {n_have} of {n_need} {season} starters have a prior line; nothing written")

        for r in rows:
            r["stat_json"] = psycopg2.extras.Json(r["stat_json"])

        try:
            with conn.cursor() as cur:
                cur.execute("delete from mlb.player_season_stats")
                removed = cur.rowcount
            for start in range(0, len(rows), 2000):
                upsert_rows(conn, "player_season_stats", rows[start:start + 2000],
                            conflict_cols=PLAYER_SEASON_KEY, strict=True)
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    "update mlb.players p set birth_date = v.bd::date "
                    "from (values %s) as v(player_id, bd) where p.player_id = v.player_id",
                    sorted(births.items()),
                    page_size=1000,
                )
                cur.execute("select count(*) from mlb.player_season_stats")
                n_rows = cur.fetchone()[0]
                cur.execute("select count(birth_date) from mlb.players")
                n_births = cur.fetchone()[0]
            if n_rows != len(rows):
                raise BuildFailed(f"wrote {n_rows} season lines, expected {len(rows)}")
            if n_births < len(births):
                raise BuildFailed(f"only {n_births} birth dates in mlb.players, expected {len(births)}")
            conn.commit()
        except Exception:
            conn.rollback()
            log.error("write rolled back; mlb.player_season_stats and birth dates are unchanged")
            raise
        log.info("done: replaced %d season lines with %d; %d players now have a birth date", removed, n_rows, n_births)


if __name__ == "__main__":
    main()
