#!/usr/bin/env python3
"""
Load Triple-A and Double-A season lines for every player in mlb.players into
mlb.player_minor_season_stats (model design group 8, rookie translations,
3 Oct 2026). Safe to re-run: the table is replaced in one transaction, and
only after the checks below pass.

    python scripts/build_minor_seasons.py --dry-run 100     # fetch only, write nothing
    python scripts/build_minor_seasons.py                   # full build

The dry run fetches the 100 most recent MLB debutants, prints how many
minor-league lines came back per level and season, and stops. Run it first:
it is the check that the API's sportId filter works the way this loader
assumes (it could not be tested when the loader was written).

Checks before anything is committed:
  1. at least one Triple-A hitting and one Triple-A pitching line came back
     (if the API ignored the sportId filter, no line passes the level check
     and this fails)
  2. every line's level is the level asked for (guaranteed by the parser)
  3. of players who debuted in MLB in 2021 or 2022, the share with at least
     one minor-league line before their debut season is printed; below 50%
     is a warning, below 10% a failure (international signings and
     two-way players skip levels, so it is not expected to be 100%)
  4. after writing, the row count in the database matches what was built
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2.extras  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from pipelines.db import get_conn, upsert_rows  # noqa: E402
from pipelines.reference.minor_seasons import (  # noqa: E402
    LEVELS, MINOR_KEY, debut_coverage, get_people_minor_year_by_year, minor_season_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("pipelines.db").setLevel(logging.WARNING)
log = logging.getLogger("build_minor_seasons")

CHUNK = 25
WORKERS = 4
FIRST_SEASON = 2017       # lines older than this are dropped (priors look back 3 seasons from 2021)


class BuildFailed(RuntimeError):
    pass


def fetch_level(ids: list[int], sport_id: int) -> list[dict]:
    chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
    people: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, part in enumerate(pool.map(lambda c: get_people_minor_year_by_year(c, sport_id), chunks), 1):
            people.extend(part)
            if i % 10 == 0:
                log.info("level %s: fetched %d/%d chunks", sport_id, i, len(chunks))
    return people


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                    help="fetch only the N most recent MLB debutants and write nothing")
    a = ap.parse_args()
    load_dotenv()
    with get_conn() as conn:
        with conn.cursor() as cur:
            if a.dry_run:
                cur.execute("select player_id from mlb.players where debut_date is not null "
                            "order by debut_date desc limit %s", (a.dry_run,))
            else:
                cur.execute("select player_id from mlb.players order by player_id")
            ids = [r[0] for r in cur.fetchall()]
            cur.execute("select player_id, debut_date from mlb.players where debut_date is not null")
            debut = {r[0]: r[1] for r in cur.fetchall()}
        conn.commit()
        log.info("%d players to fetch", len(ids))

        rows: list[dict] = []
        for name, sport_id in LEVELS.items():
            people = fetch_level(ids, sport_id)
            part = [r for p in people if p.get("id") in set(ids) for r in minor_season_rows(p, sport_id)
                    if r["season"] >= FIRST_SEASON]
            by_season: dict[int, int] = {}
            for r in part:
                by_season[r["season"]] = by_season.get(r["season"], 0) + 1
            log.info("%s (sport %s): API returned %d players, %d season lines (%d hitting, %d pitching); by season %s",
                     name, sport_id, len(people), len(part), sum(r["stat_group"] == "hitting" for r in part),
                     sum(r["stat_group"] == "pitching" for r in part), dict(sorted(by_season.items())))
            rows.extend(part)

        aaa = [r for r in rows if r["sport_id"] == 11]
        for g in ("hitting", "pitching"):
            if not any(r["stat_group"] == g for r in aaa):
                raise BuildFailed(f"no Triple-A {g} line came back; the sportId filter may not work as assumed; nothing written")

        n_rookies, n_have = debut_coverage(rows, debut, ids)
        share = n_have / n_rookies if n_rookies else 1.0
        log.info("players debuting in 2021-2022 in this fetch: %d; with a minor-league line before their debut season: %d (%.1f%%)",
                 n_rookies, n_have, 100 * share)
        if n_rookies and share < 0.5:
            log.warning("fewer than half of recent debutants have a minor-league line before debut")
        if n_rookies and share < 0.1:
            raise BuildFailed("almost no recent debutant has a minor-league line before debut; nothing written")

        if a.dry_run:
            for r in rows[:5]:
                log.info("sample: %s", {k: v for k, v in r.items() if k != "stat_json"})
            log.info("dry run: nothing written")
            return

        for r in rows:
            r["stat_json"] = psycopg2.extras.Json(r["stat_json"])
        try:
            with conn.cursor() as cur:
                cur.execute("delete from mlb.player_minor_season_stats")
                removed = cur.rowcount
            for start in range(0, len(rows), 2000):
                upsert_rows(conn, "player_minor_season_stats", rows[start:start + 2000],
                            conflict_cols=MINOR_KEY, strict=True)
            with conn.cursor() as cur:
                cur.execute("select count(*) from mlb.player_minor_season_stats")
                n_rows = cur.fetchone()[0]
            if n_rows != len(rows):
                raise BuildFailed(f"wrote {n_rows} lines, expected {len(rows)}")
            conn.commit()
        except Exception:
            conn.rollback()
            log.error("write rolled back; mlb.player_minor_season_stats is unchanged")
            raise
        log.info("done: replaced %d lines with %d", removed, n_rows)


if __name__ == "__main__":
    main()
