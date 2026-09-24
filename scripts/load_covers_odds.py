"""
Load the Covers odds file (2022-2026) into mlb.odds_moneyline, odds_runline
and odds_totals.

    python scripts/load_covers_odds.py data/covers_odds_load_2022_2026.csv.gz

WHERE THE FILE COMES FROM
Covers.com keeps a per-game "Line Movement" history for 9 sportsbooks. It was
scraped once (23-24 Sep 2026) through a browser on Colin's machine, matched to
our game_ids, and reduced to three prices per game x market x book:

  open  = earliest clean pre-game price Covers logged
  p1    = price in effect at 10:00 am Central on game day (the model's
          morning prediction point). If first pitch is before 10:05 CT, the
          price 5 min before first pitch is used instead
  close = price in effect 5 min before first pitch

"In effect at time T" = the last clean change at or before T. Covers only
logs a row when a book CHANGES its price, so the last change before T is the
price you could have bet at T, even if it was set hours earlier.

"Clean" = both prices present, |odds| <= 1000, two-way implied probability
between 0.99 and 1.15, run line exactly +/-1.5, totals quoting the same
number on both sides, and timestamped at least 5 min before first pitch.
Books post junk ticks when they switch a game to live betting (e.g. a total
of 4.5 at -1099/-855); these rules keep those out.

WHY THE FILE IS ENCRYPTED IN THE RELEASE
This repo is public. The odds are scraped from a commercial site, so they are
kept out of public view: the release asset is AES-encrypted and the key lives
only in the ODDS_FILE_KEY Actions secret. The decrypted file exists only
inside the runner.

STRICTNESS
A silent partial write is worse than a crash (see pipeline-build-status.md).
This script:
  1. checks the file's row counts against the numbers below before writing
  2. refuses to run if any game_id is missing from mlb.games
  3. after writing, re-counts rows in the database and fails if they differ
"""
from __future__ import annotations

import csv
import gzip
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import psycopg2.extras

# Make the repo root importable (running `python scripts/x.py` puts scripts/,
# not the repo root, on sys.path[0]). Same fix as scripts/backfill.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.db import get_conn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("load_covers_odds")

SOURCE = "covers"

# Row counts in the file, per (table, snapshot_type). Computed when the file
# was built on 24 Sep 2026. A mismatch means the wrong or a truncated file.
EXPECTED = {
    ("odds_moneyline", "open"): 80566, ("odds_moneyline", "p1"): 79681, ("odds_moneyline", "close"): 80566,
    ("odds_runline", "open"): 74971, ("odds_runline", "p1"): 73789, ("odds_runline", "close"): 74971,
    ("odds_totals", "open"): 80580, ("odds_totals", "p1"): 79670, ("odds_totals", "close"): 80580,
}

# Columns written per table (everything else in the table stays NULL,
# including devigged_* on purpose: the devig method is chosen later).
TABLE_COLS = {
    "odds_moneyline": ["game_id", "book", "snapshot_type", "source", "timestamp",
                       "home_odds", "away_odds", "implied_prob_home", "implied_prob_away"],
    "odds_runline": ["game_id", "book", "snapshot_type", "source", "timestamp", "line",
                     "home_odds", "away_odds", "implied_prob_home", "implied_prob_away"],
    "odds_totals": ["game_id", "book", "snapshot_type", "source", "timestamp", "total_line",
                    "over_odds", "under_odds", "implied_prob_over", "implied_prob_under"],
}
INT_COLS = {"game_id", "home_odds", "away_odds", "over_odds", "under_odds"}
NUM_COLS = {"line", "total_line", "implied_prob_home", "implied_prob_away",
            "implied_prob_over", "implied_prob_under"}
BATCH = 5000


def read_file(path):
    rows = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = r["table"]
            out = {"source": SOURCE}
            for c in TABLE_COLS[t]:
                if c == "source":
                    continue
                v = r[c]
                if c in INT_COLS:
                    v = int(v)
                elif c in NUM_COLS:
                    v = float(v) if v != "" else None
                out[c] = v
            rows[t].append(out)
    return rows


def main(path):
    rows = read_file(path)

    # 1. File row counts must match what was built.
    got = Counter()
    for t, rs in rows.items():
        for r in rs:
            got[(t, r["snapshot_type"])] += 1
    if dict(got) != EXPECTED:
        log.error("row counts in the file don't match EXPECTED:\n got %s\n want %s", dict(got), EXPECTED)
        sys.exit(1)
    log.info("file OK: %d rows", sum(got.values()))

    with get_conn() as conn, conn.cursor() as cur:
        # 2. Every game_id must already exist (foreign key to mlb.games).
        ids = sorted({r["game_id"] for rs in rows.values() for r in rs})
        cur.execute("select game_id from mlb.games where game_id = any(%s)", (ids,))
        found = {x[0] for x in cur.fetchall()}
        missing = [g for g in ids if g not in found]
        if missing:
            log.error("%d game_ids not in mlb.games, e.g. %s. Nothing written.", len(missing), missing[:10])
            sys.exit(1)
        log.info("all %d game_ids exist in mlb.games", len(ids))

    # 3. Write. One transaction per table, so a failure leaves each table
    #    either fully loaded or untouched. ON CONFLICT makes a re-run safe.
    for t, cols in TABLE_COLS.items():
        key = ["game_id", "book", "snapshot_type"]
        upd = [c for c in cols if c not in key]
        sql = (f'insert into mlb.{t} ({", ".join(chr(34) + c + chr(34) for c in cols)}, pulled_at) values %s '
               f'on conflict ({", ".join(key)}) do update set '
               + ", ".join(f'"{c}" = excluded."{c}"' for c in upd) + ", pulled_at = now()")
        template = "(" + ", ".join(f"%({c})s" for c in cols) + ", now())"
        data = rows[t]
        with get_conn() as conn, conn.cursor() as cur:
            for i in range(0, len(data), BATCH):
                psycopg2.extras.execute_values(cur, sql, data[i:i + BATCH], template=template, page_size=BATCH)
            log.info("%s: wrote %d rows", t, len(data))

    # 4. Re-count in the database. Fail loudly on any difference.
    with get_conn() as conn, conn.cursor() as cur:
        bad = []
        for (t, snap), n in EXPECTED.items():
            cur.execute(f"select count(*) from mlb.{t} where source = %s and snapshot_type = %s", (SOURCE, snap))
            have = cur.fetchone()[0]
            log.info("%-15s %-6s expected %6d  in database %6d", t, snap, n, have)
            if have != n:
                bad.append((t, snap, n, have))
        if bad:
            log.error("database counts don't match: %s", bad)
            sys.exit(1)
    log.info("done: all counts match")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/load_covers_odds.py <covers_odds_load_2022_2026.csv.gz>")
    main(sys.argv[1])
