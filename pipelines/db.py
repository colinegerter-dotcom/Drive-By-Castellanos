"""
Database connection + a single generic upsert helper.

Every ingestion module in this repo funnels its writes through
`upsert_rows`. Doing it this way (one shared function instead of
one hand-written INSERT per table) is what makes the "idempotent
upserts" and "pulled_at on every automated table" rules actually
hold everywhere, instead of depending on every module remembering
to do it right.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable, Sequence

import psycopg2
import psycopg2.extras

from pipelines.config import db_dsn

log = logging.getLogger(__name__)


@contextmanager
def get_conn():
    """Open a connection, commit on success, roll back and re-raise on error.

    Usage:
        with get_conn() as conn:
            upsert_rows(conn, "teams", rows, conflict_cols=["team_id"])
    """
    conn = psycopg2.connect(db_dsn())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_rows(
    conn,
    table: str,
    rows: Sequence[dict],
    conflict_cols: Sequence[str],
    schema: str = "mlb",
    stamp_pulled_at: bool = True,
    strict: bool = False,
) -> int:
    """Idempotent bulk upsert: INSERT ... ON CONFLICT (conflict_cols) DO UPDATE.

    Why this shape and not "delete then re-insert": a delete+insert would
    momentarily remove rows a live query might read, and would defeat
    foreign keys pointing at this table. ON CONFLICT DO UPDATE is atomic
    and safe to re-run every day without creating duplicates -- which is
    the "daily pulls need idempotent upserts" rule from the schema doc.
    (Where old rows must disappear, e.g. a corrected lineup, use
    replace_game_rows below instead.)

    Every column that appears on ANY row in the batch is treated as a
    column to write; a row missing a given key gets NULL for that column.
    This keeps callers simple (they just build a dict per row) at the cost
    of assuming rows are reasonably uniform in shape, which holds for
    everything in this pipeline.

    strict (24 Sep 2026, model design item A6): in the default mode a batch
    with a few bad rows falls back to row-by-row and writes the good ones,
    logging the rest. The function then returns normally, so a caller that
    ignores the return value "succeeds" with rows missing -- the silent
    partial write this project has been bitten by several times. strict=True
    skips the fallback: any failure rolls this batch back and raises, so
    nothing is written and the job fails loudly. All new writes use it.

    Returns the number of rows written.
    """
    if not rows:
        return 0

    # pulled_at is always a fixed `now()` SQL literal, never a data value --
    # handling it as an ordinary column (present in `columns`, bound as a
    # parameter, AND appended as a literal in the UPDATE SET clause) would
    # make Postgres see two assignments to the same column in one SET clause,
    # which it rejects outright. So: strip it out of every row up front and
    # add it back exactly once, below.
    for r in rows:
        r.pop("pulled_at", None)

    # Union of all keys across all rows, so a batch with slightly uneven
    # dicts (e.g. one row missing an optional field) still works.
    columns: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)

    update_cols = [c for c in columns if c not in conflict_cols]

    insert_columns = list(columns) + (["pulled_at"] if stamp_pulled_at else [])
    insert_values_sql = ", ".join(f"%({c})s" for c in columns) + (
        ", now()" if stamp_pulled_at else ""
    )
    conflict_sql = ", ".join(conflict_cols)
    set_clauses = [f"{c} = EXCLUDED.{c}" for c in update_cols]
    if stamp_pulled_at:
        set_clauses.append("pulled_at = now()")
    # A pure-key table (every column is part of the conflict target) has no
    # SET clause to write -- fall back to a harmless no-op update so the
    # statement stays valid.
    update_sql = ", ".join(set_clauses) or f"{conflict_cols[0]} = EXCLUDED.{conflict_cols[0]}"

    query = f"""
        INSERT INTO {schema}.{table} ({", ".join(insert_columns)})
        VALUES ({insert_values_sql})
        ON CONFLICT ({conflict_sql})
        DO UPDATE SET {update_sql}
    """

    # Normalize rows: every row needs every column key present (as None if
    # absent) since we're using a single parameterized query for the whole batch.
    normalized = [{c: r.get(c) for c in columns} for r in rows]

    with conn.cursor() as cur:
        cur.execute("SAVEPOINT upsert_batch")
        try:
            psycopg2.extras.execute_batch(cur, query, normalized, page_size=500)
            cur.execute("RELEASE SAVEPOINT upsert_batch")
        except psycopg2.Error as exc:
            # Confirmed live (17 Sep 2026): a single bad row -- e.g. a
            # foreign key to a player_id that a roster pull missed -- makes
            # Postgres abort this ENTIRE batch, and without a savepoint it
            # would poison the whole surrounding transaction (backfill.py
            # shares one `with get_conn() as conn:` connection across many
            # upsert_rows calls, sometimes a full season's worth). Rolling
            # back to this savepoint undoes only this batch, not anything
            # already written earlier in the same transaction. Retrying
            # row-by-row (each in its own savepoint) then isolates exactly
            # which row(s) are bad so the rest of a good batch still lands,
            # instead of losing all of it over one row.
            cur.execute("ROLLBACK TO SAVEPOINT upsert_batch")

            if strict:
                log.error(
                    "strict upsert into %s.%s failed with %s: %s -- nothing from this batch "
                    "of %d rows was written",
                    schema, table, type(exc).__name__, exc, len(normalized),
                )
                raise

            # WHICH ERRORS ARE WORTH ISOLATING (21 Sep 2026). Row-by-row retry
            # only makes sense when the failure is genuinely about SOME rows:
            # an IntegrityError (a foreign key to a player we haven't loaded, a
            # constraint one row violates) is per-row by definition, and
            # isolating it saves the rest of a good batch.
            #
            # Everything else is a property of the STATEMENT, not the data:
            # ProgrammingError ("can't adapt type 'NAType'", a column that
            # doesn't exist), OperationalError (connection gone), DataError (a
            # value too wide for its column). Retrying those row by row asks
            # the server the same broken question 26,000 times and gets the
            # same answer. Measured live: a NAType adaptation bug burned four
            # minutes and 1,307 round trips per chunk before the budget below
            # stopped it, and wrote 260 of 26,133 rows. Fail fast instead --
            # the loud error is the useful output, not the 1% that landed.
            if not isinstance(exc, psycopg2.IntegrityError):
                log.error(
                    "batch upsert into %s.%s failed with %s: %s -- NOT retrying row by row "
                    "(this is a statement/type problem, not a bad-row problem). "
                    "All %d rows in this batch were dropped.",
                    schema, table, type(exc).__name__, exc, len(normalized),
                )
                raise

            log.warning(
                "batch upsert into %s.%s failed (%s: %s) -- retrying %d rows one at a time to isolate the bad ones",
                schema, table, type(exc).__name__, exc, len(normalized),
            )
            # FALLBACK BUDGET (21 Sep 2026 -- added after this path took the
            # database offline). The row-by-row retry is the right tool for a
            # handful of bad rows and exactly the wrong one for a batch that's
            # bad end to end. Every SAVEPOINT opens a subtransaction, and
            # Postgres does NOT release a subtransaction's lock-table entries
            # on RELEASE -- it holds them until the OUTER transaction commits.
            # A 2025 pitch load hit ~800 FK failures in the first chunks and
            # opened tens of thousands of subtransactions inside one
            # season-long transaction, exhausting the shared lock table. The
            # server then rejected every connection, from this pipeline and
            # from anything else, with "out of shared memory". That's a
            # whole-database outage caused by an error-handling path.
            #
            # So: give up early instead. If the failures aren't a small
            # minority, the batch is structurally wrong (missing parent rows,
            # a schema mismatch) and retrying each row just multiplies the
            # damage while producing the same failure N times in the log.
            max_failures = max(25, len(normalized) // 20)  # 5% of the batch, floor of 25
            succeeded = 0
            failed = 0
            aborted = False
            for i, row in enumerate(normalized):
                cur.execute("SAVEPOINT upsert_row")
                try:
                    cur.execute(query, row)
                    cur.execute("RELEASE SAVEPOINT upsert_row")
                    succeeded += 1
                except psycopg2.Error as row_exc:
                    cur.execute("ROLLBACK TO SAVEPOINT upsert_row")
                    failed += 1
                    if failed <= 10:  # don't write the same error 40,000 times
                        key_values = {c: row.get(c) for c in conflict_cols}
                        log.warning(
                            "skipped 1 row in %s.%s (conflict key %s): %s",
                            schema, table, key_values, row_exc,
                        )
                    if failed > max_failures:
                        aborted = True
                        log.error(
                            "ABORTING row-by-row fallback for %s.%s: %d of the first %d rows failed "
                            "(budget %d). The batch is broken at the source, not row-by-row. "
                            "%d rows written, %d abandoned. Fix the caller rather than this batch.",
                            schema, table, failed, i + 1, max_failures,
                            succeeded, len(normalized) - (i + 1),
                        )
                        break
            level = log.error if aborted else log.info
            level(
                "upserted %d/%d rows into %s.%s (%d failed%s)",
                succeeded, len(rows), schema, table, failed,
                ", fallback aborted early" if aborted else "",
            )
            return succeeded

    log.info("upserted %d rows into %s.%s", len(rows), schema, table)
    return len(rows)


def replace_game_rows(
    conn,
    table: str,
    game_id: int,
    rows: Sequence[dict],
    conflict_cols: Sequence[str],
    schema: str = "mlb",
    keep_cols: Sequence[str] = (),
) -> tuple[int, int]:
    """Delete every row for one game, then insert `rows` in its place.

    Why this exists (24 Sep 2026): upsert_rows can add or overwrite rows but
    never remove one. The lineup table held end-of-game lineups; fixing the
    parser to return true starters would upsert the starters and leave every
    wrongly-stored substitute in place. Delete-and-reinsert per game is the
    only way a corrected set fully replaces the old one.

    Both statements run inside one savepoint, so a failure restores the
    game's old rows rather than leaving it empty. The insert is strict: any
    bad row raises instead of being skipped.

    keep_cols: hand-entered columns the pipeline never sets (the lineup
    table's playing_through_injury_flag). Their current values are read
    before the delete and carried onto the matching new row, so a rebuild
    can't wipe something Colin typed in by hand.

    Returns (rows deleted, rows inserted).
    """
    rows = [dict(r) for r in rows]
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT replace_game")
        try:
            if keep_cols:
                key_sql = ", ".join(c for c in conflict_cols if c != "game_id")
                keep_sql = ", ".join(keep_cols)
                cur.execute(
                    f"select {key_sql}, {keep_sql} from {schema}.{table} where game_id = %s",
                    (game_id,),
                )
                other_keys = [c for c in conflict_cols if c != "game_id"]
                kept = {}
                for rec in cur.fetchall():
                    key = tuple(rec[: len(other_keys)])
                    vals = dict(zip(keep_cols, rec[len(other_keys):]))
                    if any(v is not None for v in vals.values()):
                        kept[key] = vals
                for r in rows:
                    key = tuple(r.get(c) for c in other_keys)
                    for c, v in kept.get(key, {}).items():
                        if v is not None:
                            r[c] = v
            cur.execute(f"delete from {schema}.{table} where game_id = %s", (game_id,))
            deleted = cur.rowcount
            inserted = upsert_rows(conn, table, rows, conflict_cols=conflict_cols, schema=schema, strict=True)
            cur.execute("RELEASE SAVEPOINT replace_game")
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT replace_game")
            raise
    return deleted, inserted
