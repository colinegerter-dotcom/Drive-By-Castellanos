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
) -> int:
    """Idempotent bulk upsert: INSERT ... ON CONFLICT (conflict_cols) DO UPDATE.

    Why this shape and not "delete then re-insert": a delete+insert would
    momentarily remove rows a live query might read, and would defeat
    foreign keys pointing at this table. ON CONFLICT DO UPDATE is atomic
    and safe to re-run every day without creating duplicates -- which is
    the "daily pulls need idempotent upserts" rule from the schema doc.

    Every column that appears on ANY row in the batch is treated as a
    column to write; a row missing a given key gets NULL for that column.
    This keeps callers simple (they just build a dict per row) at the cost
    of assuming rows are reasonably uniform in shape, which holds for
    everything in this pipeline.

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
        psycopg2.extras.execute_batch(cur, query, normalized, page_size=500)

    log.info("upserted %d rows into %s.%s", len(rows), schema, table)
    return len(rows)
