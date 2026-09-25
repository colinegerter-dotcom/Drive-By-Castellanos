#!/usr/bin/env python3
"""
Dated snapshots of the database (model design item A7, 25 Sep 2026).

Why: training runs must read a fixed copy of the data, not live tables that
the nightly job keeps changing, and a bad write must always be undoable.
Supabase's free tier has no backups of its own.

    python scripts/db_snapshot.py export  --out snap/          # every mlb table -> Parquet + manifest.json
    python scripts/db_snapshot.py verify  --dir snap/          # checksums and row counts vs the manifest
    python scripts/db_snapshot.py compare --dir snap/ --dsn postgresql://...   # a restored database vs the manifest

The workflow (.github/workflows/db-snapshot.yml) exports, packs and
ENCRYPTS the snapshot (the repo is public, and so are its release files),
decrypts a copy and verifies it, restores a pg_dump backup into a scratch
Postgres and compares it too, and only then uploads everything to a new
release named db-snapshot-YYYYMMDD-HHMM.

CONSISTENCY: every table is read inside one REPEATABLE READ, read-only
transaction, so all tables come from the same moment even if a write lands
mid-export.

TYPES: Parquet keeps integers, text, dates, timestamps, booleans and integer
arrays exactly. Postgres numeric columns (rates, ERA, odds probabilities)
become 64-bit floats, which is exact enough for modeling; the pg_dump backup
is the byte-exact copy for restoring the database itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("db_snapshot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SCHEMA = "mlb"
FETCH = 50_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arrow_type(pg_type: str, pa):
    """Postgres type name (from information_schema) -> Arrow type."""
    t = pg_type.lower()
    return {
        "smallint": pa.int16(), "integer": pa.int32(), "bigint": pa.int64(),
        "numeric": pa.float64(), "real": pa.float32(), "double precision": pa.float64(),
        "boolean": pa.bool_(), "date": pa.date32(),
        "timestamp with time zone": pa.timestamp("us", tz="UTC"),
        "timestamp without time zone": pa.timestamp("us"),
        "jsonb": pa.string(), "json": pa.string(), "text": pa.string(),
        "character varying": pa.string(),
    }.get(t)


def table_columns(cur, table: str) -> list[tuple[str, str, str]]:
    cur.execute(
        """
        select column_name, data_type, udt_name
        from information_schema.columns
        where table_schema = %s and table_name = %s
        order by ordinal_position
        """,
        (SCHEMA, table),
    )
    return cur.fetchall()


def export(out: Path) -> dict:
    import psycopg2
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dotenv import load_dotenv

    from pipelines.config import db_dsn

    load_dotenv()
    out.mkdir(parents=True, exist_ok=True)
    conn = psycopg2.connect(db_dsn())
    conn.set_session(isolation_level="REPEATABLE READ", readonly=True)
    manifest = {"schema": SCHEMA, "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": os.environ.get("GITHUB_SHA"), "tables": {}}
    try:
        with conn.cursor() as cur:
            cur.execute("select version()")
            manifest["server"] = cur.fetchone()[0]
            cur.execute(
                "select table_name from information_schema.tables "
                "where table_schema = %s and table_type = 'BASE TABLE' order by table_name",
                (SCHEMA,),
            )
            tables = [r[0] for r in cur.fetchall()]
        for table in tables:
            with conn.cursor() as cur:
                cols = table_columns(cur, table)
            fields = []
            for name, data_type, udt in cols:
                if data_type == "ARRAY":
                    inner = arrow_type({"_int2": "smallint", "_int4": "integer", "_int8": "bigint",
                                        "_text": "text"}.get(udt, "text"), pa)
                    fields.append(pa.field(name, pa.list_(inner)))
                else:
                    at = arrow_type(data_type, pa)
                    if at is None:
                        raise RuntimeError(f"{table}.{name}: no Parquet mapping for type {data_type}")
                    fields.append(pa.field(name, at))
            schema = pa.schema(fields)
            json_cols = {i for i, (_, dt, _) in enumerate(cols) if dt in ("json", "jsonb")}
            path = out / f"{table}.parquet"
            n = 0
            with conn.cursor(name=f"snap_{table}") as cur, pq.ParquetWriter(path, schema, compression="zstd") as w:
                cur.itersize = FETCH
                cur.execute(f'select * from {SCHEMA}."{table}"')
                while True:
                    rows = cur.fetchmany(FETCH)
                    if not rows:
                        break
                    columns = list(zip(*rows))
                    arrays = []
                    for i, f in enumerate(schema):
                        vals = columns[i]
                        if i in json_cols:
                            vals = [None if v is None else json.dumps(v, sort_keys=True) for v in vals]
                        elif pa.types.is_floating(f.type):
                            vals = [None if v is None else float(v) for v in vals]
                        arrays.append(pa.array(vals, type=f.type))
                    w.write_table(pa.Table.from_arrays(arrays, schema=schema))
                    n += len(rows)
            with conn.cursor() as cur:
                cur.execute(f'select count(*) from {SCHEMA}."{table}"')
                live = cur.fetchone()[0]
            if live != n:
                raise RuntimeError(f"{table}: exported {n} rows but the table has {live}")
            manifest["tables"][table] = {"rows": n, "file": path.name, "sha256": sha256(path),
                                         "bytes": path.stat().st_size,
                                         "columns": [c[0] for c in cols]}
            log.info("%-24s %9d rows  %6.1f MB", table, n, path.stat().st_size / 1e6)
        conn.rollback()  # read-only; nothing to commit
    finally:
        conn.close()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(t["rows"] for t in manifest["tables"].values())
    log.info("snapshot: %d tables, %d rows", len(manifest["tables"]), total)
    return manifest


def verify(snap: Path) -> None:
    """Every file present, checksum matches, row count matches the manifest."""
    import duckdb

    manifest = json.loads((snap / "manifest.json").read_text())
    problems = []
    con = duckdb.connect()
    for table, meta in manifest["tables"].items():
        path = snap / meta["file"]
        if not path.exists():
            problems.append(f"{table}: file missing")
            continue
        if sha256(path) != meta["sha256"]:
            problems.append(f"{table}: checksum mismatch")
        n = con.execute(f"select count(*) from read_parquet('{path.as_posix()}')").fetchone()[0]
        if n != meta["rows"]:
            problems.append(f"{table}: {n} rows in the file, manifest says {meta['rows']}")
    if problems:
        raise SystemExit("snapshot verification FAILED:\n  " + "\n  ".join(problems))
    log.info("verified %d tables against the manifest (checksums and row counts)", len(manifest["tables"]))


def compare(snap: Path, dsn: str) -> None:
    """Row counts in a database restored from the pg_dump backup vs the manifest."""
    import psycopg2

    manifest = json.loads((snap / "manifest.json").read_text())
    problems = []
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        for table, meta in manifest["tables"].items():
            cur.execute(f'select count(*) from {SCHEMA}."{table}"')
            n = cur.fetchone()[0]
            if n != meta["rows"]:
                problems.append(f"{table}: restored {n} rows, snapshot has {meta['rows']}")
    if problems:
        raise SystemExit("restore check FAILED:\n  " + "\n  ".join(problems))
    log.info("restored database matches the snapshot on all %d tables", len(manifest["tables"]))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--out", required=True, type=Path)
    v = sub.add_parser("verify"); v.add_argument("--dir", required=True, type=Path)
    c = sub.add_parser("compare"); c.add_argument("--dir", required=True, type=Path); c.add_argument("--dsn", required=True)
    a = p.parse_args()
    if a.cmd == "export":
        export(a.out)
    elif a.cmd == "verify":
        verify(a.dir)
    else:
        compare(a.dir, a.dsn)


if __name__ == "__main__":
    main()
