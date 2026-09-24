"""Offline tests for strict writes and delete-and-reinsert (24 Sep 2026).

No database: a fake cursor records every statement and raises an
IntegrityError for any row whose player_id is negative, the way a missing
foreign key would.

The failure being guarded against: the default upsert path isolates bad
rows, writes the rest, and returns normally, so a caller that ignores the
return value finishes "successfully" with rows missing.

Run: python tests/test_strict_writes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2  # noqa: E402

from pipelines import db  # noqa: E402

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")
        print(f"  FAIL {label}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {label} == {got!r}")


class FakeCursor:
    def __init__(self, store, existing=()):
        self.store = store
        self.rowcount = 0
        self._result = list(existing)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def mogrify(self, sql, params):
        # execute_batch mogrifies each row and executes them joined by ";".
        self.store.setdefault("pending", []).append(dict(params))
        return f"ROW:{len(self.store['pending']) - 1}".encode()

    def execute(self, sql, params=None):
        if isinstance(sql, bytes):  # a page from execute_batch
            self.store["log"].append("INSERT BATCH")
            batch = [self.store["pending"][int(x.split(b":")[1])] for x in sql.split(b";")]
            if any(r.get("player_id", 0) < 0 for r in batch):
                raise psycopg2.IntegrityError("fk violation")
            self.store["written"].extend(batch)
            return
        s = " ".join(sql.split())
        self.store["log"].append(s.split(" ")[0].upper() + (" " + s.split(" ")[1].upper() if " " in s else ""))
        if s.upper().startswith("INSERT"):
            if params and params.get("player_id", 0) < 0:
                raise psycopg2.IntegrityError("fk violation")
            self.store["written"].append(dict(params))
        elif s.lower().startswith("delete"):
            self.rowcount = self.store.get("delete_count", 0)
        elif s.lower().startswith("select"):
            pass

    def fetchall(self):
        return self._result


class FakeConn:
    def __init__(self, existing=()):
        self.store = {"log": [], "written": [], "delete_count": 3}
        self.existing = existing

    def cursor(self):
        return FakeCursor(self.store, self.existing)


GOOD = [{"game_id": 1, "team_id": 10, "player_id": p, "batting_order_slot": i + 1} for i, p in enumerate([101, 102, 103])]
BAD = GOOD[:2] + [{"game_id": 1, "team_id": 10, "player_id": -5, "batting_order_slot": 3}]
KEY = ["game_id", "team_id", "player_id"]

print("\ndefault mode: bad row skipped, call returns normally")
conn = FakeConn()
n = db.upsert_rows(conn, "lineup", [dict(r) for r in BAD], conflict_cols=KEY)
check("returns the 2 good rows", n, 2)

print("\nstrict mode: any bad row raises, nothing kept from the batch")
conn = FakeConn()
raised = False
try:
    db.upsert_rows(conn, "lineup", [dict(r) for r in BAD], conflict_cols=KEY, strict=True)
except psycopg2.IntegrityError:
    raised = True
check("strict raised", raised, True)
check("rolled back to the batch savepoint", "ROLLBACK TO" in conn.store["log"], True)
check("no row-by-row retry", conn.store["log"].count("SAVEPOINT UPSERT_ROW"), 0)

print("\nstrict mode: clean batch writes everything")
conn = FakeConn()
check("3 rows", db.upsert_rows(conn, "lineup", [dict(r) for r in GOOD], conflict_cols=KEY, strict=True), 3)

print("\nreplace_game_rows: delete then strict insert inside one savepoint")
conn = FakeConn()
deleted, inserted = db.replace_game_rows(conn, "lineup", 1, GOOD, KEY)
check("deleted count reported", deleted, 3)
check("inserted count", inserted, 3)
log = conn.store["log"]
check("delete happens before insert", log.index("DELETE FROM") < log.index("INSERT BATCH"), True)
check("savepoint released", "RELEASE SAVEPOINT" in log, True)

print("\nreplace_game_rows: a bad row restores the game's old rows")
conn = FakeConn()
raised = False
try:
    db.replace_game_rows(conn, "lineup", 1, BAD, KEY)
except psycopg2.IntegrityError:
    raised = True
check("raised", raised, True)
check("rolled back to replace_game savepoint", conn.store["log"][-1], "ROLLBACK TO")

print("\nreplace_game_rows: hand-set column carried across the rebuild")
conn = FakeConn(existing=[(10, 102, True)])  # (team_id, player_id, playing_through_injury_flag)
db.replace_game_rows(conn, "lineup", 1, GOOD, KEY, keep_cols=["playing_through_injury_flag"])
written = {r["player_id"]: r.get("playing_through_injury_flag") for r in conn.store["written"]}
check("flag kept for 102", written[102], True)
check("others untouched", written[101], None)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all strict write checks passed")
