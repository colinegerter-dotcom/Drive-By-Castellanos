"""
Pitch-level storage on Parquet, queried with DuckDB.

WHY THIS EXISTS (21 Sep 2026)
-----------------------------
`mlb.pitches` is the only table in this pipeline with a genuinely different
shape from the rest: ~730k rows per season, append-only, never looked up by
key, and only ever scanned and aggregated. Keeping it in Supabase Postgres
cost about 200 MB of a 500 MB free tier for ONE season, which made a
multi-season dataset impossible on that plan.

The bigger problem was compute, not storage. The form tables ask roughly
101,000 separate questions of the pitch data per season (2 per batter-game,
1 per pitcher-game, 3 per game for umpires). Against Postgres each of those
is a network round trip that ships raw pitch rows to the runner so Python
can count them -- an estimated ~9 GB of egress against a 5 GB monthly
allowance, plus enough latency to push a 4h10m job toward the 6-hour
GitHub Actions cap.

Parquet plus DuckDB inverts that. One ~20 MB file per season is downloaded
once, then every aggregation runs in-process at memory speed with zero
egress. The raw pitch-by-pitch rows are fully preserved -- nothing is
pre-aggregated or discarded -- which is the point: pitch type, sequencing,
batter-vs-pitch-type and platoon splits all still need the individual rows.

WHAT STAYS IN POSTGRES
----------------------
Everything else. games, players, teams, the form tables, results and
reference data keep their foreign keys and stay the system of record. Only
pitches move. `games` is copied into DuckDB per season (a few thousand rows,
one small query) so the date/season/umpire joins can happen locally.

STORAGE LAYOUT
--------------
    data/pitches/season=<year>/chunk-<nnnn>.parquet   during a backfill
    data/pitches_<year>.parquet                       consolidated, uploaded

Chunk files exist so a backfill never has to hold a full season in memory
and so an interrupted run keeps its completed weeks. The consolidated file
is what gets attached to a GitHub Release; the workflow does that with the
`gh` CLI so no storage credentials ever enter this code.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path("data")

# Column order for pitch Parquet files. Fixed explicitly rather than inferred
# from whatever dict happens to be passed, so that chunk files written at
# different times in a run are always mergeable and so a schema change is a
# visible edit here rather than a silent drift between files.
PITCH_COLUMNS = [
    "game_id", "at_bat_id", "pitch_number",
    "pitcher_id", "batter_id",
    "inning", "balls", "strikes",
    "pitch_type", "release_speed", "spin_rate",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "pitch_result", "exit_velocity", "launch_angle",
    "events", "bb_type", "hit_location",
]


def season_dir(season: int, root: Path = DATA_DIR) -> Path:
    return root / "pitches" / f"season={season}"


def season_file(season: int, root: Path = DATA_DIR) -> Path:
    return root / f"pitches_{season}.parquet"


def write_chunk(rows: list[dict], season: int, chunk_index: int, root: Path = DATA_DIR) -> Path | None:
    """Write one backfill chunk's pitch rows to its own Parquet file.

    Returns the path written, or None for an empty chunk (March weeks that
    are entirely spring training produce no rows at all, and writing empty
    files would just confuse the consolidate step).
    """
    if not rows:
        return None

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = season_dir(season, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chunk-{chunk_index:04d}.parquet"

    columns = {c: [r.get(c) for r in rows] for c in PITCH_COLUMNS}
    table = pa.table(columns)
    # zstd over the default snappy: measurably smaller on this data (lots of
    # repeated short strings like pitch_result and events) and DuckDB reads
    # it natively with no extra dependency.
    pq.write_table(table, path, compression="zstd")
    log.info("wrote %d pitch rows to %s", len(rows), path)
    return path


def consolidate_season(season: int, root: Path = DATA_DIR) -> Path | None:
    """Merge a season's chunk files into one Parquet file for upload.

    One file per season rather than a directory because GitHub Releases
    attaches individual assets, and because a single file is far easier to
    hand to DuckDB (or to a future notebook) than a directory of fragments.
    """
    import duckdb

    src = season_dir(season, root)
    chunks = sorted(src.glob("chunk-*.parquet"))
    if not chunks:
        log.warning("no pitch chunks found in %s -- nothing to consolidate", src)
        return None

    out = season_file(season, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(
            "COPY (SELECT * FROM read_parquet($glob) ORDER BY game_id, at_bat_id, pitch_number) "
            "TO $out (FORMAT PARQUET, COMPRESSION ZSTD)",
            {"glob": str(src / "chunk-*.parquet"), "out": str(out)},
        )
        # Sorting by the natural key before writing is not cosmetic: Parquet
        # keeps min/max statistics per row group, so a sorted file lets
        # DuckDB skip whole row groups when filtering by game_id. It costs
        # one sort at write time and pays back on every later read.
        n = con.execute("SELECT count(*) FROM read_parquet($p)", {"p": str(out)}).fetchone()[0]
    finally:
        con.close()

    size_mb = out.stat().st_size / (1024 * 1024)
    log.info("consolidated %d chunks -> %s (%d rows, %.1f MB)", len(chunks), out, n, size_mb)
    return out


class _DuckCursor:
    """Quacks like a psycopg2 cursor.

    The form modules were written against psycopg2 and do exactly three
    things: open a cursor, execute a query with a dict of named parameters,
    and fetchall() a list of tuples. Presenting that same surface here means
    starting_batter_form / starting_pitcher_form / umpire_stats need no
    changes to their query-running code at all -- only the handle they're
    given changes. Fewer edits to reviewed, working aggregation logic.
    """

    def __init__(self, con):
        self._con = con
        self._result = None

    def execute(self, query: str, params: dict | None = None):
        self._result = self._con.execute(query, params or {})
        return self

    def fetchall(self):
        return self._result.fetchall() if self._result is not None else []

    def fetchone(self):
        return self._result.fetchone() if self._result is not None else None


class PitchSource:
    """A read-only, DuckDB-backed stand-in for the Postgres connection, for
    pitch queries only.

    Holds two tables: `pitches` (from the season's Parquet file) and `games`
    (copied from Postgres). Both are needed because every lookahead guard in
    sql_helpers filters on `games.date`, and the umpire queries join on
    `games.umpire_id`.
    """

    def __init__(self, con):
        self._con = con

    @contextmanager
    def cursor(self):
        yield _DuckCursor(self._con)

    def query(self, sql: str, params: dict | None = None):
        """Direct access, for ad-hoc exploration rather than the form path."""
        return self._con.execute(sql, params or {})

    def close(self):
        self._con.close()


def open_pitch_source(season: int, game_rows: list[dict], root: Path = DATA_DIR) -> PitchSource:
    """Build an in-memory DuckDB holding one season of pitches plus its games.

    game_rows: the same dicts the schedule pull produces (game_id, date,
    season, umpire_id, ...). Passed in rather than read from Postgres here so
    this module stays free of any database dependency -- it can be pointed at
    a notebook, a test fixture, or the live pipeline without change.
    """
    import duckdb

    path = season_file(season, root)
    if not path.exists():
        raise FileNotFoundError(
            f"no pitch Parquet for {season} at {path}. Run the pitch backfill for "
            f"this season first, or have the workflow download the release asset."
        )

    con = duckdb.connect()
    # DuckDB can't bind a prepared parameter inside CREATE VIEW, so the path
    # is inlined. It's a path this code constructed from an integer season,
    # never user input, and the quote-doubling keeps it well-formed anyway.
    literal_path = str(path).replace("'", "''")
    con.execute(f"CREATE VIEW pitches AS SELECT * FROM read_parquet('{literal_path}')")

    # games as a real table, not a view: it's small, it's hit by every single
    # query, and materializing it once means DuckDB doesn't re-scan a Python
    # object per query.
    con.execute("""
        CREATE TABLE games (
            game_id BIGINT, date DATE, season INTEGER, umpire_id INTEGER
        )
    """)
    con.executemany(
        "INSERT INTO games VALUES (?, ?, ?, ?)",
        [
            (g["game_id"], g.get("date"), g.get("season"), g.get("umpire_id"))
            for g in game_rows
        ],
    )
    con.execute("CREATE INDEX idx_games_game_id ON games(game_id)")

    n_pitches = con.execute("SELECT count(*) FROM pitches").fetchone()[0]
    n_umps = con.execute("SELECT count(umpire_id) FROM games").fetchone()[0]
    log.info(
        "pitch source ready for %s: %d pitches from %s, %d games, %d with an umpire",
        season, n_pitches, path.name, len(game_rows), n_umps,
    )

    # Loud warning rather than a silent empty table. Every umpire query joins
    # on games.umpire_id, so if that column is all NULL the queries match
    # nothing, every row builds as None, and umpire_stats ends up empty with
    # no error raised anywhere -- which is exactly what happened on 21 Sep
    # 2026 when these rows came from the pre-game schedule instead of from
    # Postgres. Cheap check, and it names the fix.
    if game_rows and n_umps == 0:
        log.warning(
            "NO games have an umpire_id -- umpire_stats will come out EMPTY. "
            "These game rows almost certainly came from the pre-game schedule, "
            "which never includes the plate umpire. Read them from mlb.games "
            "instead (see games_for_pitch_source in scripts/backfill.py)."
        )

    return PitchSource(con)
