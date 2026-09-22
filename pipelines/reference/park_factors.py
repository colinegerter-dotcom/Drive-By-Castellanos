"""
park_factors table -- three fields, three very different provenances:

1. park_id: MLB Stats API venue id (stable identifier even when a park's
   sponsor name changes, unlike using the name itself as the key).

2. park_factor_runs: computed IN-HOUSE from our own games/game_results
   tables. This changed on 22 Sep 2026 -- it used to come from Baseball
   Savant's Statcast park factors leaderboard, which was verified dead that
   day (see savant_client.get_park_factors for the full evidence: the
   `csv=true` export returns HTML, and the HTML contains no park data at
   all -- no "Coors", no venue_id, no embedded JSON, nothing in the DOM).

   Computing it ourselves is the better answer regardless of whether Savant
   ever comes back:
   - no external dependency that can rot silently, which is exactly the
     failure that produced a null column for a whole season
   - we control the date bounds, so it cannot leak (see NO LOOKAHEAD below)
   - it covers every season we have data for, in one pass

3. field_orientation_degrees: NOT available from any free API -- static
   geographic data, filled in by hand once in reference/park_orientation.csv,
   keyed by venue NAME. Unchanged by this rewrite.

park_factor_hr is left None. Home-run counts are not in game_results; they
would have to come from the pitch-level Parquet, which is stored one file
per season and would mean opening prior seasons' files here. That is a
real piece of work, not a one-liner, and park_factor_runs is the field a
runs model actually needs. Deliberately deferred rather than half-built.

NO LOOKAHEAD -- the part that matters
-------------------------------------
A park factor for year Y is computed ONLY from seasons strictly before Y.
Using year Y's own games to build a park factor applied to year Y's games
means every row carries a feature computed partly from its own outcome.
That is mild as leaks go, but it is still a leak, and this repo's whole
premise is that the form tables don't have any.

The practical consequence: the EARLIEST season in the database gets a null
park factor, because there is nothing before it to compute from. If seasons
2021-2025 are backfilled, 2021 parks are null and 2022-2025 are populated.
That is correct, not a bug. If a 2021 factor is wanted later, pulling 2020
game results alone (no form build, no pitch pull -- cheap) would supply it.

METHOD
------
Classic "basic park factor", which controls for team quality:

    park_factor_runs = (runs per game at this park by its home team)
                     / (runs per game by that same team in its away games)

Both numbers use TOTAL runs (both teams combined), over a trailing window
of up to PARK_FACTOR_LOOKBACK_SEASONS prior seasons. Comparing a team's
home games against its own away games is what stops a good offense from
looking like a hitters' park. A raw ratio against the league average would
conflate the two.

Regular season only (game_type = 'R'). Postseason run environments differ
and postseason venues are wildly unbalanced, so a handful of October games
at one park would distort its factor.

Two guards against small-sample noise:
- Parks with fewer than PARK_FACTOR_MIN_HOME_GAMES in the window get None,
  not a number. A null is honest; a noisy number is worse than nothing
  because nothing downstream can tell it apart from a real measurement.
- Surviving factors are regressed toward 1.0 by sample size:
      regressed = 1 + (raw - 1) * games / (games + PARK_FACTOR_REGRESSION_K)
  With K = 162 (one team-season of games), a park with one season of data
  keeps about half its raw deviation and three seasons keeps about 60%.
  This is standard practice for park factors and it is the reason a single
  weird season can't hand a park a 1.35 factor. Tune K here if you want it
  more or less aggressive; it is deliberately a named constant.

Roofed parks (config.ROOFED_PARKS): field_orientation_degrees is forced to
None regardless of what's in the CSV, even for the couple of roofed parks
where the roof happened to be open in the imagery (T-Mobile Park) or a
value was estimated anyway (American Family Field). Decided with Colin --
orientation is physically moot once a field can be enclosed, and we have no
per-game feed telling us whether a retractable roof was actually open, so
nulling it here means nothing downstream (not just game_conditions.py's
wind_effect) can silently treat a roofed park's orientation as meaningful.
"""
from __future__ import annotations

import csv
import logging
import os

from pipelines.config import ROOFED_PARKS
from pipelines.games.game_conditions import venue_name_keys
from pipelines.mlb_stats_client import get_venue_names_by_season, get_venues

log = logging.getLogger(__name__)

_ORIENTATION_CSV = os.path.join(os.path.dirname(__file__), "park_orientation.csv")

# How many prior seasons feed a park factor. 3 balances "enough games to be
# stable" against "recent enough to reflect the park as it is now" -- parks
# get renovated, fences move, humidors get installed.
PARK_FACTOR_LOOKBACK_SEASONS = 3

# Below this many home games in the window, emit None rather than a number.
#
# 70, not 81, and the difference is not arbitrary. A full home schedule is
# 81 games, but checking the real 2025 data (22 Sep 2026) showed SIX actual
# MLB parks landing just under that line because each club gave up a home
# date to a neutral-site event: Yankee Stadium 80, Citi Field 80, Rate Field
# 80, Target Field 80, Wrigley Field 79, Great American Ball Park 79. An 81
# floor would have silently dropped a fifth of the league's parks.
#
# The genuine neutral sites we DO want excluded sit far below that -- Tokyo
# Dome 2, Bristol Motor Speedway 1, Journey Bank Ballpark 1 -- so there is a
# wide empty gap between 2 and 79 to put the threshold in. 70 sits in it with
# room on both sides, and still means "most of a season of home games".
PARK_FACTOR_MIN_HOME_GAMES = 70

# Regression-to-the-mean constant, in games. See METHOD in the module docstring.
PARK_FACTOR_REGRESSION_K = 162


def _load_orientation_by_name() -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    with open(_ORIENTATION_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = row["orientation_degrees_azimuth"].strip()
            out[row["venue_name"]] = float(val) if val else None
    return out


def lookback_seasons(year: int, available_seasons: set[int] | None = None) -> list[int]:
    """The seasons that may feed year `year`'s park factor.

    Strictly BEFORE `year` -- this is the no-lookahead guarantee, enforced
    here in one place rather than trusted to the SQL. If `available_seasons`
    is given, the window is intersected with it, so asking for 2022's factor
    when only 2021+ exists correctly yields just [2021] instead of silently
    querying for seasons that aren't there.
    """
    window = [year - n for n in range(1, PARK_FACTOR_LOOKBACK_SEASONS + 1)]
    if available_seasons is not None:
        window = [s for s in window if s in available_seasons]
    return sorted(window)


def fetch_run_aggregates(conn, seasons: list[int]) -> tuple[list[dict], dict[int, dict]]:
    """Total runs and game counts, grouped two ways, for the given seasons.

    Returns (home_rows, away_by_team):
      home_rows:    [{"venue": str, "home_team": int, "games": int, "runs": int}, ...]
      away_by_team: {team_id: {"games": int, "runs": int}}

    Both are small (roughly 30-40 rows and 30 entries), so this is two cheap
    aggregate queries rather than anything that moves real volume over the
    wire -- which matters, the Supabase free tier has a 5 GB monthly egress
    budget and this runs once per season backfilled.
    """
    if not seasons:
        return [], {}

    # game_type = 'R': regular season only, see METHOD in module docstring.
    filters = """
        where g.season = any(%(seasons)s)
          and g.game_type = 'R'
          and gr.game_status = 'completed'
          and gr.home_score_final is not null
          and gr.away_score_final is not null
    """

    home_sql = f"""
        select g.venue, g.home_team,
               count(*) as games,
               sum(gr.home_score_final + gr.away_score_final) as runs
        from mlb.games g
        join mlb.game_results gr on gr.game_id = g.game_id
        {filters}
        group by g.venue, g.home_team
    """
    away_sql = f"""
        select g.away_team,
               count(*) as games,
               sum(gr.home_score_final + gr.away_score_final) as runs
        from mlb.games g
        join mlb.game_results gr on gr.game_id = g.game_id
        {filters}
        group by g.away_team
    """

    with conn.cursor() as cur:
        cur.execute(home_sql, {"seasons": seasons})
        home_rows = [
            {"venue": r[0], "home_team": r[1], "games": int(r[2]), "runs": int(r[3])}
            for r in cur.fetchall()
            if r[0] is not None
        ]
        cur.execute(away_sql, {"seasons": seasons})
        away_by_team = {
            int(r[0]): {"games": int(r[1]), "runs": int(r[2])} for r in cur.fetchall()
        }

    return home_rows, away_by_team


def compute_park_factors(
    home_rows: list[dict],
    away_by_team: dict[int, dict],
    min_home_games: int = PARK_FACTOR_MIN_HOME_GAMES,
    regression_k: int = PARK_FACTOR_REGRESSION_K,
) -> dict[str, float]:
    """Venue name -> regressed runs park factor. Pure function, no DB, no network.

    Split out from fetch_run_aggregates specifically so the arithmetic can
    be tested offline against hand-built inputs (tests/test_park_factors.py)
    -- the part that can be silently wrong is the math, not the SQL.

    Venues that don't clear `min_home_games`, or whose home teams have no
    away games in the window, are OMITTED from the returned dict rather than
    given a placeholder. Callers turn a missing key into a null column.
    """
    by_venue: dict[str, dict] = {}
    for row in home_rows:
        v = by_venue.setdefault(row["venue"], {"games": 0, "runs": 0, "teams": set()})
        v["games"] += row["games"]
        v["runs"] += row["runs"]
        v["teams"].add(row["home_team"])

    out: dict[str, float] = {}
    for venue, agg in by_venue.items():
        home_games, home_runs = agg["games"], agg["runs"]
        if home_games < min_home_games:
            continue

        # The control group: the same clubs' road games. A team that scores
        # a lot everywhere raises both sides of the ratio and so cancels out,
        # which is the entire point of doing it this way.
        away_games = sum(away_by_team.get(t, {}).get("games", 0) for t in agg["teams"])
        away_runs = sum(away_by_team.get(t, {}).get("runs", 0) for t in agg["teams"])
        if away_games == 0 or away_runs == 0:
            continue

        home_rpg = home_runs / home_games
        away_rpg = away_runs / away_games
        raw = home_rpg / away_rpg

        # Shrink toward neutral by sample size -- see METHOD in the docstring.
        shrink = home_games / (home_games + regression_k)
        out[venue] = round(1.0 + (raw - 1.0) * shrink, 4)

    return out


def build_park_factor_rows(conn, year: int, available_seasons: set[int] | None = None) -> list[dict]:
    """Rows for mlb.park_factors for a single year.

    NOTE the signature changed on 22 Sep 2026: this now takes `conn` as its
    first argument, because park factors are computed from our own tables
    instead of fetched from Savant. scripts/backfill.py was updated to match.
    """
    venues = get_venues()
    orientation_by_name = _load_orientation_by_name()

    seasons = lookback_seasons(year, available_seasons)

    # Historical venue names for the lookback window. factor_by_venue below
    # is keyed by whatever mlb.games.venue holds for those seasons, which is
    # the name the park had THEN, while get_venues() returns the name it has
    # NOW. Without this a renamed park silently gets a null factor -- the
    # same failure that was skipping weather for ~162 games of 2021. Found
    # and fixed 22 Sep 2026; see mlb_stats_client.get_venue_names_by_season.
    historical_names_by_id: dict[int, set[str]] = {}
    for season in seasons:
        try:
            for venue_id, name in get_venue_names_by_season(season).items():
                historical_names_by_id.setdefault(venue_id, set()).add(name)
        except Exception:  # noqa: BLE001 -- an alias lookup must never break a backfill
            log.warning(
                "[%s] could not fetch %s venue names for historical aliases; parks "
                "renamed since then may get a null park factor",
                year,
                season,
                exc_info=True,
            )
    if not seasons:
        log.warning(
            "[%s] no prior seasons available, so park_factor_runs will be null for "
            "every park this year. Expected for the earliest season in the database; "
            "see park_factors.py module docstring.",
            year,
        )
        factor_by_venue: dict[str, float] = {}
    else:
        home_rows, away_by_team = fetch_run_aggregates(conn, seasons)
        factor_by_venue = compute_park_factors(home_rows, away_by_team)
        log.info(
            "[%s] park factors computed from seasons %s: %d venues with a factor, "
            "%d venue-groups seen",
            year,
            seasons,
            len(factor_by_venue),
            len(home_rows),
        )

    rows = []
    unmatched_orientation = []
    matched_factor = 0
    for v in venues:
        # Only active MLB venues -- the /venues endpoint returns a lot of
        # spring training / minor league facilities too, which we don't want.
        if not v.get("active", True):
            continue
        name = v.get("name")
        park_id = v.get("id")
        if park_id is None or name is None:
            continue

        # Try each alias for this venue, not just the current (possibly
        # sponsor-renamed) name -- see game_conditions.venue_name_keys. The
        # orientation CSV, ROOFED_PARKS and our own games.venue text are all
        # keyed by the everyday stadium name ("Dodger Stadium"), while
        # /venues may now return "UNIQLO Field at Dodger Stadium".
        # Current name first, then any name this same venue id carried in the
        # lookback seasons, so a rename doesn't cost the park its factor.
        name_keys = list(venue_name_keys(name))
        for historical in sorted(historical_names_by_id.get(park_id, set())):
            for k in venue_name_keys(historical):
                if k not in name_keys:
                    name_keys.append(k)

        factor_key = next((k for k in name_keys if k in factor_by_venue), None)
        park_factor_runs = factor_by_venue.get(factor_key) if factor_key else None
        if park_factor_runs is not None:
            matched_factor += 1

        matched_orientation_key = next((k for k in name_keys if k in orientation_by_name), None)
        is_roofed = any(k in ROOFED_PARKS for k in name_keys)
        orientation = None if is_roofed else orientation_by_name.get(matched_orientation_key)
        if matched_orientation_key is None:
            unmatched_orientation.append(name)

        rows.append(
            {
                "park_id": str(park_id),
                "year": year,
                # See module docstring: needs the pitch Parquet, deliberately deferred.
                "park_factor_hr": None,
                "park_factor_runs": park_factor_runs,
                "field_orientation_degrees": orientation,
            }
        )

    if seasons and matched_factor == 0:
        # Loud, because this is the shape of the bug that produced a null
        # column for a whole season: plenty of computed factors, but none of
        # them matched a venue, and nothing said so.
        log.warning(
            "[%s] computed %d park factors but matched NONE of them to an active venue "
            "-- venue naming has probably drifted between mlb.games.venue and the "
            "/venues endpoint. Computed venue names: %s",
            year,
            len(factor_by_venue),
            sorted(factor_by_venue)[:10],
        )

    if unmatched_orientation:
        log.warning(
            "%d venues have no row in park_orientation.csv (add them): %s",
            len(unmatched_orientation),
            unmatched_orientation,
        )
    return rows
