# There's a Drive From Castellanos -- data pipeline

Data pipeline for a data-driven MLB prediction model (first-five-inning lines,
run line, totals, moneyline on nationally televised games). This repo covers
the free-source tables only: MLB Stats API, Baseball Savant/Statcast (via
`pybaseball`), and Open-Meteo for weather. Odds tables and everything
downstream of them (`model_predictions`, `bets_placed`, `bankroll_log`,
`model_registry`) are a separate, already-decided build -- not here.

Full field-level schema: see the project's `claude/mlb-model-schema.md` doc.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in the real Supabase password
```

**Windows on ARM (Surface, etc.):** `psycopg2-binary` (and `psycopg`, its
successor) publish no precompiled wheel for `win_arm64` -- confirmed
directly against PyPI, not just a guess. `pip install` will try to compile
it from source instead and fail with `pg_config not found`. Fix: install
the 64-bit (x64) build of Python from python.org or via
`winget install Python.Python.3.12 --architecture x64 --force` (Windows'
ARM64 x64-emulation runs it fine), then build your venv from that x64
`python.exe` specifically rather than whatever ARM64-native Python `python`
resolves to by default. `where.exe python` will show every install on
PATH if you're not sure which is which.

**Running the scripts from the repo root** (`python scripts/backfill.py
...`, `python scripts/daily_pull.py`, exactly as documented below and in
the GitHub Actions workflow) works as-is -- both scripts insert the repo
root onto `sys.path` themselves at the top of the file, so `import
pipelines...` resolves correctly no matter where `python <script>` puts
`scripts/` on the path. If you ever see `ModuleNotFoundError: No module
named 'pipelines'`, it means you're running a stale copy of one of these
files from before that fix.

Database: Supabase project `drive-from-castellanos` (org: Colin's Personal
Org, region us-east-2), created dedicated to this pipeline rather than
reusing Colin's Workbench project -- see the schema migration at
`sql/migrations/0001_init.sql` (already applied to that project) for why.

No API keys needed anywhere -- MLB Stats API, Baseball Savant, and
Open-Meteo are all free and unauthenticated.

## Running it

```bash
# One-time historical backfill (2021-2025, per the schema doc's "4-5 completed
# seasons" depth requirement):
python scripts/backfill.py --seasons 2021 2022 2023 2024 2025

# Debugging a single season without the slow Statcast pull:
python scripts/backfill.py --seasons 2025 --skip-pitches

# Ongoing daily incremental pull (this is what .github/workflows/daily-pull.yml
# runs on a schedule -- see that file for why GitHub Actions rather than an
# ad hoc script or Claude's own scheduled-task tooling):
python scripts/daily_pull.py
```

For the GitHub Actions workflow to actually run, add these repo secrets
(Settings -> Secrets and variables -> Actions): `SUPABASE_DB_HOST`,
`SUPABASE_DB_PORT`, `SUPABASE_DB_NAME`, `SUPABASE_DB_USER`,
`SUPABASE_DB_PASSWORD` -- from this project's Supabase connection string.

## Repo layout

```
pipelines/
  config.py            timezone, season list, national-TV network list
  db.py                connection + the one shared idempotent-upsert helper
  mlb_stats_client.py  MLB Stats API (schedule, box score, roster, people, venues)
  savant_client.py     Baseball Savant (Statcast pulls, team OAA, park factors)
  reference/           players, teams, park_factors (+ the orientation CSV)
  games/               games, game_results, lineup, game_conditions, team_form, bullpen_status
  player_form/         starting_pitcher_form, starting_batter_form, umpire_stats
  pitches/             raw Statcast ingestion, date-chunked
scripts/
  backfill.py          historical backfill, season by season
  daily_pull.py         daily incremental job (see its docstring for step ordering)
sql/migrations/0001_init.sql   the 13 in-scope tables, applied to Supabase already
.github/workflows/daily-pull.yml
```

## Design notes worth knowing before touching this

**No-lookahead bias** is enforced mechanically, not by convention: every
rolling/season-stat query in `pipelines/games/team_form.py`,
`pipelines/games/bullpen_status.py`, and everything in `pipelines/player_form/`
filters strictly on `date < as_of_date` in SQL. Because that filter is
airtight regardless of what else is in the table, `backfill.py` loads a
whole season's raw data (games, pitches, game_results) up front and computes
the derived form tables in a second pass -- see that script's module
docstring for why this ordering is safe rather than a shortcut.

**Idempotent upserts**: every write goes through `pipelines/db.py`'s
`upsert_rows()`, one shared `INSERT ... ON CONFLICT DO UPDATE` builder. This
is also why `lineup.playing_through_injury_flag` (the one manual field in
the schema) is safe from being silently overwritten by a re-pull -- see that
module's docstring.

**Timestamps**: every timestamp column is Postgres `timestamptz`. That's
the correct implementation of "timezone-aware, storing the UTC offset, no
DST ambiguity" -- see `pipelines/config.py`'s docstring for why storing a
literal "America/Chicago" string per row isn't necessary or better.

## Known gaps and approximations (read before trusting a number)

This was built and reviewed for logical correctness, but **could not be run
against the live internet from the environment that built it** -- this
cloud sandbox's network is locked to package registries only (no
`statsapi.mlb.com`, `baseballsavant.mlb.com`, or `open-meteo.com`), and the
one attempt to fall back to a live-internet-connected machine hit a dropped
bridge connection. Nothing in this repo has been run end-to-end against
real data yet. Treat the first real backfill run as a validation pass, not
a formality -- check row counts, spot-check a handful of games by hand
against mlb.com, and watch the logs for the warnings this code emits on
purpose (see below).

**Park orientation -- researched, with caveats:**
`reference/park_orientation.csv` is filled in for every park. No clean
machine-readable source exists for this anywhere (tried Baseball Almanac's
AL/NL orientation pages and a couple of others -- all diagrams, not
tables), so every value was estimated visually from Google Maps satellite
imagery: navigate to the park's coordinates in satellite view, identify
home plate and the deepest point of the outfield fence, estimate the
compass bearing between them. That's good to roughly +/-15-20 degrees,
which is plenty for the 4-bucket, 45-degree-wide `wind_effect`
classification it feeds -- but it is a visual estimate, not a surveyed
figure, and a couple of parks came out at bearings that are unusual enough
relative to the rest of the league (Comerica Park, Rate Field/Guaranteed
Rate Field) that they're flagged in the CSV's notes column as worth a
second-source spot-check before fully trusting them. See that file's notes
column for a per-park confidence note.

Six parks stayed unresolved: Chase Field, Daikin Park, Globe Life Field,
loanDepot park, Rogers Centre, and Tropicana Field all had their roof
closed in the available satellite imagery, so the field itself isn't
visible and there's nothing to estimate a bearing from. (Two other roofed
parks, American Family Field and T-Mobile Park, do have a value -- roof
was open, or partially inferable, in their imagery -- but see the next
paragraph for why that value doesn't actually reach the model.)

**Roofed parks: wind/orientation forced to neutral, on purpose.** Decided
with Colin: a closed roof makes wind and park-orientation effects moot, and
there's no per-game feed telling us whether a retractable roof was
actually open or closed on a given day, so guessing is worse than just
turning it off. `config.ROOFED_PARKS` lists the eight venues with a roof
(retractable or fixed). `reference/park_factors.py` nulls out
`field_orientation_degrees` for all of them regardless of what's in the
CSV, and `pipelines/games/game_conditions.py` forces `wind_effect` to
`"neutral"` for every game at one of those venues. `temp_f`/`humidity`/
`precip_flag` still populate from Open-Meteo for these games (ambient
outside-the-stadium context), but treat those as weak signal too -- they
don't reflect the climate-controlled conditions actually played in.

**Needs live-API verification (couldn't confirm the exact response shape
without network access):**
- `games.national_tv_flag` classification (`pipelines/games/games.py`,
  `_is_national_broadcast`) -- checks a couple of plausible field shapes on
  MLB's broadcast objects, falls back to a curated network-name list in
  `config.py`. Spot-check a known national game once live.
- `savant_client.get_team_outs_above_average`'s `through_date` param --
  unverified whether Savant's leaderboard actually honors a date-bounded
  query in "Team" mode. If it doesn't, `team_form.def_oaa_season` is
  silently season-end data reused on every game, which **would be a
  lookahead-bias violation** -- verify this before trusting that one column.
- Savant CSV column names in `savant_client.py` / `park_factors.py` --
  Savant has changed these before; code picks from a candidate list and
  logs a warning if nothing matches.

**Not yet built (explicitly, not silently skipped -- these come back
`None`/`null`):**
- `starting_pitcher_form.fip` / `.xfip` (and the `_season` variants),
  `.spin_rate_percentile` -- FIP needs a league-wide constant computable
  from data we already pull; xFIP and the percentile need a league-wide
  aggregation step. Straightforward follow-up, just not wired up yet.
- `starting_batter_form.vs_pitcher_hand_split` -- needs the opposing
  starter's throwing hand as a per-game input, which isn't threaded through
  the current call signature.
- `*.days_since_trade` (both pitcher and batter form) -- no transactions
  table is in scope for this build; the schema doc doesn't define one.
- `team_form.travel_fatigue_score` -- needs schedule/venue-distance logic,
  not built.
- Live/probable lineups for upcoming games -- `lineup.py` currently only
  reads the box score, which is empty pregame. `daily_pull.py` calls it
  anyway for "today," which will just return zero rows until a real
  probable-lineup puller is added (flagged in that script's inline comment).

**Approximations, by design (documented in the relevant module, repeated
here for visibility):**
- `starting_batter_form.barrel_pct_*` uses the common simplified barrel
  definition (EV >= 98mph and 26-30 degree launch angle), not Statcast's
  full EV/LA matrix.
- `starting_batter_form.woba_*` uses stable ~2023-2025 FanGraphs linear
  weights rather than a per-season lookup table.
- `umpire_stats` strike-zone edge uses the commonly-used +/-0.83ft
  approximation of the plate's edge (accounts for ball radius), not an
  exact rulebook geometry calculation.
- `bullpen_status.closer_available_flag` proxies "the closer" as whichever
  reliever has the team's most saves so far this season -- noisy for
  committee-closer teams or early in a season.

**Performance, not correctness:** `bullpen_status.py` re-fetches a game's
full box score (`get_live_feed`) once per lookback game per team per game
being scored -- during a 5-season backfill the same game gets refetched
many times. Fine for a first backfill; worth adding a simple cache of
live-feed payloads by `game_id` before running this repeatedly.

## Supabase note

Free tier caps an org at 2 active projects. Colin already had 2
(`goated-fitness`, `Colins Workbench`) when this project was created, so
`goated-fitness` was paused (not deleted -- reversible any time) to make
room. Unpause it in the Supabase dashboard if you need it back; you'll need
to pause or delete something else first to create another new free project
while this one and Workbench are both active.
