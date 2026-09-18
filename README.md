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

## What the 18 Sep 2026 backfill run found (read this first)

The second full attempt ran 5h35m, got through all 2,511 games of postgame
work and 1,400 games of form tables, then hit GitHub's 6-hour job cap. Four
findings came out of it, all fixed in this commit:

**1. The form-table stage was quadratic, and could never have finished.**
Chunk times grew steadily -- ~9 minutes per 100 games early in the season,
~36 minutes per 100 games by game 1,400 -- on track for roughly 15 hours for
a single season. Cause: `bullpen_status.py` fetched a full box score for
every prior game in a team's lookback window, per team, per game, with no
cache, and the season-long window grows as the season goes on. That is on
the order of 400,000 HTTP fetches for one season, almost all of them
refetching a box score already fetched. It now caches each game's parsed
pitching lines by `game_id` (both teams at once), so it's at most ~2,500
fetches per season. Verified by stubbing the fetcher: 1,000 repeat calls
collapse to 1.

**2. Savant's OAA leaderboard was being called once per team per game**
(~5,000 times a season) when it only ever varies by as-of date (~190
distinct values). Now cached by `(year, through_date)`. It also went through
the non-raising wrapper, and `savant_client` now has a circuit breaker: after
3 failures against a URL it stops calling it for the rest of the run. That
run ate a 60-second read timeout against Savant; without the breaker, a
Savant outage would mean thousands of 60-second stalls.

**3. LOOKAHEAD BIAS in every API-sourced form number.** MLB's `byDateRange`
endpoint is inclusive of `endDate`, and both `starting_batter_form.py` and
`starting_pitcher_form.py` passed the game's own date as `endDate`. So
`woba_season`, `k_pct_*`, `bb_pct_*`, `mlb_pa_count` and the pitcher
equivalents all included the game being predicted -- a batter's 4-for-4 was
inside the wOBA a model would have used to predict that same game. The
Statcast/SQL-sourced columns were never affected (`sql_helpers` filters
`g.date < as_of_date`). Both modules now end their ranges the day before.
**Any form rows written before this commit are contaminated and should be
recomputed, not resumed onto.**

**4. Dodgers home games had no weather, all season.** MLB's `/venues`
endpoint now calls venue 22 "UNIQLO Field at Dodger Stadium" (sponsor
rename) while the schedule still reports the game's venue as "Dodger
Stadium", so the name-keyed coordinate lookup missed every game there.
`game_conditions.venue_name_keys` now also registers the part after " at ",
covering the whole "<Sponsor> Field at <Stadium>" pattern, and
`park_factors.py` uses the same aliases. The durable fix is to key parks by
venue id rather than name -- `games.venue` stores a name, so that's a schema
change and is not done here.

Also fixed: players who appear in a box score but were on no roster pull
(668904, 506702 in 2025) had their lineup rows silently dropped by the
foreign key, and triggered the row-by-row retry on nearly every chunk.
`reference/players.ensure_players_exist` now fetches and inserts them before
the lineup write, in both the backfill and the daily job.

**Resuming:** `scripts/backfill.py --resume` skips games that already have
form rows. It is off by default on purpose -- it is only safe when
continuing an interrupted run of the *same* code, which is not the case
across the lookahead fix above.

**Still open, not fixed here:** the form stage still makes ~60 MLB Stats API
calls per game (3 per batter, 3 per starting pitcher), which is roughly 3-4
hours per season on its own. Batching those via
`/people?personIds=...&hydrate=stats(...)` would cut it to a few calls per
game; the league-wide `byDateRange` leaderboard is NOT a substitute (checked
live -- it returns ~151 qualified players, not all ~900 batters, so bench
players would go missing). Worth doing before backfilling 2021-2024, both
for wall-clock and for GitHub Actions minutes.

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

**Confirmed broken, live (first real backfill run, 17 Sep 2026):**
`park_factor_hr` / `park_factor_runs` will come back null for every park,
every season, until this is fixed. Savant's statcast-park-factors
leaderboard page no longer returns CSV data for the `csv=true` trick this
pipeline (and pybaseball) uses everywhere else -- it now always serves the
full interactive HTML page instead, which broke `backfill.py` outright on
first run (`pandas.errors.ParserError`). `savant_client.get_park_factors`
now catches that failure and returns an empty result instead of crashing
(see `_read_savant_csv_optional`'s docstring), so the rest of the pipeline
runs fine, but the two HR/runs park-factor columns are simply unpopulated
until someone finds the current working endpoint for that specific
leaderboard (or a replacement source -- FanGraphs has park factors too,
via a different methodology, and is what at least one other Savant-scraper
project fell back to for this same gap; that'd be a real methodology
change worth deciding on deliberately, not defaulting into).
`field_orientation_degrees` on this table is unaffected -- that column
comes from the hand-researched CSV, not Savant, and still populates
normally.

**Also confirmed live, same run:** MLB's `rosterType: fullSeason` (used to
seed the `players` table from each team's roster) does not reliably return
every player who shows up elsewhere in a season's data -- a real,
currently-active starter (not some replacement-level fringe case) was
missing from every team's roster pull, which broke `games.home_starter_id`'s
foreign key and rolled back the entire games insert. Root cause
unconfirmed (a mid-season trade/DFA edge in how the Stats API scopes
"fullSeason" is the leading guess). Fixed two ways, both in this commit:
1. `backfill.py` and `daily_pull.py` now pull the schedule first, harvest
   every probable-starter `player_id` out of it, and pass those into
   `collect_player_ids_for_season` (`pipelines/reference/players.py`) so
   they're seeded into `players` before anything references them --
   fixing the root cause for that one column, not just papering over it.
2. `db.upsert_rows` (the one function every table write in this repo goes
   through) now falls back to a row-by-row retry, each in its own SQL
   savepoint, if a batch insert fails -- so one bad foreign key (an
   umpire, a lineup player, whatever) logs a warning and gets skipped
   instead of rolling back potentially a whole season's worth of work in
   one shared transaction. `game_results.update_game_umpire` (the one
   write that goes around `upsert_rows` entirely) got the same
   savepoint treatment directly. This is a safety net, not a fix for the
   underlying roster-completeness gap -- if it fires, the warning log
   says exactly which row and column, which is the signal to go add that
   case the same way #1 does.

**Confirmed live, same day, bigger problem:** the fixes above let a real
backfill run get further, and it then ran for 4+ hours and was still under
half done on a single season before Colin cancelled it. Root cause:
`backfill_postgame` and `backfill_form_tables` (`scripts/backfill.py`) were
writing one row at a time -- a separate `upsert_rows` call, i.e. a separate
network round trip to Supabase, per team/pitcher/batter *per game* --
inside one shared transaction that never committed until the entire
season's postgame-and-form-tables stage finished. `starting_batter_form`
alone is 15-20+ rows per game, so a full season is tens of thousands of
individual round trips. Worse: because nothing committed until the very
end, a killed run (GitHub's 6-hour job cap, a dropped connection, anything)
would have lost ALL of that work, not just whatever was mid-flight.

Fixed by batching: both functions now accumulate rows per table across
`COMMIT_EVERY_N_GAMES` (100) games, upsert each table once per chunk, and
commit at that checkpoint. An interruption now only costs a re-run of the
last partial chunk -- cheap and safe, since every write here is an
idempotent upsert. This does NOT address the read side -- `build_team_form_row`,
`build_bullpen_status_row`, `build_starting_pitcher_form_row`, and
`build_starting_batter_form_row` all still run their own SQL queries one
player at a time per game, same as before. If a rerun is still
unexpectedly slow, that's the next thing to profile (and `bullpen_status.py`'s
own known live-feed-refetching issue, noted below, is a specific known
contributor to it). `daily_pull.py` was left as-is -- it only ever touches
1-2 days of games at a time, so this same pattern there is a few hundred
rows at most, not a real problem.

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
  (`team_form.py` already catches any failure from this call per-game and
  leaves `def_oaa_season` null rather than crashing, so if this leaderboard
  has also stopped honoring `csv=true` the backfill will still complete --
  just watch the Action log for repeated "failed to pull team OAA"
  warnings, which would tell you that's the case.)
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
