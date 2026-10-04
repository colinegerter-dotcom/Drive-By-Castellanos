-- Minor-league team season totals (model design E19, rookie translations), 4 Oct 2026.
-- Adds only: one table. Nothing existing changes.
--
-- One row per season, level (sport id 11 = Triple-A, 12 = Double-A), stat
-- group and team: every team at the level, not only players who reached MLB.
-- Summed over teams it gives each level's league average for a season, so a
-- rookie's minor-league rates are compared with the whole league, not with
-- the players who later reached MLB (the E19 pre-build review found that
-- average depends on future call-ups). Filled by scripts/build_minor_seasons.py.

set search_path to mlb, public;

create table if not exists minor_league_team_totals (
    season             integer  not null,
    sport_id           smallint not null check (sport_id in (11, 12)),
    stat_group         text     not null check (stat_group in ('hitting', 'pitching')),
    team_id            integer  not null,
    team_name          text,
    games              integer,
    plate_appearances  integer,
    at_bats            integer,
    batters_faced      integer,
    hits               integer,
    doubles            integer,
    triples            integer,
    home_runs          integer,
    walks              integer,
    intentional_walks  integer,
    hit_by_pitch       integer,
    strikeouts         integer,
    sac_flies          integer,
    sac_bunts          integer,
    ground_outs        integer,
    air_outs           integer,
    stat_json          jsonb    not null,
    pulled_at          timestamptz not null default now(),
    primary key (season, sport_id, stat_group, team_id)
);
