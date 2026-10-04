-- Minor-league season lines (model design group 8, rookie translations), 3 Oct 2026.
-- Adds only: one table. Nothing existing changes.
--
-- One row per player, season, stat group and level (sport id 11 = Triple-A,
-- 12 = Double-A). A player with several teams at one level in a season gets
-- the API's combined line when there is one, never the stints added up (the
-- same rule as player_season_stats). Filled by scripts/build_minor_seasons.py
-- for players in mlb.players only, 2017 on.

set search_path to mlb, public;

create table if not exists player_minor_season_stats (
    player_id          integer  not null references players(player_id),
    season             integer  not null,
    stat_group         text     not null check (stat_group in ('hitting', 'pitching')),
    sport_id           smallint not null check (sport_id in (11, 12)),
    num_teams          smallint not null default 1,
    age                smallint,
    games              integer,
    games_started      integer,
    plate_appearances  integer,
    at_bats            integer,
    batters_faced      integer,
    outs               integer,
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
    runs               integer,
    earned_runs        integer,
    number_of_pitches  integer,
    stat_json          jsonb    not null,
    pulled_at          timestamptz not null default now(),
    primary key (player_id, season, stat_group, sport_id)
);
