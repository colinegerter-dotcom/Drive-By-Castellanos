-- MLB prediction model schema — in-scope tables only for this build
-- (odds/model/tracking tables are a separate, already-decided build and excluded here)
-- All timestamps: timestamptz (UTC-normalized, timezone-aware, safely renderable in
-- America/Chicago without DST fall-back ambiguity). Every automated table carries pulled_at.
--
-- Already applied to Supabase project "drive-from-castellanos" (fqokvprujriyldqgzozy,
-- us-east-2) via the Supabase MCP tool. This file is the repo's copy of record --
-- if you ever need to recreate the schema from scratch (a new environment, a local
-- Postgres for testing), run this file directly.

create schema if not exists mlb;
set search_path to mlb, public;

-- ============ Reference & lookup ============

create table if not exists teams (
    team_id           integer primary key,
    team_name         text not null,
    league            text,
    division          text,
    pulled_at         timestamptz not null default now()
);

create table if not exists players (
    player_id         integer primary key,
    full_name         text not null,
    primary_position  text,
    bats              text,
    throws            text,
    debut_date        date,
    current_team_id   integer references teams(team_id),
    pulled_at         timestamptz not null default now()
);
create index if not exists idx_players_current_team on players(current_team_id);

-- ============ Game & situational context ============

create table if not exists games (
    game_id             bigint primary key,
    date                date not null,
    season              integer not null,
    home_team           integer not null references teams(team_id),
    away_team           integer not null references teams(team_id),
    home_starter_id     integer references players(player_id),
    away_starter_id     integer references players(player_id),
    first_pitch_time    timestamptz,
    venue               text,
    day_night           text check (day_night in ('day','night')),
    doubleheader_flag   boolean not null default false,
    national_tv_flag    boolean not null default false,
    umpire_id           integer,
    game_type           text,
    pulled_at           timestamptz not null default now()
);
create index if not exists idx_games_date on games(date);
create index if not exists idx_games_season on games(season);

create table if not exists game_results (
    game_id                 bigint primary key references games(game_id),
    home_score_final        integer,
    away_score_final        integer,
    home_score_f5           integer,
    away_score_f5           integer,
    winning_team             integer references teams(team_id),
    innings_played           integer,
    game_status              text check (game_status in ('completed','postponed','suspended','forfeit')),
    actual_home_starter_id   integer references players(player_id),
    actual_away_starter_id   integer references players(player_id),
    pulled_at                timestamptz not null default now()
);

create table if not exists park_factors (
    park_id                    text not null,
    year                       integer not null,
    park_factor_hr             numeric,
    park_factor_runs           numeric,
    field_orientation_degrees  numeric,
    pulled_at                  timestamptz not null default now(),
    primary key (park_id, year)
);

create table if not exists game_conditions (
    game_id        bigint primary key references games(game_id),
    temp_f         numeric,
    wind_speed     numeric,
    wind_direction numeric,
    humidity       numeric,
    precip_flag    boolean,
    wind_effect    text check (wind_effect in ('blowing_out','blowing_in','crosswind','neutral')),
    is_forecast    boolean not null,
    pulled_at      timestamptz not null default now()
);

create table if not exists team_form (
    team_id                     integer not null references teams(team_id),
    game_id                     bigint not null references games(game_id),
    record_last_10              text,
    run_diff_last_10            integer,
    games_back_playoff          numeric,
    clinched_or_eliminated_flag boolean,
    def_oaa_season               numeric,
    travel_fatigue_score          numeric,
    pulled_at                     timestamptz not null default now(),
    primary key (team_id, game_id)
);

create table if not exists lineup (
    game_id                      bigint not null references games(game_id),
    team_id                      integer not null references teams(team_id),
    player_id                    integer not null references players(player_id),
    batting_order_slot           integer,
    defensive_position           text,
    bats_hand                    text,
    playing_through_injury_flag  boolean,
    pulled_at                    timestamptz not null default now(),
    primary key (game_id, team_id, player_id)
);

create table if not exists bullpen_status (
    team_id                   integer not null references teams(team_id),
    game_id                   bigint not null references games(game_id),
    pitches_thrown_last_3d    integer,
    back_to_back_appearances  boolean,
    closer_available_flag     boolean,
    bullpen_era_last_15d      numeric,
    bullpen_era_season        numeric,
    pulled_at                 timestamptz not null default now(),
    primary key (team_id, game_id)
);

-- ============ Player performance (computed in-house from pitches / official stats) ============

create table if not exists starting_pitcher_form (
    pitcher_id              integer not null references players(player_id),
    game_id                 bigint not null references games(game_id),
    era_last_30d            numeric,
    era_season               numeric,
    fip                      numeric,
    fip_season                numeric,
    xfip                      numeric,
    xfip_season                numeric,
    k_pct_last_30d             numeric,
    k_pct_season                numeric,
    bb_pct_last_30d              numeric,
    bb_pct_season                 numeric,
    avg_velo_last_start            numeric,
    avg_velo_season                 numeric,
    velo_trend                       numeric,
    days_rest                        integer,
    pitch_count_last_start           integer,
    spin_rate_percentile             numeric,
    ground_ball_pct                  numeric,
    whiff_pct                        numeric,
    mlb_ip_count                     numeric,
    days_since_trade                 integer,
    pulled_at                        timestamptz not null default now(),
    primary key (pitcher_id, game_id)
);

create table if not exists starting_batter_form (
    batter_id                integer not null references players(player_id),
    game_id                  bigint not null references games(game_id),
    woba_season               numeric,
    woba_last_30d              numeric,
    k_pct_season                numeric,
    k_pct_last_30d                numeric,
    bb_pct_season                   numeric,
    bb_pct_last_30d                   numeric,
    avg_exit_velo_season                numeric,
    avg_exit_velo_last_30d                numeric,
    barrel_pct_season                       numeric,
    barrel_pct_last_30d                       numeric,
    vs_pitcher_hand_split                       text,
    days_since_last_game                         integer,
    mlb_pa_count                                 integer,
    days_since_trade                             integer,
    pulled_at                                    timestamptz not null default now(),
    primary key (batter_id, game_id)
);

create table if not exists umpire_stats (
    umpire_id                  integer not null,
    season                     integer not null,
    games_umpired               integer,
    ball_strike_accuracy_pct     numeric,
    zone_favor_score               numeric,
    k_rate_boost                     numeric,
    bb_rate_boost                      numeric,
    pulled_at                          timestamptz not null default now(),
    primary key (umpire_id, season)
);

-- ============ Raw data ============

create table if not exists pitches (
    pitch_id        bigserial primary key,
    at_bat_id       bigint not null,
    game_id         bigint not null references games(game_id),
    pitcher_id      integer,
    batter_id       integer,
    inning          integer,
    balls           integer,
    strikes         integer,
    pitch_type      text,
    release_speed   numeric,
    spin_rate       numeric,
    plate_x         numeric,
    plate_z         numeric,
    sz_top          numeric,
    sz_bot          numeric,
    pitch_result    text,
    exit_velocity   numeric,
    launch_angle    numeric,
    events          text,
    bb_type         text,
    hit_location    integer,
    pitch_number    integer not null,
    pulled_at       timestamptz not null default now(),
    constraint uq_pitches_natural_key unique (game_id, at_bat_id, pitch_number)
);
create index if not exists idx_pitches_game on pitches(game_id);
create index if not exists idx_pitches_pitcher on pitches(pitcher_id);
create index if not exists idx_pitches_batter on pitches(batter_id);
