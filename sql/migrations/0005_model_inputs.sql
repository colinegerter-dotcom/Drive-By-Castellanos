-- Model inputs (model design v3.2, phase A items A3 and A5), 25 Sep 2026.
-- Adds only: two tables, one view, one nullable column. Nothing existing
-- changes, so code from before this migration keeps working unchanged.
--
-- Applied to Supabase project fqokvprujriyldqgzozy via the Supabase MCP tool
-- on 25 Sep 2026. This file is the repo's copy of record.

set search_path to mlb, public;

-- A3: runs in every half inning, from the pitch files
-- (pipelines/games/inning_scores.py). A skipped bottom of the 9th has no
-- row because it was never played.
create table if not exists inning_scores (
    game_id        bigint   not null references games(game_id),
    inning         smallint not null,
    half           text     not null check (half in ('top', 'bottom')),
    batting_team   integer  not null references teams(team_id),
    runs           smallint not null check (runs >= 0),
    pulled_at      timestamptz not null default now(),
    primary key (game_id, inning, half)
);

-- The model's targets per team-game: runs through 5, through 8, and final.
-- innings_played tells a full game from one that stopped early: runs_8
-- equals runs_total in a 7-inning 2021 doubleheader game or a game called
-- after 8, so filter on it when fitting the 8-inning target.
create or replace view team_game_runs as
select
    i.game_id,
    i.batting_team                                   as team_id,
    (i.batting_team = g.home_team)                   as is_home,
    sum(i.runs) filter (where i.inning <= 5)         as runs_f5,
    sum(i.runs) filter (where i.inning <= 8)         as runs_8,
    sum(i.runs)                                      as runs_total,
    max(i.inning)                                    as last_inning_batted,
    max(r.innings_played)                            as innings_played
from inning_scores i
join games g on g.game_id = i.game_id
left join game_results r on r.game_id = i.game_id
group by i.game_id, i.batting_team, g.home_team;

-- A5: birth dates, for the age adjustment in the prior layer.
alter table players add column if not exists birth_date date;

-- A5: official MLB season lines (regular season), one row per player,
-- season and group. A traded player's row is MLB's own combined line for
-- the season, never the team stints added up (the API returns both).
-- Seasons before 2021 feed the prior layer; 2021 on are for cross-checks,
-- since the model computes those seasons from the pitch files itself.
create table if not exists player_season_stats (
    player_id          integer  not null references players(player_id),
    season             integer  not null,
    stat_group         text     not null check (stat_group in ('hitting', 'pitching')),
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
    primary key (player_id, season, stat_group)
);
