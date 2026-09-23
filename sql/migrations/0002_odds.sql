-- Odds tables, per the locked schema (mlb-model-schema.md, "Odds and model").
-- Composite key (game_id, book, snapshot_type) so one book's snapshot can never
-- overwrite another's.
--
-- Two additions beyond the locked schema, both needed for archive data:
--   * source: 'sbr_archive' for the sportsbookreviewsonline historical files,
--     'odds_api' for live snapshots later. Keeps backtest and live rows separable.
--   * timestamp is nullable: the archive gives open/close prices with no time.
--
-- Applied to Supabase project fqokvprujriyldqgzozy via the Supabase MCP tool
-- on 23 Sep 2026. This file is the repo's copy of record, same as 0001_init.sql.

set search_path to mlb, public;

create table if not exists odds_moneyline (
    game_id              bigint not null references games(game_id),
    book                 text not null,
    snapshot_type        text not null check (snapshot_type in ('open','mid','close')),
    source               text not null,
    "timestamp"          timestamptz,
    home_odds            integer,
    away_odds            integer,
    implied_prob_home    numeric,
    implied_prob_away    numeric,
    devigged_prob_home   numeric,
    pulled_at            timestamptz not null default now(),
    primary key (game_id, book, snapshot_type)
);

create table if not exists odds_runline (
    game_id              bigint not null references games(game_id),
    book                 text not null,
    snapshot_type        text not null check (snapshot_type in ('open','mid','close')),
    source               text not null,
    "timestamp"          timestamptz,
    line                 numeric,          -- from the HOME team's side, e.g. -1.5
    home_odds            integer,
    away_odds            integer,
    implied_prob_home    numeric,
    implied_prob_away    numeric,
    devigged_prob_home   numeric,          -- probability home covers `line`
    pulled_at            timestamptz not null default now(),
    primary key (game_id, book, snapshot_type)
);

-- F5 moneyline can be three-way (a tie after 5 is common), so a draw price is
-- allowed and the devig must account for it when present.
create table if not exists odds_f5 (
    game_id              bigint not null references games(game_id),
    book                 text not null,
    snapshot_type        text not null check (snapshot_type in ('open','mid','close')),
    source               text not null,
    "timestamp"          timestamptz,
    line                 numeric,
    home_odds            integer,
    away_odds            integer,
    draw_odds            integer,
    implied_prob_home    numeric,
    implied_prob_away    numeric,
    devigged_prob_home   numeric,
    pulled_at            timestamptz not null default now(),
    primary key (game_id, book, snapshot_type)
);

create table if not exists odds_totals (
    game_id              bigint not null references games(game_id),
    book                 text not null,
    snapshot_type        text not null check (snapshot_type in ('open','mid','close')),
    source               text not null,
    "timestamp"          timestamptz,
    total_line           numeric,
    over_odds            integer,
    under_odds           integer,
    implied_prob_over    numeric,
    implied_prob_under   numeric,
    devigged_prob_over   numeric,
    pulled_at            timestamptz not null default now(),
    primary key (game_id, book, snapshot_type)
);
