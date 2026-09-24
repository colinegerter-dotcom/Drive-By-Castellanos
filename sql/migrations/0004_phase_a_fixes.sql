-- Phase A data fixes (model design v3.1, items A1, A2, A4), 24 Sep 2026.
-- Every change here only ADDS a nullable column or a new table, so code
-- from before this migration keeps working against it unchanged.
--
-- Applied to Supabase project fqokvprujriyldqgzozy via the Supabase MCP tool
-- on 24 Sep 2026. This file is the repo's copy of record.

set search_path to mlb, public;

-- A2: stable park key. games.venue is a name, and names change with
-- sponsors (Minute Maid Park -> Daikin Park). park_factors.park_id is
-- already this id, as text.
alter table games add column if not exists venue_id integer;

-- Bullpen availability, per reliever (see pipelines/games/bullpen_status.py).
-- back_to_back_appearances keeps its name but now means "any reliever
-- pitched on each of the last two days".
alter table bullpen_status add column if not exists relievers_back_to_back integer;
alter table bullpen_status add column if not exists unavailable_reliever_ids integer[];

-- A4: suspended-and-resumed games. MLB files all of a resumed game's
-- pitches under original_date; anything computing "as of a date" must treat
-- them as thrown on resume_date.
create table if not exists resumed_games (
    game_id        bigint primary key references games(game_id),
    original_date  date not null,
    resume_date    date not null,
    status         text,
    pulled_at      timestamptz not null default now()
);
