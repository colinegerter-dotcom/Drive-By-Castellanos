-- Covers odds (2022-2026): allow a 'p1' snapshot and record the source.
--
-- Why 'p1': the model's G4 test is closing-line value, which needs the price
-- at bet time as well as the close. P1 is the morning prediction point, fixed
-- at 10:00 am Central on game day (decided 24 Sep 2026). A P1 row holds the
-- last clean price at or before 10:00 CT (or 5 min before first pitch, if the
-- game starts earlier).
--
-- Snapshot meanings for source = 'covers':
--   open  = earliest clean pre-game price Covers logged for that book
--   p1    = price in effect at 10:00 am CT on game day
--   close = price in effect 5 min before first pitch
-- "Clean" and "in effect" are defined in scripts/load_covers_odds.py.
--
-- book = the real sportsbook name (DraftKings, FanDuel, ...), not a consensus.
-- devigged_* columns are left NULL on purpose: the devig method
-- (multiplicative / power / Shin) is chosen per market in the model's market
-- layer (model-design.md, section 13), not fixed at load time.
--
-- Applied to Supabase project fqokvprujriyldqgzozy via the Supabase MCP tool
-- on 24 Sep 2026. This file is the repo's copy of record.

set search_path to mlb, public;

alter table odds_moneyline drop constraint if exists odds_moneyline_snapshot_type_check;
alter table odds_moneyline add constraint odds_moneyline_snapshot_type_check
    check (snapshot_type in ('open','p1','mid','close'));

alter table odds_runline drop constraint if exists odds_runline_snapshot_type_check;
alter table odds_runline add constraint odds_runline_snapshot_type_check
    check (snapshot_type in ('open','p1','mid','close'));

alter table odds_totals drop constraint if exists odds_totals_snapshot_type_check;
alter table odds_totals add constraint odds_totals_snapshot_type_check
    check (snapshot_type in ('open','p1','mid','close'));

alter table odds_f5 drop constraint if exists odds_f5_snapshot_type_check;
alter table odds_f5 add constraint odds_f5_snapshot_type_check
    check (snapshot_type in ('open','p1','mid','close'));
