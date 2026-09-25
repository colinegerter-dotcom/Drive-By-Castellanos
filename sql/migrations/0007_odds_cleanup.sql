-- 0007: odds cleanup without deleting anything (25 Sep 2026, rules approved by Colin)
--
-- The raw odds tables stay exactly as loaded. mlb.odds_exclusions lists
-- every row (or whole game) the model should ignore and why, and the
-- *_clean views are the raw tables minus those rows. The model reads only
-- the clean views. To undo: drop the views and the table.
--
-- Rules (checked 25 Sep 2026 against the 2021-2026 load):
--   stale_close   whole game, every market: the latest closing moneyline
--                 across books is more than 12 hours before scheduled first
--                 pitch. Catches doubleheader game 2 and makeup games whose
--                 line stopped updating days early, and a July 2024 cluster
--                 where Covers dates are one day off so odds sit on the
--                 wrong day's game. 30 games.
--   alt_line      totals row: the price is lopsided (no-vig over chance
--                 outside 30-70%), so it is an alternate line, not the main one
--   off_market    totals row: 1.5+ runs from the median of the other books'
--                 main lines (needs 3+ books). Mostly first-five totals that
--                 BetMGM, BetRivers and Caesars showed in the full-game slot
--                 in late Jul-Aug 2024
--   incoherent    any row whose two sides' implied chances add to under 100%
--                 or over 112%
--
-- Re-run after any odds reload:  select mlb.refresh_odds_exclusions();

create table if not exists mlb.odds_exclusions (
    market        text   not null,  -- moneyline | runline | totals | all
    game_id       bigint not null,
    book          text,             -- null = every book
    snapshot_type text,             -- null = every snapshot
    reason        text   not null,
    detail        text
);
create index if not exists odds_exclusions_game on mlb.odds_exclusions (game_id);

create or replace function mlb.refresh_odds_exclusions()
returns table (reason text, market text, n bigint)
language plpgsql
set search_path = ''
as $$
begin
    delete from mlb.odds_exclusions;

    -- stale_close: whole game
    insert into mlb.odds_exclusions (market, game_id, reason, detail)
    select 'all', g.game_id, 'stale_close',
           format('latest close %s, first pitch %s', c.last_close, g.first_pitch_time)
    from (select game_id, max("timestamp") last_close
          from mlb.odds_moneyline
          where snapshot_type = 'close' and book <> 'sbr_consensus'
          group by 1) c
    join mlb.games g using (game_id)
    where c.last_close < g.first_pitch_time - interval '12 hours';

    -- totals: alt_line and off_market
    insert into mlb.odds_exclusions (market, game_id, book, snapshot_type, reason, detail)
    with t as (
        select game_id, book, snapshot_type, total_line tl, over_odds, under_odds,
               implied_prob_over / (implied_prob_over + implied_prob_under) p
        from mlb.odds_totals
        where book <> 'sbr_consensus'
    ), m as (
        select game_id, snapshot_type,
               percentile_cont(0.5) within group (order by tl) med, count(*) nb
        from t where p between 0.3 and 0.7
        group by 1, 2
    )
    select 'totals', t.game_id, t.book, t.snapshot_type,
           case when t.p not between 0.3 and 0.7 then 'alt_line' else 'off_market' end,
           format('line %s at %s/%s, other books median %s', t.tl, t.over_odds, t.under_odds, m.med)
    from t left join m using (game_id, snapshot_type)
    where t.p not between 0.3 and 0.7
       or (m.nb >= 3 and abs(t.tl - m.med) >= 1.5);

    -- incoherent prices, all three markets
    insert into mlb.odds_exclusions (market, game_id, book, snapshot_type, reason, detail)
    select 'moneyline', game_id, book, snapshot_type, 'incoherent', format('%s/%s', home_odds, away_odds)
    from mlb.odds_moneyline
    where implied_prob_home + implied_prob_away not between 1 and 1.12
    union all
    select 'runline', game_id, book, snapshot_type, 'incoherent', format('%s %s/%s', line, home_odds, away_odds)
    from mlb.odds_runline
    where implied_prob_home + implied_prob_away not between 1 and 1.12
    union all
    select 'totals', game_id, book, snapshot_type, 'incoherent', format('%s %s/%s', total_line, over_odds, under_odds)
    from mlb.odds_totals
    where implied_prob_over + implied_prob_under not between 1 and 1.12;

    return query
    select e.reason, e.market, count(*) from mlb.odds_exclusions e group by 1, 2 order by 1, 2;
end;
$$;

-- A row is kept unless a whole-game exclusion or a matching row exclusion exists.
create or replace view mlb.odds_moneyline_clean with (security_invoker = true) as
select o.* from mlb.odds_moneyline o
where not exists (
    select 1 from mlb.odds_exclusions e
    where e.game_id = o.game_id and e.market in ('all', 'moneyline')
      and (e.book is null or e.book = o.book)
      and (e.snapshot_type is null or e.snapshot_type = o.snapshot_type));

create or replace view mlb.odds_runline_clean with (security_invoker = true) as
select o.* from mlb.odds_runline o
where not exists (
    select 1 from mlb.odds_exclusions e
    where e.game_id = o.game_id and e.market in ('all', 'runline')
      and (e.book is null or e.book = o.book)
      and (e.snapshot_type is null or e.snapshot_type = o.snapshot_type));

create or replace view mlb.odds_totals_clean with (security_invoker = true) as
select o.* from mlb.odds_totals o
where not exists (
    select 1 from mlb.odds_exclusions e
    where e.game_id = o.game_id and e.market in ('all', 'totals')
      and (e.book is null or e.book = o.book)
      and (e.snapshot_type is null or e.snapshot_type = o.snapshot_type));

select * from mlb.refresh_odds_exclusions();
