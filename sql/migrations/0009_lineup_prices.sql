-- 0009: lineup-time odds snapshot for 2022-2026 (model design D6 / A8, 25 Sep 2026)
--
-- snapshot_type 'lineup' = the last clean price at or before T, where
-- T = first pitch - 90 minutes. First pitch is the scheduled start, except
-- doubleheader games, which use mlb.games.actual_first_pitch because game 2's
-- scheduled start is a placeholder (game 1 + 5 minutes).
--
-- Built from the raw Covers tick history on Colin's computer by
-- build_lineup_prices.py (covers_odds folder). The file travels encrypted
-- (OpenPGP AES-256; key in Supabase Vault as odds_lineup_key, never in the
-- repo) because the repo is public. The database fetches it, decrypts it,
-- checks it and loads it in one transaction: all rows or none.
--
-- Usage (two calls, because the download finishes in the background):
--   select net.http_get('<raw file url>', timeout_milliseconds := 60000);   -- returns a request id
--   select * from ops.load_lineup_prices(<request id>, dry_run := true);    -- checks only
--   select * from ops.load_lineup_prices(<request id>, dry_run := false);   -- checks, then writes

alter table mlb.odds_moneyline drop constraint odds_moneyline_snapshot_type_check;
alter table mlb.odds_moneyline add constraint odds_moneyline_snapshot_type_check
    check (snapshot_type = any (array['open', 'p1', 'mid', 'lineup', 'close']));
alter table mlb.odds_runline drop constraint odds_runline_snapshot_type_check;
alter table mlb.odds_runline add constraint odds_runline_snapshot_type_check
    check (snapshot_type = any (array['open', 'p1', 'mid', 'lineup', 'close']));
alter table mlb.odds_totals drop constraint odds_totals_snapshot_type_check;
alter table mlb.odds_totals add constraint odds_totals_snapshot_type_check
    check (snapshot_type = any (array['open', 'p1', 'mid', 'lineup', 'close']));

create or replace function ops.load_lineup_prices(p_request_id bigint, dry_run boolean default true)
returns table (check_name text, result text)
language plpgsql
security definer
set search_path = ''
as $$
declare
    armored text;
    plain   text;
    expected_sha constant text := 'cc2e103f9ae9a507dc1a6dfe6176e8b9793ea5703b2efa2ba9c05b7bb2fae3f1';
    expected_ml  constant int := 80525;
    expected_rl  constant int := 74857;
    expected_tot constant int := 80537;
    n_ml int; n_rl int; n_tot int; n_missing int; n_after_close int; n_after_t int; n_bad_price int; n_to_close int; n_tmp int;
begin
    select content into armored from net._http_response where id = p_request_id and status_code = 200;
    if armored is null then
        raise exception 'request % has no 200 response (still downloading, failed, or older than 6 hours)', p_request_id;
    end if;

    plain := convert_from(extensions.pgp_sym_decrypt_bytea(
                 extensions.dearmor(armored),
                 (select decrypted_secret from vault.decrypted_secrets where name = 'odds_lineup_key')), 'UTF8');
    if encode(extensions.digest(plain, 'sha256'), 'hex') <> expected_sha then
        raise exception 'decrypted file checksum does not match the file that was built';
    end if;

    create temp table lp on commit drop as
    select split_part(l, ',', 1) market,
           split_part(l, ',', 2)::bigint game_id,
           split_part(l, ',', 3) book,
           split_part(l, ',', 4)::timestamptz ts,
           nullif(split_part(l, ',', 5), '')::numeric line,
           split_part(l, ',', 6)::int price_a,
           split_part(l, ',', 7)::int price_b
    from string_to_table(plain, E'\n') with ordinality as t(l, n)
    where n > 1 and l <> '';

    -- build_odds_load.py only counts ticks up to 5 minutes before Covers'
    -- start time as clean. Where T falls after that cutoff (Covers listed an
    -- earlier start than the game really had), the last clean price at or
    -- before T IS the close, so use the close row.
    update lp set ts = c."timestamp", price_a = c.home_odds, price_b = c.away_odds
    from mlb.odds_moneyline c
    where lp.market = 'moneyline' and c.game_id = lp.game_id and c.book = lp.book and c.snapshot_type = 'close' and lp.ts > c."timestamp";
    get diagnostics n_to_close = row_count;
    update lp set ts = c."timestamp", line = c.line, price_a = c.home_odds, price_b = c.away_odds
    from mlb.odds_runline c
    where lp.market = 'runline' and c.game_id = lp.game_id and c.book = lp.book and c.snapshot_type = 'close' and lp.ts > c."timestamp";
    get diagnostics n_tmp = row_count; n_to_close := n_to_close + n_tmp;
    update lp set ts = c."timestamp", line = c.total_line, price_a = c.over_odds, price_b = c.under_odds
    from mlb.odds_totals c
    where lp.market = 'totals' and c.game_id = lp.game_id and c.book = lp.book and c.snapshot_type = 'close' and lp.ts > c."timestamp";
    get diagnostics n_tmp = row_count; n_to_close := n_to_close + n_tmp;

    select count(*) filter (where market = 'moneyline'), count(*) filter (where market = 'runline'),
           count(*) filter (where market = 'totals')
    into n_ml, n_rl, n_tot from lp;
    select count(*) into n_missing from lp where not exists (select 1 from mlb.games g where g.game_id = lp.game_id);
    select count(*) into n_bad_price from lp where abs(price_a) < 100 or abs(price_b) < 100 or abs(price_a) > 1000 or abs(price_b) > 1000;

    -- A lineup price can never be later than that book's close for the game,
    -- or later than T.
    select count(*) into n_after_close from lp
    where exists (
        select 1 from mlb.odds_moneyline c where lp.market = 'moneyline' and c.game_id = lp.game_id and c.book = lp.book and c.snapshot_type = 'close' and lp.ts > c."timestamp"
        union all
        select 1 from mlb.odds_runline c where lp.market = 'runline' and c.game_id = lp.game_id and c.book = lp.book and c.snapshot_type = 'close' and lp.ts > c."timestamp"
        union all
        select 1 from mlb.odds_totals c where lp.market = 'totals' and c.game_id = lp.game_id and c.book = lp.book and c.snapshot_type = 'close' and lp.ts > c."timestamp");
    select count(*) into n_after_t from lp join mlb.games g using (game_id)
    where lp.ts > (case when g.doubleheader_flag and g.actual_first_pitch is not null
                        then g.actual_first_pitch else g.first_pitch_time end) - interval '90 minutes';

    check_name := 'rows moneyline / runline / totals'; result := format('%s / %s / %s (expected %s / %s / %s)', n_ml, n_rl, n_tot, expected_ml, expected_rl, expected_tot); return next;
    check_name := 'game_ids missing from mlb.games'; result := n_missing::text; return next;
    check_name := 'prices outside +/-100..1000'; result := n_bad_price::text; return next;
    check_name := 'rows set to the close (T after the close cutoff)'; result := n_to_close::text; return next;
    check_name := 'lineup price later than the close'; result := n_after_close::text; return next;
    check_name := 'lineup price later than T'; result := n_after_t::text; return next;

    if n_ml <> expected_ml or n_rl <> expected_rl or n_tot <> expected_tot
       or n_missing > 0 or n_bad_price > 0 or n_after_close > 0 or n_after_t > 0 then
        check_name := 'RESULT'; result := 'checks failed, nothing written'; return next;
        return;
    end if;
    if dry_run then
        check_name := 'RESULT'; result := 'all checks pass (dry run, nothing written)'; return next;
        return;
    end if;

    insert into mlb.odds_moneyline (game_id, book, snapshot_type, source, "timestamp", home_odds, away_odds, implied_prob_home, implied_prob_away, pulled_at)
    select game_id, book, 'lineup', 'covers', ts, price_a, price_b,
           round(case when price_a > 0 then 100.0 / (price_a + 100) else -price_a / (-price_a + 100.0) end, 5),
           round(case when price_b > 0 then 100.0 / (price_b + 100) else -price_b / (-price_b + 100.0) end, 5), now()
    from lp where market = 'moneyline'
    on conflict (game_id, book, snapshot_type) do update set
        source = excluded.source, "timestamp" = excluded."timestamp", home_odds = excluded.home_odds, away_odds = excluded.away_odds,
        implied_prob_home = excluded.implied_prob_home, implied_prob_away = excluded.implied_prob_away, pulled_at = now();

    insert into mlb.odds_runline (game_id, book, snapshot_type, source, "timestamp", line, home_odds, away_odds, implied_prob_home, implied_prob_away, pulled_at)
    select game_id, book, 'lineup', 'covers', ts, line, price_a, price_b,
           round(case when price_a > 0 then 100.0 / (price_a + 100) else -price_a / (-price_a + 100.0) end, 5),
           round(case when price_b > 0 then 100.0 / (price_b + 100) else -price_b / (-price_b + 100.0) end, 5), now()
    from lp where market = 'runline'
    on conflict (game_id, book, snapshot_type) do update set
        source = excluded.source, "timestamp" = excluded."timestamp", line = excluded.line, home_odds = excluded.home_odds, away_odds = excluded.away_odds,
        implied_prob_home = excluded.implied_prob_home, implied_prob_away = excluded.implied_prob_away, pulled_at = now();

    insert into mlb.odds_totals (game_id, book, snapshot_type, source, "timestamp", total_line, over_odds, under_odds, implied_prob_over, implied_prob_under, pulled_at)
    select game_id, book, 'lineup', 'covers', ts, line, price_a, price_b,
           round(case when price_a > 0 then 100.0 / (price_a + 100) else -price_a / (-price_a + 100.0) end, 5),
           round(case when price_b > 0 then 100.0 / (price_b + 100) else -price_b / (-price_b + 100.0) end, 5), now()
    from lp where market = 'totals'
    on conflict (game_id, book, snapshot_type) do update set
        source = excluded.source, "timestamp" = excluded."timestamp", total_line = excluded.total_line, over_odds = excluded.over_odds, under_odds = excluded.under_odds,
        implied_prob_over = excluded.implied_prob_over, implied_prob_under = excluded.implied_prob_under, pulled_at = now();

    select count(*) into n_ml from mlb.odds_moneyline where snapshot_type = 'lineup';
    select count(*) into n_rl from mlb.odds_runline where snapshot_type = 'lineup';
    select count(*) into n_tot from mlb.odds_totals where snapshot_type = 'lineup';
    if n_ml <> expected_ml or n_rl <> expected_rl or n_tot <> expected_tot then
        raise exception 'after writing, lineup rows are % / % / %, expected % / % / %; rolled back',
            n_ml, n_rl, n_tot, expected_ml, expected_rl, expected_tot;
    end if;

    perform mlb.refresh_odds_exclusions();
    check_name := 'RESULT'; result := format('written: %s / %s / %s lineup rows; odds exclusions refreshed', n_ml, n_rl, n_tot); return next;
end;
$$;
revoke all on function ops.load_lineup_prices(bigint, boolean) from public, anon, authenticated;
