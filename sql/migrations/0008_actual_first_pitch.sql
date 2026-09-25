-- 0008: actual first pitch time (25 Sep 2026)
--
-- first_pitch_time is the scheduled start. For a traditional doubleheader
-- game 2 it is a placeholder (game 1 + 5 minutes) and the game really starts
-- about 3 hours later, which breaks any "N minutes before first pitch" rule.
-- actual_first_pitch comes from the MLB feed (gameData.gameInfo.firstPitch).
--
-- Backfilled 25 Sep 2026 for every 2021-2026 game played (14,741 of 14,743;
-- two games have no first pitch in the feed) by having Supabase itself call
-- the MLB API (pg_net), since the backfill needs no Python:
--   insert into ops.feed_requests ... net.http_get(<feed url>?fields=gameData,gameInfo,firstPitch,game,pk)
--   select ops.apply_first_pitch();
-- Median gap to the scheduled start is 1 minute; 317 games started 1+ hours
-- late (doubleheader game 2s and rain delays).

alter table mlb.games add column if not exists actual_first_pitch timestamptz;
comment on column mlb.games.actual_first_pitch is 'Actual first pitch from the MLB feed (gameData.gameInfo.firstPitch). first_pitch_time is the scheduled start, which for traditional doubleheader game 2 is a placeholder (game 1 + 5 min).';

create table if not exists ops.feed_requests (
    game_id      bigint primary key,
    request_id   bigint not null,
    requested_at timestamptz default now()
);
revoke all on ops.feed_requests from public, anon, authenticated;

create or replace function ops.apply_first_pitch() returns int
language plpgsql security definer set search_path = '' as $$
declare n int;
begin
    update mlb.games g
    set actual_first_pitch = (h.content::jsonb #>> '{gameData,gameInfo,firstPitch}')::timestamptz
    from ops.feed_requests f join net._http_response h on h.id = f.request_id
    where g.game_id = f.game_id and h.status_code = 200 and g.actual_first_pitch is null
      and (h.content::jsonb #>> '{gameData,game,pk}')::bigint = f.game_id
      and h.content::jsonb #>> '{gameData,gameInfo,firstPitch}' is not null;
    get diagnostics n = row_count;
    return n;
end $$;
revoke all on function ops.apply_first_pitch() from public, anon, authenticated;
