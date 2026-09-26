-- 0010: archived day-before weather forecasts for P1 features (25-26 Sep 2026)
--
-- Design 5.2: P1 (10am) must be trained on forecasts, not observed weather.
-- Source: Open-Meteo previous-runs archive, the forecast issued the day
-- before (`*_previous_day1`), at the hour of first pitch (scheduled start;
-- the real start for doubleheader games, whose scheduled game 2 time is a
-- placeholder). Fetched by Supabase itself through pg_net, one request per
-- venue per season (201 requests); the site allows only a few at a time, so
-- a temporary pg_cron job sent 3 a minute (castellanos-forecast-backfill,
-- removed when done).
--
-- Result, 2021-2026: 14,758 of 14,760 games played have a forecast. Against
-- the observed game-time temperature: mean absolute error 3.3-3.7 F per
-- season, bias -0.9 to +0.3 F, correlation 0.93-0.95.
--
-- Not yet done: forecasts for upcoming games are not pulled nightly (needed
-- for live P1 in 2027, phase E).

create table if not exists mlb.venue_coords (
    venue_id int primary key, name text, lat double precision not null, lon double precision not null,
    pulled_at timestamptz default now()
);
create table if not exists mlb.game_forecasts (
    game_id bigint primary key references mlb.games(game_id),
    fc_temp_f numeric, fc_wind_mph numeric, fc_wind_dir numeric, fc_precip_prob numeric,
    lead text not null default 'previous_day1',
    source text not null default 'open-meteo previous-runs',
    pulled_at timestamptz default now()
);
create table if not exists ops.forecast_requests (
    venue_id int, season int, request_id bigint, primary key (venue_id, season)
);
revoke all on ops.forecast_requests from public, anon, authenticated;

-- Steps used for the backfill (kept for reference):
--   1. venue coordinates: net.http_get(statsapi /api/v1/venues?venueIds=...&hydrate=location)
--      -> mlb.venue_coords (Estadio Alfredo Harp Helu, 5340, entered by hand)
--   2. one previous-runs request per venue and season into ops.forecast_requests,
--      retried in small batches by ops.requeue_forecasts(n) until all returned 200
--   3. responses copied to ops.forecast_raw (pg_net keeps them only 6 hours),
--      unpacked to ops.forecast_hours (one row per venue and hour), then
--      matched to each game's first-pitch hour into mlb.game_forecasts
