-- 0006: start the GitHub workflows from Supabase instead of GitHub's timer (25 Sep 2026)
--
-- Why: GitHub's scheduled runs started 2.5 to 6 hours late on 19-23 Sep and
-- did not run at all on 24 Sep. GitHub also switches off scheduled runs in a
-- public repo after 60 days with no commits, which the offseason would hit.
-- pg_cron runs on time, and a workflow started through the API is not
-- affected by the 60-day rule.
--
-- How:
--   * ops.dispatch_workflow() calls GitHub's workflow_dispatch API with a
--     fine-grained token (Actions read/write on this one repo only) that
--     Colin stores in Supabase Vault under the name github_actions_token.
--     The token never appears in code, in this file, or in chat.
--   * Every call is logged in ops.workflow_dispatches, and GitHub's reply
--     (204 = accepted) is copied in by ops.record_dispatch_results().
--   * ops.daily_pull_check() runs twice later in the morning. If yesterday
--     had games but none of them was written since midnight Central, the
--     nightly pull did not run (or failed), so it starts it again. At most
--     3 starts per day, so a broken run cannot loop.
--
-- The ops schema is private: no access for the anon/authenticated API roles.

create extension if not exists pg_net with schema extensions;
create extension if not exists pg_cron with schema pg_catalog;

create schema if not exists ops;
revoke all on schema ops from public, anon, authenticated;

create table if not exists ops.workflow_dispatches (
    id           bigserial primary key,
    workflow     text        not null,
    reason       text        not null,
    requested_at timestamptz not null default now(),
    request_id   bigint,          -- pg_net request id
    status_code  int,             -- GitHub's reply; 204 means the run was accepted
    error        text
);
revoke all on ops.workflow_dispatches from public, anon, authenticated;

create or replace function ops.dispatch_workflow(p_workflow text, p_reason text)
returns bigint
language plpgsql
security definer
set search_path = ''
as $$
declare
    tok text;
    rid bigint;
    lid bigint;
begin
    select decrypted_secret into tok
    from vault.decrypted_secrets
    where name = 'github_actions_token';

    if tok is null then
        insert into ops.workflow_dispatches (workflow, reason, error)
        values (p_workflow, p_reason, 'github_actions_token is missing from Vault')
        returning id into lid;
        return lid;
    end if;

    select net.http_post(
        url := format('https://api.github.com/repos/colinegerter-dotcom/Drive-By-Castellanos/actions/workflows/%s/dispatches', p_workflow),
        body := jsonb_build_object('ref', 'main'),
        headers := jsonb_build_object(
            'Authorization', 'Bearer ' || tok,
            'Accept', 'application/vnd.github+json',
            'X-GitHub-Api-Version', '2022-11-28',
            'User-Agent', 'castellanos-supabase-cron',
            'Content-Type', 'application/json'),
        timeout_milliseconds := 15000)
    into rid;

    insert into ops.workflow_dispatches (workflow, reason, request_id)
    values (p_workflow, p_reason, rid)
    returning id into lid;
    return lid;
end;
$$;

-- pg_net keeps replies for about 6 hours, so copy them into the log.
create or replace function ops.record_dispatch_results()
returns int
language plpgsql
security definer
set search_path = ''
as $$
declare n int;
begin
    update ops.workflow_dispatches d
    set status_code = r.status_code,
        error = case when r.status_code = 204 then null
                     else coalesce(r.error_msg, left(r.content::text, 500)) end
    from net._http_response r
    where r.id = d.request_id
      and d.status_code is null;
    get diagnostics n = row_count;
    return n;
end;
$$;

create or replace function ops.daily_pull_check()
returns text
language plpgsql
security definer
set search_path = ''
as $$
declare
    today_ct date := (now() at time zone 'America/Chicago')::date;
    yday     date := today_ct - 1;
    since    timestamptz := today_ct::timestamp at time zone 'America/Chicago';
    starts   int;
begin
    perform ops.record_dispatch_results();

    if not exists (select 1 from mlb.games where date = yday) then
        return 'no games yesterday';
    end if;

    if exists (select 1 from mlb.games where date = yday and pulled_at >= since) then
        return 'nightly pull already ran';
    end if;

    select count(*) into starts
    from ops.workflow_dispatches
    where workflow = 'daily-pull.yml' and requested_at >= since;

    if starts >= 3 then
        return 'not run yet, but already started 3 times today; leaving it';
    end if;

    perform ops.dispatch_workflow('daily-pull.yml', 'retry: yesterday not written yet');
    return 'not run yet; started it again';
end;
$$;

revoke all on function ops.dispatch_workflow(text, text) from public, anon, authenticated;
revoke all on function ops.record_dispatch_results() from public, anon, authenticated;
revoke all on function ops.daily_pull_check() from public, anon, authenticated;

-- Schedules (pg_cron uses UTC; Central times shown for daylight time, one hour
-- earlier in standard time).
select cron.schedule('castellanos-daily-pull',     '17 11 * * *',    $$select ops.dispatch_workflow('daily-pull.yml', 'scheduled')$$);   -- 6:17 am
select cron.schedule('castellanos-daily-check',    '17 13,15 * * *', $$select ops.daily_pull_check()$$);                                -- 8:17 and 10:17 am
select cron.schedule('castellanos-weekly-snapshot','7 16 * * 1',     $$select ops.dispatch_workflow('db-snapshot.yml', 'scheduled')$$);  -- Mondays 11:07 am
select cron.schedule('castellanos-record-results', '27 * * * *',     $$select ops.record_dispatch_results()$$);
select cron.schedule('castellanos-cron-cleanup',   '37 9 * * 0',     $$delete from cron.job_run_details where end_time < now() - interval '30 days'$$);
