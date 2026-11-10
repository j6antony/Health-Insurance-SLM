-- Run once in your Supabase project's SQL Editor. No secrets belong here.
create table if not exists public.api_usage_events (
    event_id uuid primary key,
    run_id uuid not null,
    source text not null,
    logged_at timestamptz,
    provider text,
    model text,
    operation text,
    status text,
    input_tokens bigint,
    output_tokens bigint,
    total_tokens bigint,
    estimated_cost_usd numeric(24,12),
    details jsonb not null,
    created_at timestamptz not null default clock_timestamp(),
    check (estimated_cost_usd is null or estimated_cost_usd >= 0)
);
create index if not exists api_usage_events_run_idx
    on public.api_usage_events (run_id, created_at, event_id);
alter table public.api_usage_events enable row level security;
-- Private ledger: no anonymous/authenticated policies or grants.
revoke all on public.api_usage_events from anon, authenticated;
grant usage on schema public to service_role;
grant select, insert on public.api_usage_events to service_role;

-- Query in SQL Editor for a per-run cost table. Null cost remains unknown.
-- select run_id, count(*) as calls,
--        sum(input_tokens) as input_tokens, sum(output_tokens) as output_tokens,
--        sum(estimated_cost_usd) as known_estimated_cost_usd,
--        count(*) filter (where estimated_cost_usd is null) as unknown_cost_calls
-- from public.api_usage_events group by run_id order by min(created_at) desc;

create or replace view public.api_usage_run_totals
with (security_invoker = true) as
select run_id, min(logged_at) as first_call, max(logged_at) as last_call,
       count(*) as calls, sum(input_tokens) as input_tokens,
       sum(output_tokens) as output_tokens,
       sum(estimated_cost_usd) as known_estimated_cost_usd,
       count(*) filter (where estimated_cost_usd is null) as unknown_cost_calls
from public.api_usage_events group by run_id;
revoke all on public.api_usage_run_totals from public, anon, authenticated;
grant select on public.api_usage_run_totals to service_role;
