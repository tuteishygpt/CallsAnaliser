-- Token pricing and usage schema for Calls Analyser.
-- Run in Supabase SQL editor or include in your migration flow.
-- The application should use a server-side Supabase key with service_role access.

create table if not exists public.model_pricing (
    id bigserial primary key,
    provider text not null,
    model_key text not null,
    currency text not null default 'USD',
    input_cost_per_1m numeric(12, 6) not null default 0,
    output_cost_per_1m numeric(12, 6) not null default 0,
    input_price_per_1m numeric(12, 6) not null default 0,
    output_price_per_1m numeric(12, 6) not null default 0,
    effective_from date not null default current_date,
    effective_to date,
    is_active boolean not null default true,
    created_at timestamptz not null default now()
);

create index if not exists idx_model_pricing_lookup
    on public.model_pricing (provider, model_key, is_active, effective_from desc);

create table if not exists public.analysis_usage (
    id bigserial primary key,
    tenant_id text not null,
    call_unique_id text not null,
    call_started_at timestamptz,
    call_user text,
    caller_id text,
    destination text,
    duration_seconds integer,
    prompt_key text not null,
    custom_fragment_hash text not null default '',
    provider_name text not null,
    model_key text not null,
    mode text not null,
    cache_hit boolean not null default false,
    prompt_token_count integer not null default 0,
    candidates_token_count integer not null default 0,
    thoughts_token_count integer not null default 0,
    total_token_count integer not null default 0,
    input_cost_per_1m_snapshot numeric(12, 6) not null default 0,
    output_cost_per_1m_snapshot numeric(12, 6) not null default 0,
    input_price_per_1m_snapshot numeric(12, 6) not null default 0,
    output_price_per_1m_snapshot numeric(12, 6) not null default 0,
    estimated_cost numeric(14, 8) not null default 0,
    estimated_client_price numeric(14, 8) not null default 0,
    currency text not null default 'USD',
    analysis_result_cache_key text,
    created_at timestamptz not null default now()
);

create index if not exists idx_analysis_usage_tenant_created
    on public.analysis_usage (tenant_id, created_at desc);

create index if not exists idx_analysis_usage_call
    on public.analysis_usage (tenant_id, call_unique_id);

alter table public.model_pricing enable row level security;
alter table public.analysis_usage enable row level security;

grant select on public.model_pricing to service_role;
grant insert, select on public.analysis_usage to service_role;
grant usage, select on sequence public.model_pricing_id_seq to service_role;
grant usage, select on sequence public.analysis_usage_id_seq to service_role;

insert into public.model_pricing (
    provider,
    model_key,
    currency,
    input_cost_per_1m,
    output_cost_per_1m,
    input_price_per_1m,
    output_price_per_1m,
    effective_from
) values (
    'gemini',
    'models/gemini-test',
    'USD',
    0,
    0,
    0,
    0,
    current_date
) on conflict do nothing;
