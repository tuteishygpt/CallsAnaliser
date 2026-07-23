-- Multi-tenant foundation schema for Calls Analyser.
-- Apply with a Supabase service-role connection.

create extension if not exists pgcrypto;

create table if not exists public.tenants (
    id text primary key,
    display_name text not null,
    status text not null default 'active',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.tenant_users (
    id uuid primary key default gen_random_uuid(),
    login text unique not null,
    password_hash text not null,
    display_name text,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    last_login_at timestamptz
);

create table if not exists public.tenant_user_access (
    user_id uuid not null references public.tenant_users(id) on delete cascade,
    tenant_id text not null references public.tenants(id) on delete cascade,
    role text not null default 'operator',
    primary key (user_id, tenant_id)
);

create table if not exists public.tenant_secrets (
    tenant_id text not null references public.tenants(id) on delete cascade,
    key text not null,
    encrypted_value text not null,
    updated_at timestamptz not null default now(),
    primary key (tenant_id, key)
);

comment on column public.tenant_secrets.encrypted_value is
    'AES-256-GCM envelope enc:v1:<unpadded nonce base64url>:<unpadded ciphertext+tag base64url>; tenant_id and key are AAD.';

create table if not exists public.tenant_settings (
    tenant_id text not null references public.tenants(id) on delete cascade,
    key text not null,
    value jsonb not null,
    updated_at timestamptz not null default now(),
    primary key (tenant_id, key)
);

create table if not exists public.tenant_prompt_templates (
    id uuid primary key default gen_random_uuid(),
    tenant_id text not null references public.tenants(id) on delete cascade,
    key text not null,
    title text not null,
    body text not null,
    is_active boolean not null default true,
    version integer not null default 1,
    created_by uuid references public.tenant_users(id),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (tenant_id, key, version)
);

create table if not exists public.analysis_results (
    tenant_id text not null references public.tenants(id) on delete cascade,
    call_unique_id text not null,
    prompt_key text not null,
    prompt_version integer not null default 1,
    provider_name text not null,
    model_key text not null,
    custom_fragment text not null default '',
    result_text text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (
        tenant_id,
        call_unique_id,
        prompt_key,
        prompt_version,
        provider_name,
        model_key,
        custom_fragment
    )
);

create table if not exists public.scheduler_runs (
    tenant_id text not null references public.tenants(id) on delete cascade,
    scheduled_for timestamptz not null,
    prompt_key text not null,
    prompt_version integer not null,
    model_key text not null,
    status text not null check (status in ('running', 'success', 'partial', 'failed')),
    total_count integer not null default 0,
    success_count integer not null default 0,
    failure_count integer not null default 0,
    cached_count integer not null default 0,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    primary key (tenant_id, scheduled_for, prompt_key, prompt_version, model_key)
);

create index if not exists idx_tenant_user_access_tenant
    on public.tenant_user_access (tenant_id);

create index if not exists idx_tenant_prompt_templates_active
    on public.tenant_prompt_templates (tenant_id, key)
    where is_active;

alter table public.tenants enable row level security;
alter table public.tenant_users enable row level security;
alter table public.tenant_user_access enable row level security;
alter table public.tenant_settings enable row level security;
alter table public.tenant_prompt_templates enable row level security;
alter table public.tenant_secrets enable row level security;
alter table public.analysis_results enable row level security;
alter table public.scheduler_runs enable row level security;

drop policy if exists "service role manages tenants" on public.tenants;
create policy "service role manages tenants"
    on public.tenants
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "service role manages tenant users" on public.tenant_users;
create policy "service role manages tenant users"
    on public.tenant_users
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "service role manages tenant user access" on public.tenant_user_access;
create policy "service role manages tenant user access"
    on public.tenant_user_access
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "service role manages tenant settings" on public.tenant_settings;
create policy "service role manages tenant settings"
    on public.tenant_settings
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "service role manages tenant prompt templates" on public.tenant_prompt_templates;
create policy "service role manages tenant prompt templates"
    on public.tenant_prompt_templates
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "service role manages tenant secrets" on public.tenant_secrets;
create policy "service role manages tenant secrets"
    on public.tenant_secrets
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "service role manages analysis results" on public.analysis_results;
create policy "service role manages analysis results"
    on public.analysis_results
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists "service role manages scheduler runs" on public.scheduler_runs;
create policy "service role manages scheduler runs"
    on public.scheduler_runs
    for all
    to service_role
    using (true)
    with check (true);
