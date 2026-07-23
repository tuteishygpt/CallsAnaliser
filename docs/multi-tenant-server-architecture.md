# Multi-tenant architecture for one shared Calls Analyser server

Date: 2026-07-06  
Branch: `codex/multi-tenant-architecture`

## Goal

Run one deployed copy of the Calls Analyser code for multiple tenants on the same server. Each tenant must have isolated configuration, prompts, telephony credentials, batch settings, cache/results, usage reports, and user access. Users must sign in with their own login and password, and after sign-in the app must infer the allowed tenant instead of trusting a manually typed tenant id.

## Current State

The project already has several tenant-ready pieces:

- `TenantService` resolves tenant-scoped telephony settings from environment variables such as `<TENANT>_TELEPHONY_PROVIDER`, `<TENANT>_MTS_DOMAIN`, `<TENANT>_MTS_API_KEY`, and `<TENANT>_VOCHI_API_KEY`.
- `AnalysisService` includes `tenant_id` in the cache key, so AI results are logically separated by tenant.
- `CallLogService.ensure_recording()` prefixes local recording filenames with `tenant_id`.
- Supabase usage reporting stores `tenant_id` in `analysis_usage` and filters reports by tenant.
- The UI already has a tenant id textbox and passes it into call filtering, playback, batch analysis, direct AI analysis, email reports, and usage reports.

The current implementation is not yet safe for multi-tenant production on a shared server:

- UI access is protected by one global `VOCHI_UI_PASSWORD`, not by per-user credentials.
- Any authenticated user can type any tenant id in the UI.
- `CallLogService` is built once at startup from the default tenant's provider adapter, so switching tenants at runtime is unsafe when tenants use different providers or credentials.
- Prompts and batch prompt defaults are global Python constants in `calls_analyser/config.py`.
- `batch_params.json` is global for the whole app.
- Email settings are global (`EMAIL_TO`, `BREVO_API_KEY`, `GOOGLE_app`), so tenant-specific recipients and sender rules are not modeled.
- Supabase schema for `analysis_results` is referenced by code but no migration file is present in `docs/supabase`.
- The optional FastAPI API accepts `tenant_id` query params without authentication or tenant authorization.

## Recommended Approach

Use one process and one codebase, but introduce a tenant-aware application layer:

1. Add persistent tenant/user/settings tables.
2. Replace the global password gate with login/password authentication.
3. Bind each authenticated session to an allowed tenant set.
4. Remove editable tenant id as a trust boundary; use tenant from session, or present a dropdown of tenants the user can access.
5. Resolve telephony adapters per tenant at call time or through a tenant-aware provider factory.
6. Load prompts and batch settings from tenant settings first, falling back to code defaults.

This keeps deployment simple while making data access explicit and auditable.

## Data Model

Recommended Supabase/Postgres tables:

### `tenants`

Stores public tenant metadata and status.

Columns:

- `id text primary key`
- `display_name text not null`
- `status text not null default 'active'`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

### `tenant_users`

Stores application users. Passwords must be hashed with a strong password hashing function such as Argon2 or bcrypt; never store raw passwords.

Columns:

- `id uuid primary key default gen_random_uuid()`
- `login text unique not null`
- `password_hash text not null`
- `display_name text`
- `is_active boolean not null default true`
- `created_at timestamptz not null default now()`
- `last_login_at timestamptz`

### `tenant_user_access`

Maps users to one or more tenants.

Columns:

- `user_id uuid references tenant_users(id) on delete cascade`
- `tenant_id text references tenants(id) on delete cascade`
- `role text not null default 'operator'`
- `primary key (user_id, tenant_id)`

Suggested roles:

- `admin`: manage tenant settings, users, prompts, reports.
- `manager`: run analysis and view reports.
- `operator`: filter calls, play recordings, run allowed analyses.

### `tenant_secrets`

Stores encrypted tenant-specific credentials, or stores references to secrets managed by the hosting platform. If Supabase is used directly, keep app access server-side only and use encryption before inserting secret values.

Columns:

- `tenant_id text references tenants(id) on delete cascade`
- `key text not null`
- `encrypted_value text not null`
- `updated_at timestamptz not null default now()`
- `primary key (tenant_id, key)`

Initial keys:

- `TELEPHONY_PROVIDER`
- `VOCHI_BASE_URL`
- `VOCHI_API_KEY`
- `MTS_DOMAIN`
- `MTS_API_KEY`
- `EMAIL_TO`
- `EMAIL_FROM`
- `EMAIL_FROM_NAME`

### `tenant_settings`

Stores non-secret tenant settings.

Columns:

- `tenant_id text references tenants(id) on delete cascade`
- `key text not null`
- `value jsonb not null`
- `updated_at timestamptz not null default now()`
- `primary key (tenant_id, key)`

Initial settings:

- `default_language`
- `default_model_key`
- `telephony_provider`
- `batch_model_key`
- `batch_language_code`
- `batch_enabled`
- `batch_size`
- `scheduler_enabled`
- `scheduler_mode`
- `scheduler_cron_time`
- `scheduler_interval_minutes`
- `scheduler_filters`
- `custom_batch_enabled`

### `tenant_prompt_templates`

Stores tenant-specific prompts.

Columns:

- `id uuid primary key default gen_random_uuid()`
- `tenant_id text references tenants(id) on delete cascade`
- `key text not null`
- `title text not null`
- `body text not null`
- `is_active boolean not null default true`
- `version integer not null default 1`
- `created_by uuid references tenant_users(id)`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`
- `unique (tenant_id, key, version)`

Runtime behavior:

- Load active prompts for the tenant.
- If a tenant does not override a prompt, fall back to `calls_analyser/config.py`.
- Include prompt key, prompt version, and custom prompt hash in cache/usage records.

### `analysis_results`

Add or formalize the table used by `SupabaseCache`.

Columns:

- `tenant_id text not null`
- `call_unique_id text not null`
- `prompt_key text not null`
- `prompt_version integer not null default 1`
- `provider_name text not null`
- `model_key text not null`
- `custom_fragment text not null default ''`
- `result_text text not null`
- `metadata jsonb not null default '{}'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Unique key:

- `(tenant_id, call_unique_id, prompt_key, prompt_version, provider_name, model_key, custom_fragment)`

## Authentication and Tenant Selection

Replace the current password-only gate with an `AuthService`:

- `authenticate(login, password) -> AuthenticatedUser | None`
- `list_allowed_tenants(user_id) -> list[TenantSummary]`
- `can_access_tenant(user_id, tenant_id) -> bool`

Recommended UI behavior:

- Login form asks for login and password.
- After login, store `user_id`, `login`, roles, and allowed tenants in Gradio state/session.
- If the user has one tenant, select it automatically.
- If the user has multiple tenants, show a dropdown with only allowed tenants.
- Hide or remove free-text tenant input for normal users.
- Every handler receives the selected tenant from session state and validates access before doing work.

The FastAPI API should use the same auth layer. Endpoints should not accept arbitrary `tenant_id` from unauthenticated callers. For API clients, use bearer tokens or signed session cookies and enforce `can_access_tenant()`.

## Service Changes

### Tenant resolution

Keep `TenantService.resolve(tenant_id)`, but back it with a combined settings/secrets repository:

1. Check database-backed tenant secrets/settings.
2. Fall back to environment variables for migration compatibility.
3. Fail closed if required tenant-specific credentials are missing.

### Telephony provider factory

Replace startup-time `_build_call_log_service()` provider binding with a tenant-aware factory:

- `TelephonyProviderFactory.create(tenant_config) -> TelephonyPort`
- `CallLogService` either receives the factory and resolves inside `list_calls()`/`ensure_recording()`, or a lightweight `CallLogServiceFactory` creates a service per tenant.

This is necessary because one server may serve a VoChi tenant and an MTS VATS tenant in the same process.

The mechanism should be universal at two levels:

- Adding a tenant that uses an already supported telephony provider must be data/config only: create the tenant, assign users, set `telephony_provider`, add that provider's secrets, prompts, and batch settings.
- Adding a completely new telephony service requires one new adapter that implements `TelephonyPort`, plus one provider registration entry. It must not require changes in UI handlers, analysis logic, cache logic, reports, or scheduler code.

Recommended provider registry:

```python
@dataclass(frozen=True)
class TelephonyProviderDefinition:
    key: str
    title: str
    required_secrets: tuple[str, ...]
    optional_settings: tuple[str, ...]
    factory: Callable[[TenantConfig], TelephonyPort]
```

Initial registry entries:

- `vochi`: requires `VOCHI_API_KEY`, optionally uses `VOCHI_BASE_URL`.
- `mts_vats`: requires `MTS_DOMAIN` and `MTS_API_KEY`.

New provider onboarding flow:

1. Create `calls_analyser/adapters/telephony/<provider_key>.py`.
2. Implement `TelephonyPort.list_calls()` and `TelephonyPort.get_recording()`.
3. Normalize provider payloads into `CallLogEntry`; keep provider-specific fields inside `raw`.
4. Register `TelephonyProviderDefinition(key=<provider_key>, required_secrets=..., factory=...)`.
5. Add unit tests for list parsing, filtering behavior, recording download, and error mapping.
6. Onboard a tenant by setting `tenant_settings.telephony_provider=<provider_key>` and filling required `tenant_secrets`.

Provider-specific branching should be isolated to telephony adapters and the provider factory. UI code, batch processing, reports, and email rendering should use normalized `CallLogEntry` and `RecordingHandle` fields instead of checking provider names such as `mts_vats`.

### Prompt service

Extend `PromptService` to resolve tenant prompts:

- `get_prompt(tenant_id, key, fallback_key='simple')`
- `list_templates(tenant_id)`

The default code prompts remain as fallback templates. Tenant prompts live in `tenant_prompt_templates`.

### Batch settings

Move `batch_params.json` to a fallback only. Runtime batch settings should come from `tenant_settings`.

Scheduler behavior should iterate active tenants:

1. Load tenants with `scheduler_enabled=true`.
2. For each tenant, load its scheduler filters and prompt/model settings.
3. Run batch analysis with that tenant's config.
4. Send reports to that tenant's email recipients.

## Data Isolation Rules

Every stored or fetched row must be scoped by tenant:

- analysis cache: include `tenant_id`
- usage report: filter by `tenant_id`
- recordings: include tenant folder or tenant prefix
- prompts: filter by `tenant_id`
- settings: filter by `tenant_id`
- email reports: use tenant-specific recipients

Recommended local recording path:

```text
.cache/recordings/<tenant_id>/<call_unique_id>.mp3
```

This is clearer than a flat filename prefix and reduces collision risk.

## Migration Plan

### Phase 1: Safe foundations

- Add Supabase migrations for `tenants`, `tenant_users`, `tenant_user_access`, `tenant_settings`, `tenant_prompt_templates`, `tenant_secrets`, and `analysis_results`.
- Add repository interfaces for tenant settings, tenant prompts, users, and tenant secrets.
- Keep env-variable fallback so existing deployments continue to work.
- Add tests for tenant-scoped prompt lookup and tenant access checks.

### Phase 2: Authentication

- Replace `VOCHI_UI_PASSWORD` gate with login/password.
- Bind authenticated user to allowed tenants.
- Replace free-text tenant field with allowed tenant dropdown.
- Add server-side authorization checks in every handler and API endpoint.

### Phase 3: Tenant-aware runtime config

- Add tenant-aware telephony provider factory.
- Make `CallLogService` resolve the correct provider per tenant.
- Move provider-specific recording-link behavior out of UI/report code and into telephony adapter output or `RecordingHandle`.
- Add a telephony provider registry so new provider types can be registered without changing UI, analysis, reports, or scheduler code.
- Load prompts and batch settings by tenant.
- Make email reports use tenant settings.

### Phase 4: Scheduler and reports

- Change scheduler from one default tenant to iterating enabled tenants.
- Store scheduler settings per tenant.
- Ensure report filters cannot cross tenant boundaries.
- Add audit logs for login, analysis run, prompt edit, settings edit, and report export.

### Phase 5: Cleanup

- Deprecate global `VOCHI_UI_PASSWORD`.
- Deprecate global `batch_params.json` except as local-development fallback.
- Document tenant onboarding commands and required settings.
- Add a minimal admin flow for creating tenants, users, and prompts.

## Security Notes

- Passwords must be hashed, never encrypted or stored raw.
- Tenant secrets must not be exposed to the browser.
- Supabase service-role key must stay server-side only.
- All user-provided tenant ids must be checked against `tenant_user_access`.
- Reports and exports must be generated from tenant-filtered queries only.
- Custom prompts should be versioned because changing a prompt changes cache semantics and report interpretation.

## Acceptance Criteria

The multi-tenant work is ready when:

- Two tenants can use the same deployed app instance without separate code copies.
- Each tenant can have different telephony provider credentials.
- Adding a tenant on an existing telephony provider is possible by data/config only.
- Adding a new telephony provider requires only a new `TelephonyPort` adapter, provider registration, and adapter tests; no UI, analysis, cache, report, or scheduler changes.
- Each tenant can have different prompts and batch settings.
- Users sign in with individual login/password credentials.
- A user can access only assigned tenants.
- The UI no longer trusts arbitrary tenant text input.
- Batch scheduler can run per tenant with separate filters/prompts/report recipients.
- Cache, usage, recordings, reports, prompts, and settings are tenant-scoped.
- Automated tests cover auth, tenant access, prompt isolation, provider selection, cache keys, and report filtering.
