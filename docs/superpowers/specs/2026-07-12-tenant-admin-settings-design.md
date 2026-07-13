# Tenant Admin Settings — Simplified MVP Design

## Goal

Add a Gradio `Tenant Settings` tab where an authenticated user with the current
`admin` role can view and edit the most important configuration of an existing
tenant.

The MVP manages:

- tenant display name and active/inactive status;
- known telephony settings and secrets;
- known AI, batch, scheduler, and email settings;
- read-only active prompt templates.

It does not manage users, passwords, roles, tenant IDs, arbitrary key/value
records, or prompt history. Tenant creation and tenant renaming remain separate
operations.

The purpose of this reduced scope is to provide a useful administration screen
without introducing a large transactional RPC, prompt-version workflow, or
schema migration before those capabilities are actually needed.

## User experience

`Tenant Settings` is a top-level tab next to `Calls`, `AI Analysis`, and
`Reports`. It is hidden before login and hidden when the current user
administers no tenant.

The tenant selector is populated from the live authorization source and includes
inactive tenants so an administrator can reactivate them. Calls and Reports
continue to show only active tenants.

If exactly one tenant is available, it is selected and loaded automatically.
Changing the selection or clicking `Reload` replaces the form and discards
unsaved values.

`Save` validates the form, writes the changed values, reloads the persisted
document, and displays a generic success or failure message. Status messages
never contain secret values.

The form contains these sections:

1. **General**: read-only tenant ID, editable display name and status.
2. **Telephony**: provider, VoChi base URL/API key, MTS domain/API key. Both
   provider configurations remain visible.
3. **AI defaults**: default language/model and batch language/model.
4. **Batch processing**: enabled, batch size, and custom batch enabled.
5. **Scheduler**: enabled, cron/interval mode, cron time, interval, and call/time
   filters.
6. **Email**: recipient, sender address, and sender name.
7. **Prompt templates**: read-only list of active tenant templates. Prompt
   editing and version history are deferred.

An empty tenant selection keeps the form disabled.

## Canonical fields

The form reads raw tenant records, not resolved environment or global fallbacks.
A missing tenant override appears empty. This prevents a load/save cycle from
copying global defaults into tenant-specific records.

| Form field | Storage | Canonical key/type |
| --- | --- | --- |
| Display name, status | `tenants` | `display_name`, `status` |
| Provider | `tenant_settings` | `telephony_provider`: string |
| VoChi base URL | `tenant_settings` | `vochi_base_url`: string |
| VoChi API key | `tenant_secrets` | `VOCHI_API_KEY`: secret string |
| MTS domain | `tenant_secrets` | `MTS_DOMAIN`: secret string |
| MTS API key | `tenant_secrets` | `MTS_API_KEY`: secret string |
| Default language/model | `tenant_settings` | `default_language`, `default_model_key`: strings |
| Batch language/model | `tenant_settings` | `batch_language_code`, `batch_model_key`: strings |
| Batch controls | `tenant_settings` | `batch_enabled`: boolean, `batch_size`: integer, `custom_batch_enabled`: boolean |
| Scheduler controls | `tenant_settings` | `scheduler_enabled`: boolean, `scheduler_mode`: string, `scheduler_cron_time`: string, `scheduler_interval_minutes`: integer |
| Scheduler filters | `tenant_settings` | `scheduler_filters`: JSON object |
| Email fields | `tenant_settings` | `email_to`, `email_from`, `email_from_name`: strings |

Known legacy aliases remain readable at runtime. When the corresponding typed
field is saved, the canonical key is written and that field’s known legacy alias
is removed. A separate bulk cleanup of all legacy records is out of scope.

Empty optional strings delete their tenant override and restore fallback
behavior. Boolean and numeric controls persist explicit values. If every
scheduler filter is empty, `scheduler_filters` is deleted; otherwise only
non-empty members are stored.

## Architecture

### Shared repository

Create one tenant configuration repository instance in `build_dependencies()`.
Inject it into:

- `TenantAdminSettingsService` for raw admin reads and writes;
- `TenantSettingsService` for runtime settings;
- `TenantService` for telephony configuration;
- the prompt repository used by `PromptService` for read-only prompt display.

The Supabase composition and the local/in-memory composition each share one
instance. A successful save is therefore visible to runtime services on the next
read without restarting the process or invalidating a cache.

The repository exposes focused operations rather than one full-document RPC:

- read tenant profile, settings, secrets, and active prompts;
- update tenant profile;
- upsert or delete individual settings;
- upsert or delete individual secrets.

The in-memory implementation applies the same operations under a lock.

### Live authorization

Extend the auth service with:

- `list_admin_tenants(user_id, include_inactive=True)`;
- `can_administer_tenant(user_id, tenant_id)`.

Both methods query the current repository state, require an active user, and
require a current access row whose role equals `admin` case-insensitively.

Session `allowed_tenants` is presentation data only. Selector refresh, load,
reload, and save each perform a live authorization check before calling the
admin repository. Demotion, access removal, user deactivation, and forged tenant
IDs therefore take effect immediately.

### Admin service

`TenantAdminSettingsService` owns:

- the canonical typed-field catalog;
- conversion between raw records and the editable document;
- validation;
- blank-value deletion rules;
- field-level legacy alias normalization;
- persistence orchestration.

It does not decide authorization and does not edit prompt templates.

### UI wiring

`build_demo()` adds the tab and typed form. Login and selector refresh load the
live admin tenant list. Selector change and `Reload` call the load handler.
`Save` calls the save handler and repopulates the controls from the persisted
readback.

Existing Calls and Reports selectors are unchanged.

## Secrets at rest

New and changed tenant secrets are encrypted at the repository boundary with
AES-256-GCM.

`TENANT_SECRETS_MASTER_KEY` is an unpadded base64url encoding of exactly 32
random bytes. A stored encrypted value has this form:

```text
enc:v1:<nonce_b64url>:<ciphertext_and_tag_b64url>
```

The nonce is 12 random bytes. Encryption uses the UTF-8 bytes of
`tenant_id + "\0" + key` as additional authenticated data so ciphertext cannot
be moved to another tenant or key.

Values that do not start with `enc:` are treated as legacy plaintext and remain
readable. A legacy value is rewritten encrypted only when that secret is saved.
Malformed or unsupported `enc:*` values fail closed with a generic error.

If the master key is missing or invalid:

- legacy plaintext can still be read for compatibility;
- encrypted values cannot be read;
- secret writes are rejected;
- non-secret settings may still be saved;
- UI errors remain generic and never include stored or submitted values.

Key rotation is out of scope.

## Persistence and database changes

The MVP uses the existing `tenants`, `tenant_settings`, `tenant_secrets`, and
`tenant_prompt_templates` tables. No schema migration or new SQL RPC is required.

Save performs a short sequence of repository operations:

1. validate the complete form before any write;
2. update the tenant profile if changed;
3. upsert or delete changed settings;
4. encrypt and upsert or delete changed secrets;
5. read the document back.

This is intentionally simpler than a cross-table transaction. A database or
network failure can leave some fields saved while later fields fail. The UI
reports a generic failure and `Reload` shows the actual persisted state.

This trade-off is accepted for the MVP because tenant administration is expected
to be infrequent and performed by few administrators. If partial saves become a
real operational problem, a transactional RPC can be added later without
changing the UI document format.

Prompt editing, prompt-version locking, duplicate reconciliation, and the
partial unique active-prompt index are deferred to that later phase.

## Validation and errors

Validation completes before the first write and includes:

- non-empty display name;
- tenant status in `{active, inactive}`;
- known provider and scheduler choices;
- positive batch size and scheduler interval;
- valid `HH:MM` times;
- provider-specific required fields based on the effective post-save values.

Validation errors identify the section and field. Authorization failures are
generic and make no admin-repository call. Persistence and crypto errors are
generic and never interpolate form values, stored values, or ciphertext.

## Testing

Implementation follows test-driven development.

- Auth tests cover active admin, inactive tenant, operator, demotion, access
  removal, inactive user, and forged tenant behavior.
- Codec tests cover encrypted round trips, nonce uniqueness, legacy plaintext,
  missing/wrong keys, malformed envelopes, and value-free errors.
- Repository tests cover raw reads, typed upserts/deletes, field-level alias
  cleanup, encryption at rest, and shared in-memory state.
- Service tests cover raw document assembly, blank deletion, validation,
  scheduler filter handling, and read-only prompts.
- Handler tests prove every operation performs a live role check and denied
  requests make zero admin-service calls.
- Wiring tests prove admin and runtime services share one repository and observe
  successful saves immediately.
- Layout/login tests cover tab visibility, inactive administered tenants,
  single-tenant auto-selection, event signatures, and unchanged Calls/Reports
  selectors.
- The complete existing suite runs for regression coverage.

## Acceptance criteria

- The tab is visible only to a currently active administrator and includes
  inactive administered tenants.
- Every load, reload, and save performs a live admin-role check.
- Selecting a tenant loads its raw known settings, plaintext secret controls,
  and active prompts without materializing global fallbacks.
- Saving validates first, persists canonical typed fields, and reloads the
  actual stored state.
- Blank typed strings delete tenant overrides as specified.
- New or changed secrets are never stored as plaintext after a successful secret
  write.
- Saved configuration is visible to runtime services without restart.
- Prompt templates remain read-only in the MVP.
- No secret value appears in logs, exceptions, status messages, or test
  snapshots.

## Deferred enhancements

The following require a separate design and are not part of this MVP:

- arbitrary additional settings and secrets;
- prompt creation, editing, activation, and version history;
- one-active-prompt database invariant and duplicate reconciliation;
- one cross-table transactional save RPC;
- optimistic locking or conflict detection for concurrent administrators;
- bulk migration of every legacy alias and plaintext secret;
- tenant creation, ID changes, user management, and role management.
