# Tenant Admin Settings Design

## Goal and boundaries

Add a Gradio `Tenant Settings` tab for authenticated users with the `admin` role.
An administrator selects a tenant and can view or edit all tenant configuration,
including plaintext secret values, grouped into semantic sections.

The feature manages `tenants`, `tenant_settings`, `tenant_secrets`, and
`tenant_prompt_templates`. It does not manage users, passwords, or role
assignments. The tenant ID, database IDs, versions, and timestamps are system
identity/audit fields rather than configuration and are read-only. The user
explicitly approved keeping Tenant ID read-only on 2026-07-12. Renaming a tenant
is a separate data-migration operation because it is referenced throughout the
database; display name and status remain editable here.

## User experience

`Tenant Settings` is a top-level tab next to `Calls`, `AI Analysis`, and
`Reports`. It is hidden before login and when the live authorization source says
the user administers no tenant. The selector contains every tenant for which the
current active user currently has role `admin`, including inactive tenants. This
separate admin list allows an administrator to reactivate a tenant. Calls and
Reports continue to list only active tenants.

If there is one eligible tenant it is selected and loaded automatically. Changing
the selection replaces the form and discards unsaved values. `Reload` does the
same. `Save` validates the whole document, writes it atomically, reads it back,
and reports success without echoing secret values.

The sections are:

1. **General**: read-only tenant ID, editable display name and active/inactive
   status.
2. **Telephony**: provider, VoChi base URL/API key, MTS domain/API key. Both
   provider configurations stay visible so a switch can be prepared in one save.
3. **AI defaults**: default language/model and batch language/model.
4. **Batch processing**: enabled, batch size, and custom batch enabled.
5. **Scheduler**: enabled, cron/interval mode, cron time, interval, and call/time
   filters.
6. **Email**: recipient, sender address, and sender name.
7. **Prompt templates**: an active-template editor and read-only version history.
8. **Additional settings** and **Additional secrets**: key/value tables for every
   persisted key not represented by a typed field, including installation-specific
   email credentials. Rows can be added, changed, or deleted.

Secret values are intentionally decrypted and returned to the browser as ordinary
plaintext controls. They never appear in logs, exceptions, status messages, or
test snapshots.

## Canonical field map and round-trip rules

The admin form reads raw tenant records, not resolved global/environment
fallbacks. A missing tenant override is displayed as empty, optionally alongside
a non-editable effective-value hint for non-secret fields. This prevents a simple
load/save from materializing global defaults as tenant overrides.

| Section field | Table | Canonical key/type |
| --- | --- | --- |
| Display name, status | `tenants` | columns `display_name`, `status` |
| Provider | `tenant_settings` | `telephony_provider`: string |
| VoChi base URL | `tenant_settings` | `vochi_base_url`: string |
| VoChi API key | `tenant_secrets` | `VOCHI_API_KEY`: secret string |
| MTS domain | `tenant_secrets` | `MTS_DOMAIN`: secret string |
| MTS API key | `tenant_secrets` | `MTS_API_KEY`: secret string |
| Default language/model | `tenant_settings` | `default_language`, `default_model_key`: strings |
| Batch language/model | `tenant_settings` | `batch_language_code`, `batch_model_key`: strings |
| Batch controls | `tenant_settings` | `batch_enabled`: boolean, `batch_size`: integer, `custom_batch_enabled`: boolean |
| Scheduler controls | `tenant_settings` | `scheduler_enabled`: boolean, `scheduler_mode`: string, `scheduler_cron_time`: string, `scheduler_interval_minutes`: integer |
| Scheduler filters | `tenant_settings` | `scheduler_filters`: JSON object with `time_from`, `time_to`, `call_type` |
| Email fields | `tenant_settings` | `email_to`, `email_from`, `email_from_name`: strings |

Known legacy aliases (`TELEPHONY_PROVIDER`, `VOCHI_BASE_URL`, setting-table
`MTS_DOMAIN`, and setting-table API keys) are normalized to the canonical owner
on save and removed in the same transaction. They are excluded from additional
rows so they never appear twice. Runtime readers retain legacy-read compatibility.

An empty optional typed string deletes its tenant record, restoring fallback
behavior. Boolean and numeric controls always persist explicit values. Clearing
all scheduler filter inputs deletes `scheduler_filters`; partial filters persist
only non-empty members. Removing an additional row deletes it. Additional setting
values must be valid JSON; additional secret values are strings, and an explicit
delete action—not an empty value—removes an arbitrary secret.

## Prompt semantics

The editor selects an existing prompt family by its immutable key or creates a
new key. Version and audit metadata are read-only and generated by the database.
Saving changed title/body for an active family atomically creates version
`max(version)+1`, deactivates the previous active version, and activates the new
one. Turning a family inactive deactivates its current version without deleting
history. Reactivating a family creates a new version copied from the selected
historical content. Deleting prompt history is out of scope. At most one active
version may exist for each `(tenant_id, key)`.

## Architecture

### Shared repository composition

Create one write-capable tenant repository instance in `build_dependencies()`
and inject it into `TenantAdminSettingsService`, `TenantSettingsService`,
`TenantService`, and the tenant prompt repository used by `PromptService`. The
local/in-memory composition likewise shares one store. Consequently settings,
secrets, and prompts saved in the admin tab are visible to runtime resolution on
the next call without process restart or cache invalidation.

The repository interface supports raw document reads plus transactional saves.
The Supabase implementation calls a new SQL RPC; the in-memory implementation
applies a validated copy under a lock and swaps it only on success.

### Live authorization

Extend the auth repository/service with:

- `list_admin_tenants(user_id, include_inactive=True)`;
- `can_administer_tenant(user_id, tenant_id)`.

Both query the current data source, require an active user, require the current
access row role (case-insensitive) to be `admin`, and include inactive tenants for
admin management. Session `allowed_tenants` data controls presentation only. On
every selector refresh, load, reload, save, and prompt action, `UIHandlers` calls
the live method before touching the admin repository. Demotion, access removal,
user deactivation, or a forged tenant ID therefore takes effect immediately.

### Admin service

`TenantAdminSettingsService` owns the canonical field catalog, conversion between
records and an editable document, validation, alias normalization, encryption
boundary, and persistence orchestration. It does not decide authorization.

### UI wiring

`build_demo()` adds the tab and form. Login/selector refresh calls the live admin
tenant list and updates tab visibility/choices. Dropdown change and `Reload` call
the load handler. `Save` calls the save handler and then uses the returned
persisted document to repopulate controls. Existing Calls and Reports behavior is
unchanged.

## Secrets at rest

Introduce a secrets codec owned by the repository boundary. New and updated
secrets are encrypted with AES-256-GCM. `TENANT_SECRETS_MASTER_KEY` is an
unpadded base64url encoding of exactly 32 random bytes; every other encoding or
decoded size is rejected. A stored value has the exact form
`enc:v1:<nonce_b64url>:<ciphertext_and_tag_b64url>`, where both binary fields use
unpadded base64url, the nonce is 12 random bytes, and the final field is the
AES-GCM ciphertext with its 16-byte tag. Encryption/decryption uses the UTF-8
bytes of `tenant_id + "\\0" + key` as additional authenticated data, preventing
a ciphertext from being moved to another tenant or key. The database and
transactional RPC see only the encrypted envelope. The admin service receives
plaintext after repository decryption.

Only rows that do not start with `enc:` are treated as legacy plaintext so current
deployments remain readable. Malformed `enc:v1:` values and every unsupported
`enc:*` version fail closed with a generic error; they are never returned or
treated as legacy plaintext. Any legacy value included in a successful save is
rewritten encrypted. If the master key is missing or invalid, legacy plaintext
may still be read for compatibility, but encrypted values cannot be read and no
secret write is allowed; the UI reports a generic configuration error without
returning ciphertext or plaintext. Key rotation is a separate operational task.

The schema keeps `encrypted_value` and gains a comment documenting the envelope.

## Atomic persistence and database invariants

Add a Supabase migration with one service-role-only RPC that accepts a validated
tenant document containing profile changes, setting/secret upserts and deletes,
and prompt operations. The RPC:

- executes the complete save in one database transaction;
- locks the tenant and affected prompt families;
- applies canonical upserts/deletes and legacy-alias cleanup;
- derives each new prompt version as `max(version)+1` while locked;
- deactivates the former version and inserts the new version atomically;
- records the already-authorized user's UUID in `created_by` for audit only;
- rolls back every change on any failure.

Before adding the partial unique index, the migration deterministically reconciles
existing duplicates: for each `(tenant_id, key)`, it keeps active the row with the
highest version, breaking ties by newest `updated_at`, then newest `created_at`,
then greatest UUID; all other active rows are set inactive in the same migration
transaction. The migration emits a count of reconciled rows for deployment audit
and aborts if the post-reconciliation invariant still fails. A partial unique
index then enforces one active row per `(tenant_id, key)` in
`tenant_prompt_templates`. No prompt history is deleted. The RPC is not an
authorization boundary—the server uses its service role—so the handler's live
check remains mandatory.

## Validation and errors

Validation completes before the RPC call and includes non-empty display name;
tenant status strictly in `{active, inactive}`; known provider and scheduler
choices; positive batch size and interval; valid
`HH:MM` times; unique additional keys with no typed/alias collision; valid JSON
setting values; and non-empty prompt key/title/body. Conditional telephony
requirements are checked against the effective post-save configuration.

Validation errors identify section/field and perform no writes. Authorization
errors are generic and perform no admin-repository call. Persistence and crypto
errors are generic and never interpolate payloads. An empty tenant selection
keeps the form disabled.

## Testing

Implementation follows test-driven development.

- Auth tests cover active admin, inactive tenant, operator, demotion, access
  removal, inactive user, and forged tenant behavior.
- Codec tests cover encrypted round trips, nonce uniqueness, legacy plaintext
  reads/migration, missing/wrong keys, and absence of values in errors/logs.
- Repository/RPC tests cover raw list/upsert/delete behavior, alias cleanup,
  all-or-nothing rollback, concurrent prompt edits, generated versions, audit
  user, and the unique-active invariant.
- Service tests cover typed/raw document assembly, raw-versus-effective display,
  blank deletion/fallback, arbitrary JSON and secrets, validation, and prompt
  operations.
- Handler tests prove every operation performs a live role check and that denied
  requests make zero admin-repository calls.
- Wiring tests prove the admin service and runtime resolvers share a store, and
  saved settings, secrets, and prompts are observed immediately in both Supabase
  and in-memory compositions.
- Layout/login tests cover visibility, admin-only choices, inactive tenant
  reactivation, event signatures, and unchanged Calls/Reports selectors.

The complete existing suite is run for regression coverage.

## Acceptance criteria

- The tab and selector reflect the current live admin roles and include inactive
  administered tenants; operators cannot invoke its operations.
- Selecting a tenant loads every raw known and arbitrary setting, decrypted
  plaintext secret, and prompt family/history without duplicating aliases.
- One save atomically persists all profile, setting, secret, and prompt changes;
  failures leave the prior document unchanged.
- Blank/delete behavior restores runtime fallbacks as specified.
- Saved configuration is visible to runtime services without restart.
- Prompt history is preserved and exactly one active version exists per family.
- No secret value appears in logs, errors, status messages, or stored plaintext
  after a successful save.
