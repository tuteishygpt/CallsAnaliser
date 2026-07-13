# Tenant Admin Settings Simplified MVP Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the simplified admin-only Tenant Settings tab with focused persistence, shared runtime state, encrypted changed secrets, and read-only active prompts.

**Architecture:** One shared tenant repository exposes raw reads and focused profile/setting/secret operations. The admin service validates the complete typed form, computes field changes, performs the accepted sequential writes, and reloads persisted state. UI handlers perform live admin checks before every admin service call.

**Tech Stack:** Python, Gradio, Supabase client, AES-256-GCM via `cryptography`, pytest.

---

## Task 1: Focused repository and simplified admin service

**Files:**
- Modify: `calls_analyser/adapters/storage/supabase_tenant.py`
- Rewrite: `calls_analyser/services/tenant_admin_settings.py`
- Keep/refine: `calls_analyser/services/tenant_secret_codec.py`
- Remove: `docs/supabase/migrations/20260712_tenant_admin_settings.sql`
- Revert deferred changes: `docs/supabase/multi_tenant_schema.sql`
- Rewrite: `tests/test_tenant_admin_settings.py`
- Modify: `tests/test_supabase_tenant_repositories.py`
- Modify: `tests/test_multi_tenant_schema.py`

- [ ] Write failing tests for focused raw reads, profile update, individual typed setting/secret upsert/delete, preservation of arbitrary rows, field-level alias cleanup, changed-secret-only encryption, and non-secret saves without a master key.
- [ ] Run focused tests and confirm expected failures.
- [ ] Replace full-document RPC persistence with focused repository operations in both Supabase and in-memory implementations.
- [ ] Simplify `TenantAdminSettingsService`: raw missing fields remain empty; prompts are active/read-only; no additional rows; validate before first write; compute changes; apply accepted sequential operations; reload persisted state.
- [ ] Add provider-specific effective post-save validation and generic crypto/persistence failures.
- [ ] Remove migration/RPC/prompt unique-index work and obsolete tests.
- [ ] Run focused tests to green.

## Task 2: Simplified handlers and Gradio form

**Files:**
- Modify: `calls_analyser/ui/handlers.py`
- Modify: `calls_analyser/ui/layout.py`
- Rewrite: `tests/test_ui_tenant_admin_settings.py`
- Add/modify layout tests as needed.

- [ ] Write failing tests for live refresh/load/reload/save authorization, zero service calls on denial, generic failures, persisted readback, single-tenant auto-load, empty-selection disabled state, and read-only active prompts.
- [ ] Run focused tests and confirm expected failures.
- [ ] Remove prompt editing/history and Additional settings/secrets controls and mappings.
- [ ] Keep only typed editable fields plus a read-only active-prompts output; define Save inputs explicitly.
- [ ] Disable the form and actions for empty selection; retain live inactive-admin choices and existing Calls/Reports behavior.
- [ ] Run focused tests to green.

## Task 3: Shared composition and runtime visibility

**Files:**
- Modify: `calls_analyser/ui/dependencies.py`
- Modify only as needed: `calls_analyser/services/tenant_settings.py`, `calls_analyser/services/tenant.py`, `calls_analyser/services/prompt.py`
- Modify: `tests/test_dependency_wiring_auth_settings.py`
- Add: `tests/test_tenant_admin_dependency_wiring.py`

- [ ] Write failing tests proving one repository instance is shared by admin, runtime settings, telephony, and prompt reads for Supabase and in-memory composition.
- [ ] Run tests and confirm expected failures.
- [ ] Construct the repository with the codec through a public constructor; remove private `_codec` mutation.
- [ ] Ensure successful focused saves are immediately visible to runtime services.
- [ ] Run focused tests to green.

## Task 4: Verification

- [ ] Run all tenant-admin/auth/codec/repository/wiring/layout tests.
- [ ] Run `pytest -q` for regression coverage.
- [ ] Run `git diff --check` and inspect the complete diff for deferred features or secret leakage.
