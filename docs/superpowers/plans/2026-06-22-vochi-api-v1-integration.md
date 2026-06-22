# VoChi API v1 Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the VoChi adapter, tenant configuration, and UI links from the legacy calllogs API to `bot.vochi.by/api/v1`.

**Architecture:** Preserve the existing `TelephonyPort` contract and replace only the VoChi adapter's HTTP protocol. Tenant configuration supplies the new API key, while the adapter owns pagination, payload mapping, recording metadata lookup, and S3 download.

**Tech Stack:** Python 3, requests, Pydantic, pytest, Gradio

---

## Chunk 1: Configuration and VoChi adapter

### Task 1: Migrate tenant configuration

**Files:**
- Modify: `calls_analyser/services/tenant.py`
- Modify: `calls_analyser/ui/config.py`
- Modify: `calls_analyser/ui/dependencies.py`
- Test: `tests/test_tenant_service.py`

- [ ] **Step 1: Write failing tenant tests**

Add tests proving that VoChi uses `VOCHI_API_KEY`, normalizes
`https://bot.vochi.by/api/v1`, no longer requires client ID/bearer, and builds
`/recording/{unique_id}` as the non-secret fallback URL.

- [ ] **Step 2: Run tenant tests and verify RED**

Run: `pytest tests/test_tenant_service.py -q`

Expected: failures because `vochi_api_key` and new configuration behavior do
not exist.

- [ ] **Step 3: Implement minimal tenant configuration**

Add `vochi_api_key` to `TenantConfig`, resolve `VOCHI_API_KEY`, change the
default base URL, preserve MTS behavior, and pass the API key into the VoChi
adapter.

- [ ] **Step 4: Run tenant tests and verify GREEN**

Run: `pytest tests/test_tenant_service.py -q`

Expected: all tenant tests pass.

### Task 2: Replace VoChi call listing

**Files:**
- Modify: `calls_analyser/adapters/telephony/vochi.py`
- Test: `tests/test_vochi_adapter.py`

- [ ] **Step 1: Write failing listing tests**

Cover `direction=all/incoming/outgoing`, no HTTP request for internal calls,
100-item pagination, mapping, malformed optional fields, skipped missing IDs,
invalid payloads, HTTP failures, and secret redaction.

- [ ] **Step 2: Run listing tests and verify RED**

Run: `pytest tests/test_vochi_adapter.py -q`

Expected: failures against legacy `/calllogs` behavior.

- [ ] **Step 3: Implement minimal listing behavior**

Request `/unsuccessful-calls`, paginate defensively, map new snake-case fields,
and translate provider/request errors to secret-safe `TelephonyError`.

- [ ] **Step 4: Run listing tests and verify GREEN**

Run: `pytest tests/test_vochi_adapter.py -q`

Expected: listing tests pass.

### Task 3: Replace VoChi recording download

**Files:**
- Modify: `calls_analyser/adapters/telephony/vochi.py`
- Test: `tests/test_vochi_adapter.py`

- [ ] **Step 1: Write failing recording tests**

Cover metadata lookup parameters, S3 download without API credentials,
permanent source URL, content type, fallback source URI, missing
`download_url`, malformed metadata, download errors, and secret redaction.

- [ ] **Step 2: Run recording tests and verify RED**

Run: `pytest tests/test_vochi_adapter.py -q`

Expected: failures because the adapter still downloads legacy URLs directly.

- [ ] **Step 3: Implement minimal recording behavior**

Fetch `/recording` metadata, validate `download_url`, download bytes, and
return the permanent URL and response content type.

- [ ] **Step 4: Run recording tests and verify GREEN**

Run: `pytest tests/test_vochi_adapter.py -q`

Expected: all VoChi adapter tests pass.

## Chunk 2: UI links, documentation, and verification

### Task 4: Use permanent VoChi recording links in UI

**Files:**
- Modify: `calls_analyser/ui/handlers.py`
- Test: `tests/test_app_batch.py`

- [ ] **Step 1: Write a failing UI link test**

Prove that a VoChi result row uses `entry.raw["recording_url"]` and does not
construct a legacy `/calllogs` URL.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/test_app_batch.py -q`

Expected: the new assertion fails against legacy link construction.

- [ ] **Step 3: Implement the link selection**

Use raw permanent URL first and `tenant.recording_url()` as fallback.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `pytest tests/test_app_batch.py -q`

Expected: all batch/UI tests pass.

### Task 5: Update configuration documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Document the new VoChi environment variables and behavior**

Replace legacy client ID/bearer references with `VOCHI_API_KEY`, document the
new base URL, and state that the list contains unsuccessful external calls.

- [ ] **Step 2: Check documentation diff**

Run: `git diff --check`

Expected: no whitespace errors.

### Task 6: Full verification

**Files:**
- Verify all changed files

- [ ] **Step 1: Run focused tests**

Run:
`pytest tests/test_vochi_adapter.py tests/test_tenant_service.py tests/test_app_batch.py -q`

Expected: all focused tests pass.

- [ ] **Step 2: Run full test suite**

Run: `pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 3: Inspect final diff and secret safety**

Run:
`git diff --check`

Run:
`rg -n "vchi_[A-Za-z0-9]+" . --glob "!.git/**"`

Expected: no whitespace errors and no committed VoChi API key.
