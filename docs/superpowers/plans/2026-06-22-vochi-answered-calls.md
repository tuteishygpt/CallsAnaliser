# VoChi Answered Calls Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load answered VoChi calls across all phone numbers without changing the UI.

**Architecture:** Keep the existing `TelephonyPort` and UI contracts. Change only the VoChi adapter to paginate `/calls` with an explicitly empty `phone` parameter, retain `call_status=2`, and apply the existing `call_type` filter locally.

**Tech Stack:** Python, requests, pytest

---

## Chunk 1: Adapter behavior

### Task 1: Replace unsuccessful-call listing

**Files:**
- Modify: `calls_analyser/adapters/telephony/vochi.py`
- Test: `tests/test_vochi_adapter.py`

- [ ] **Step 1: Write failing request and filtering tests**

Assert `/calls`, `phone=""`, page size 50, pagination, exclusion of statuses
0/1, inclusion of status 2, and local filtering for call types 0/1/2.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_vochi_adapter.py -q`

Expected: failures because the adapter still requests `/unsuccessful-calls`.

- [ ] **Step 3: Implement minimal adapter change**

Request all calls, paginate using raw page length and total, map only answered
calls matching the selected call type, and keep existing error redaction.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_vochi_adapter.py -q`

Expected: all VoChi adapter tests pass.

## Chunk 2: Verification and runtime

### Task 2: Verify and restart

**Files:**
- Verify: `calls_analyser/adapters/telephony/vochi.py`
- Verify: `tests/test_vochi_adapter.py`

- [ ] **Step 1: Run focused and regression tests**

Run: `pytest tests --ignore=tests/test_integration_gemini.py --ignore=tests/test_mts_vats_adapter.py -q`

- [ ] **Step 2: Run a live VoChi smoke test**

Confirm the returned list contains only `call_status=2`.

- [ ] **Step 3: Restart Amedis/VoChi server**

Start with `DEFAULT_TENANT_ID=Amedis`,
`AMEDIS_TELEPHONY_PROVIDER=vochi`, and the API v1 base URL.

- [ ] **Step 4: Verify HTTP and browser UI**

Confirm HTTP 200 and reload the open in-app browser tab.
