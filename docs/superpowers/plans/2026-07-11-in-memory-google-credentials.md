# In-memory Google Credentials Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Authenticate Google clients from the HF base64 secret without creating credential files.

**Architecture:** A single cached loader creates an in-memory service-account credentials object. Precedence is API key, valid B64 service account, then local ADC. Gemini and GCS constructors receive the same explicit object in B64 mode and no explicit object in ADC mode.

**Tech Stack:** Python, google-auth, google-genai, google-cloud-storage, pytest

---

## Chunk 1: Credentials loader and client wiring

### Task 1: Add the in-memory loader

**Files:**
- Create: `calls_analyser/google_credentials.py`
- Create: `tests/test_google_credentials.py`

- [ ] Write tests for base64 decoding, process-level caching, sanitized invalid-input handling, and absence of filesystem writes.
- [ ] Run the focused tests and confirm they fail because the loader does not exist.
- [ ] Implement the minimal cached loader using `Credentials.from_service_account_info`.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Wire clients and remove temp files

**Files:**
- Modify: `app.py`
- Modify: `calls_analyser/runner.py`
- Modify: `calls_analyser/adapters/ai/gemini.py`
- Modify: `calls_analyser/services/gemini_batch.py`
- Modify: `calls_analyser/ui/dependencies.py`
- Test: `tests/test_gemini_adapter.py`
- Test: `tests/test_gemini_batch.py`
- Test: `tests/test_google_credentials.py`

- [ ] Write failing tests that require the same in-memory credentials object in Gemini and GCS clients, omit explicit credentials for ADC, preserve injected adapter factories, and recognize B64-only UI configuration.
- [ ] Add static regression assertions that `app.py` and `runner.py` contain no service-account tempfile bootstrap or credential-related `tempfile` usage.
- [ ] Confirm the new tests fail for the expected missing behavior.
- [ ] Remove both temporary-file bootstrap blocks and pass the cached credentials object to clients.
- [ ] Update model registration and help text to recognize API key, B64 credentials, or ADC without exposing secret data.
- [ ] Run focused and full tests.
- [ ] Verify the active source branch and deployment target before committing or pushing.
- [ ] Deploy the verified source commit to `archivartaunik/tst4`; confirm B64-only client initialization, no `GOOGLE_APPLICATION_CREDENTIALS`, no credential-file log, healthy runtime, and HTTP.
