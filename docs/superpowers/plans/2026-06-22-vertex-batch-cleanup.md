# Vertex Batch Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee best-effort GCS staging cleanup for every Vertex Batch lifecycle outcome.

**Architecture:** Keep the existing runner API and cleanup helper. Wrap the single-batch lifecycle in `try/finally` so the generated prefix is cleaned without masking the original error.

**Tech Stack:** Python, pytest, google-genai adapter abstractions

---

## Chunk 1: Regression coverage and fix

### Task 1: Add lifecycle cleanup tests

**Files:**
- Create: `tests/test_gemini_batch.py`
- Test: `tests/test_gemini_batch.py`

- [ ] Write parameterized tests that bypass `__init__`, stub lifecycle methods,
      and assert `_cleanup_gcs` runs after success and each failure boundary.
- [ ] Run `python -m pytest tests/test_gemini_batch.py -q` and confirm the
      failure-path test fails because cleanup is not called.

### Task 2: Guarantee cleanup

**Files:**
- Modify: `calls_analyser/services/gemini_batch.py:177`
- Test: `tests/test_gemini_batch.py`

- [ ] Wrap the existing `_run_single_batch` lifecycle in `try/finally`.
- [ ] Remove the success-only cleanup call from the `try` body.
- [ ] Run `python -m pytest tests/test_gemini_batch.py -q`.
- [ ] Run `python -m pytest -q`.
