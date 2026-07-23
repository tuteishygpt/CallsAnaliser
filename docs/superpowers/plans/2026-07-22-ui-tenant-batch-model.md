# UI Tenant Batch Model Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Gradio mass analysis prefer the selected tenant's batch model and language, with `models/gemini-3.1-flash-lite` as the global fallback.

**Architecture:** Resolve optional tenant runtime settings once inside the mass-analysis handler, derive effective model/language values with explicit fallbacks, validate the model before fetching calls, and pass those values through the existing sequential loop. Keep scheduler and direct single-call behavior unchanged.

**Tech Stack:** Python 3.13, Gradio, pytest

---

## Chunk 1: Tenant-aware UI mass analysis

### Task 1: Add failing regression coverage

**Files:**
- Modify: `tests/test_app_batch.py`
- Modify: `tests/test_analysis_service.py`
- Test: `tests/test_app_batch.py`
- Test: `tests/test_analysis_service.py`

- [ ] Add `_RecordingTenantSettingsService`, patched directly onto `app.handlers.deps.tenant_settings_service` because `_sync_test_overrides()` does not copy it. Its `resolve(tenant_id)` returns a supplied namespace or raises a supplied exception and records every tenant id.
- [ ] Preserve `_StubAnalysisService.calls` as existing `(unique_id, tenant, options)` tuples and add a separate `languages` list populated by `analyze_call`; extend `_StubCallLogService` with a `list_calls_count` increment so language and pre-fetch validation assertions are executable without breaking existing consumers.
- [ ] Add `test_ui_mass_analyze_uses_tenant_batch_model_and_language_once`, configuring registry keys `global-model` and `tenant-model`, and assert both processed calls receive `tenant-model`, `Language.BELARUSIAN`, and exactly one settings resolution.
- [ ] Run `python -m pytest tests/test_app_batch.py::test_ui_mass_analyze_uses_tenant_batch_model_and_language_once -q`; expect FAIL because current calls receive the global model/language.
- [ ] Add `test_ui_mass_analyze_rejects_unregistered_tenant_model_before_listing_calls`; require the exact message `## ❌ Configured batch model 'missing-model' is not available.` and zero `list_calls`/analysis calls.
- [ ] Run that exact node; expect FAIL because current code validates only the global model and proceeds.
- [ ] Add exact fallback nodes `test_ui_mass_analyze_uses_global_runtime_values_without_settings_service`, `test_ui_mass_analyze_uses_global_runtime_values_for_blank_settings`, `test_ui_mass_analyze_uses_global_runtime_values_when_settings_resolution_fails`, `test_ui_mass_analyze_uses_global_language_for_invalid_tenant_language`, and parametrized `test_ui_mass_analyze_maps_tenant_auto_language_to_auto` with `"auto"`/`"default"`. Run these nodes before implementation; expect the exception/unavailable cases to retain global behavior, but blank/invalid/auto/default tenant-value assertions to fail until runtime resolution exists.
- [ ] Add characterization tests `test_analysis_service_cache_separates_registry_model_keys_with_same_provider_name` and `test_direct_analysis_keeps_dropdown_model_when_tenant_batch_model_differs`. The cache test uses two model stubs with the same `provider_name`, provider-owned result model strings, and asserts each provider runs once and both exact cache tuples differ at model-key position. The direct test patches conflicting tenant batch settings and asserts dropdown `model_pref` reaches `AnalysisOptions` without resolving tenant batch settings.
- [ ] Run those two exact nodes and require PASS before implementation; they document existing boundaries rather than the missing behavior.

### Task 2: Resolve effective tenant model and language

**Files:**
- Modify: `calls_analyser/ui/handlers.py`
- Test: `tests/test_app_batch.py`

- [ ] Add `_resolve_tenant_batch_settings(tenant)` that calls `tenant_settings_service.resolve(tenant.tenant_id)` once and returns `None` for an absent service, a non-callable `resolve`, or a resolution exception.
- [ ] Add focused value helpers: a non-empty `batch_model_key` overrides `deps.batch_model_key`; language values `"auto"` and `"default"` map to `config.Language.AUTO`; other valid codes convert through `config.Language`; blank/invalid values fall back to `deps.batch_language`.
- [ ] In `_run_mass_analyze`, resolve the tenant before model validation, derive the effective model/language, and validate the effective model with `ai_registry.get()` before fetching calls.
- [ ] Return exactly `## ❌ Configured batch model '<key>' is not available.` when a non-empty effective model is not registered.
- [ ] Pass the effective model and language into `AnalysisOptions` and `analysis_service.analyze_call` for every call instead of reading the global dependencies inside the loop.
- [ ] Run `python -m pytest tests/test_app_batch.py tests/test_analysis_service.py -q` and confirm all focused tests pass.
- [ ] Rerun all fallback nodes added in Task 1 and require PASS; assert the expected global or `Language.AUTO` value and one-or-zero resolver calls as appropriate.

### Task 3: Change the global fallback model

**Files:**
- Modify: `calls_analyser/config.py`
- Modify: `tests/test_google_credentials.py`

- [ ] Add an assertion that `config.BATCH_MODEL_KEY == "models/gemini-3.1-flash-lite"`.
- [ ] Name it `test_batch_model_default_is_gemini_3_1_flash_lite` and run `python -m pytest tests/test_google_credentials.py::test_batch_model_default_is_gemini_3_1_flash_lite -q`; confirm it fails against the old value.
- [ ] Change the global `BATCH_MODEL_KEY` constant to `models/gemini-3.1-flash-lite`.
- [ ] Run `python -m pytest tests/test_google_credentials.py -q` and confirm it passes.

### Task 4: Full verification

**Files:**
- Verify: `calls_analyser/config.py`
- Verify: `calls_analyser/ui/handlers.py`
- Verify: `tests/test_app_batch.py`
- Verify: `tests/test_analysis_service.py`
- Verify: `tests/test_google_credentials.py`

- [ ] Run `python -m pytest -q` and require zero failures.
- [ ] Explicitly run `python -m pytest tests/test_app_scheduler.py tests/test_scheduler_service.py tests/test_runner_email.py -q` to preserve existing scheduler/runner behavior.
- [ ] Run `git diff --check` and inspect `git diff` to confirm scheduler/runner code is unchanged and unrelated local log files remain untracked.
