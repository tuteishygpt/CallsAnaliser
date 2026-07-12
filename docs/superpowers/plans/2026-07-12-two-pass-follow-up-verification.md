# Two-Pass Follow-Up Verification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one tenant-aware two-pass follow-up decision pipeline shared by sequential UI batches and scheduled Vertex batches.

**Architecture:** A framework-independent `BatchAnalysisOrchestrator` owns strict decision parsing, candidate selection, bounded retry, final decision rules, counters, progress events, and ordered results. `SequentialBatchExecutor` and `VertexBatchExecutor` own execution/cache/usage concerns and expose a narrow `record_validation(results)` acknowledgement so the orchestrator can persist `decision_valid` without moving parsing into executors. UI and scheduler remain thin adapters that discover calls and render/send the canonical row projection.

**Tech Stack:** Python 3.13, dataclasses, Pydantic domain entries, pandas, pytest, existing `AnalysisService`, `VertexBatchRunner`, `PromptService`, tenant settings, and cache adapters.

---

## Chunk 1: Core decision pipeline

### Task 1: Strict and compatibility follow-up parsing

**Files:**
- Create: `calls_analyser/services/follow_up.py`
- Modify: `calls_analyser/services/batch_results.py`
- Test: `tests/test_follow_up.py`

- [ ] **Step 1: Write strict-parser tests** for valid JSON booleans, fenced JSON, string/number booleans, missing fields, empty reasons, arrays, and prose.
- [ ] **Step 2: Run `pytest tests/test_follow_up.py -v`** and verify import/behavior failures prove the parser is absent.
- [ ] **Step 3: Implement `FollowUpDecision`, `FollowUpDecisionParser.parse_strict`, and `parse_compatibility`**. Strict parsing must require an object, exact `bool`, and non-empty string reason; compatibility additionally accepts historical `Needs follow-up: Yes/No` text.
- [ ] **Step 4: Run `pytest tests/test_follow_up.py -v`** and verify all parser tests pass.
- [ ] **Step 5: Refactor `batch_results.parse_follow_up_fields`** to delegate to the shared compatibility parser without changing its public tuple contract yet.
- [ ] **Step 6: Run `pytest tests/test_follow_up.py tests/test_runner_email.py tests/test_app_batch.py -q`** and preserve existing behavior.

### Task 2: Orchestrator data contracts and input validation

**Files:**
- Create: `calls_analyser/services/batch_orchestrator.py`
- Test: `tests/test_batch_orchestrator.py`

- [ ] **Step 1: Write failing tests** for duplicate input IDs, input ordering, missing executor IDs, and unrequested executor IDs.
- [ ] **Step 2: Run `pytest tests/test_batch_orchestrator.py -v`** and verify expected failures.
- [ ] **Step 3: Add dataclasses** `RoundSpec`, `RoundExecutionResult`, `BatchItemResult`, `BatchRunResult`, and `BatchProgressEvent`, plus the `BatchRoundExecutor` protocol and closed status validation. `RoundExecutionResult` carries its cache key/identity; the executor protocol includes `record_validation(round_spec, validated_results)` to rewrite cached metadata after parsing while keeping parsing in the orchestrator.
- [ ] **Step 4: Add the minimal orchestrator skeleton** that rejects duplicate IDs before resolution/execution, synthesizes `missing`, rejects extra IDs, and preserves input order.
- [ ] **Step 5: Run `pytest tests/test_batch_orchestrator.py -v`** and verify green.
- [ ] **Step 6: Commit** with `git add calls_analyser/services/batch_orchestrator.py tests/test_batch_orchestrator.py && git commit -m "feat: add batch orchestration contracts"`.

### Task 3: Primary decisions, verification matrix, and counters

**Files:**
- Modify: `calls_analyser/services/batch_orchestrator.py`
- Test: `tests/test_batch_orchestrator.py`

- [ ] **Step 1: Add failing parameterized tests** for every row in the design decision matrix, including shadow disagreement and safe fallback.
- [ ] **Step 2: Add failing tests** proving only strict valid primary positives are verification candidates, original entries are passed to round two, and primary raw text is absent from `RoundSpec`.
- [ ] **Step 3: Run the focused tests** and confirm matrix/candidate failures.
- [ ] **Step 4: Implement primary parsing, verification selection, configuration-error fallback, and final decision application** with `off`, `shadow`, and `enforce` semantics.
- [ ] **Step 5: Add failing mixed-batch counter tests** for all seven aggregate counters.
- [ ] **Step 6: Implement counters from finalized item state** and run the focused suite green.
- [ ] **Step 7: Write failing progress tests** for ordered `primary_started`, per-item `primary_complete`, `verification_started`, per-item `verification_complete`, and `run_complete` events; pending events must expose no final decision, callback exceptions must be logged/ignored, and event order must be deterministic for synchronous execution.
- [ ] **Step 8: Implement progress emission** and run `pytest tests/test_batch_orchestrator.py -v`; expect all matrix, counter, and progress tests to pass.
- [ ] **Step 9: Commit** with `git add calls_analyser/services/batch_orchestrator.py tests/test_batch_orchestrator.py && git commit -m "feat: apply two-pass decision policy"`.

### Task 4: One bounded cache-bypassing retry

**Files:**
- Modify: `calls_analyser/services/batch_orchestrator.py`
- Test: `tests/test_batch_orchestrator.py`

- [ ] **Step 1: Write failing tests** for primary and verification `error`, `missing`, and invalid responses, asserting exactly one retry call with only affected entries and `bypass_cache=True`.
- [ ] **Step 2: Write failing tests** proving legacy cached primary output is accepted only in `off`, but retried under strict modes.
- [ ] **Step 3: Implement a shared per-round execute/validate/retry helper** with no retry loop and merge successful unaffected results.
- [ ] **Step 4: Add failing persistence acknowledgement tests** asserting invalid raw output is stored with `decision_valid=false`, a valid response has `true`, and a successful bypass retry overwrites invalid raw text/metadata at the same cache identity; run `pytest tests/test_batch_orchestrator.py -v` and verify the missing acknowledgement calls fail.
- [ ] **Step 5: Call `record_validation` after every parse** and implement the minimal executor acknowledgement behavior required by the fakes.
- [ ] **Step 6: Run `pytest tests/test_batch_orchestrator.py -v`** and verify retry behavior, persistence acknowledgements, and terminal statuses pass.
- [ ] **Step 7: Commit** with `git add calls_analyser/services/batch_orchestrator.py tests/test_batch_orchestrator.py && git commit -m "feat: retry invalid batch decisions once"`.

## Chunk 2: Configuration and executors

### Task 5: Tenant verification settings and prompt defaults

**Files:**
- Modify: `calls_analyser/services/tenant_settings.py`
- Modify: `calls_analyser/config.py`
- Modify: `docs/supabase/multi_tenant_schema.sql`
- Test: `tests/test_tenant_settings.py`
- Test: `tests/test_multi_tenant_schema.py`

- [ ] **Step 1: Write failing settings tests** for safe defaults and tenant overrides of mode/model/prompt.
- [ ] **Step 2: Run `pytest tests/test_tenant_settings.py -v`** and verify missing attributes/assertion failures.
- [ ] **Step 3: Implement `follow_up_verification_mode`, `follow_up_verification_model_key`, and `follow_up_verification_prompt_key` resolution**, coercing invalid modes to `off`.
- [ ] **Step 4: Add failing prompt and schema assertions** for a discoverable strict verification prompt and the documented tenant fields; run `pytest tests/test_prompt_service.py tests/test_multi_tenant_schema.py -v` and verify failure.
- [ ] **Step 5: Add the verification prompt/template and schema documentation**, then run `pytest tests/test_tenant_settings.py tests/test_prompt_service.py tests/test_multi_tenant_schema.py -v` and expect PASS.
- [ ] **Step 6: Commit** with `git add calls_analyser/services/tenant_settings.py calls_analyser/config.py docs/supabase/multi_tenant_schema.sql tests/test_tenant_settings.py tests/test_prompt_service.py tests/test_multi_tenant_schema.py && git commit -m "feat: configure tenant follow-up verification"`.

### Task 6: Round-spec resolution and validation

**Files:**
- Modify: `calls_analyser/services/batch_orchestrator.py`
- Test: `tests/test_batch_orchestrator.py`

- [ ] **Step 1: Write failing tests** for tenant prompt/version/model/language resolution, primary custom override isolation, missing model, empty verification prompt, same prompt key, and defensive identical cache identity.
- [ ] **Step 2: Implement round-spec construction** using tenant settings, `PromptService`, and AI registry provider identity.
- [ ] **Step 3: Ensure invalid verification configuration does not abort primary execution** and marks each primary positive `config_error` fallback.
- [ ] **Step 4: Run focused orchestrator tests**.
- [ ] **Step 5: Commit** the round-spec validation changes.

### Task 7: Sequential executor

**Files:**
- Modify: `calls_analyser/services/analysis.py`
- Create: `calls_analyser/services/batch_executors.py`
- Test: `tests/test_batch_executors.py`
- Test: `tests/test_analysis_service.py`

- [ ] **Step 1: Write failing `AnalysisService` tests** for explicit cache bypass and saved audit metadata (`batch_stage`, `decision_valid`, `batch_execution`).
- [ ] **Step 2: Extend `AnalysisOptions` minimally** with bypass/audit fields and preserve direct-analysis defaults. When bypassing, skip only cache read, still upsert the new result; executor validation acknowledgement rewrites metadata on that cached `AnalysisResult`.
- [ ] **Step 3: Write failing sequential-executor tests** for cache hit, model call, per-item error, progress, `ui_mass`/`ui_mass_verify` usage modes, and tenant isolation using the same call/model/prompt for two tenants with distinct cache keys/model calls.
- [ ] **Step 4: Implement `SequentialBatchExecutor`** as an adapter over `AnalysisService`, returning one `RoundExecutionResult` per request without parsing decisions.
- [ ] **Step 5: Run `pytest tests/test_analysis_service.py tests/test_batch_executors.py -v`**.
- [ ] **Step 6: Run `pytest tests/test_analysis_service.py tests/test_batch_executors.py -v` again** and confirm tenant-isolation coverage passes before commit.
- [ ] **Step 7: Commit** the sequential executor and tests.

### Task 8: Vertex executor

**Files:**
- Modify: `calls_analyser/services/batch_executors.py`
- Test: `tests/test_batch_executors.py`
- Test: `tests/test_gemini_batch.py`

- [ ] **Step 1: Write failing tests** for bulk cache lookup, cached/non-cached partition, audio preparation, chunk size, partial output, per-item terminal errors, bypass, separate primary/verification usage modes, and tenant isolation for the same call/model/prompt across two tenant IDs.
- [ ] **Step 1a: Add a failing language test** asserting Vertex prompt text contains the `RoundSpec.language` system instruction equivalent to sequential `Language` propagation.
- [ ] **Step 2: Implement `VertexBatchExecutor`** using the existing cache identity and `VertexBatchRunner.run_batch_results`; create one runner per non-empty uncached round and honor tenant batch size.
- [ ] **Step 3: Persist result metadata and record `scheduler_batch` / `scheduler_batch_verify` usage** while preserving provider usage metadata.
- [ ] **Step 4: Ensure executor returns no extras and leaves omissions for orchestrator synthesis**; run executor and Gemini tests.
- [ ] **Step 5: Run `pytest tests/test_batch_executors.py tests/test_gemini_batch.py -v` again** and confirm language and tenant-isolation coverage passes before commit.
- [ ] **Step 6: Commit** the Vertex executor and tests.

## Chunk 3: Shared output and adapters

### Task 9: Canonical result-row projection

**Files:**
- Modify: `calls_analyser/services/batch_results.py`
- Modify: `calls_analyser/ui/utils.py`
- Modify: `calls_analyser/services/email_report.py`
- Modify: `calls_analyser/ui/layout.py`
- Test: `tests/test_batch_results.py`
- Test: `tests/test_ui_utils.py`
- Test: `tests/test_email_report_service.py`
- Test: `tests/test_usage_report_ui.py`

- [ ] **Step 1: Write failing row-projection tests** for primary pending, verification pending, complete, fallback, primary error, and invalid primary.
- [ ] **Step 2: Run `pytest tests/test_batch_results.py -v`** and verify the projection import/status assertions fail.
- [ ] **Step 3: Implement `build_batch_item_row`** with final columns plus initial/verification audit columns and status mappings.
- [ ] **Step 4: Add failing export/email-order and usage-mode-choice tests** before editing column/mode lists; verify failure with `pytest tests/test_ui_utils.py tests/test_email_report_service.py tests/test_usage_report_ui.py -v`.
- [ ] **Step 5: Add audit columns to canonical export/email order and `ui_mass_verify`/`scheduler_batch_verify` to usage-report choices**, while allowing display helpers to hide technical identity only.
- [ ] **Step 6: Run `pytest tests/test_batch_results.py tests/test_ui_utils.py tests/test_email_report_service.py tests/test_usage_report_ui.py -v`** and expect PASS.
- [ ] **Step 7: Commit** the canonical projection and reporting updates.

### Task 10: UI orchestration integration

**Files:**
- Modify: `calls_analyser/ui/dependencies.py`
- Modify: `calls_analyser/ui/handlers.py`
- Test: `tests/test_app_batch.py`
- Test: `tests/test_dependency_wiring_auth_settings.py`

- [ ] **Step 1: Write failing UI tests** proving tenant-resolved primary model/language, phase-aware pending rows, verification calls only for positives, full audit state, final filtering, and aggregate final message.
- [ ] **Step 2: Wire orchestrator and sequential executor dependencies** without affecting direct single-call analysis.
- [ ] **Step 3: Replace duplicated UI parsing/decision logic** with one orchestrator call and progress-to-DataFrame projection; retain existing authentication, filters, custom primary prompt, Gradio yields, and display hiding.
- [ ] **Step 4: Run `pytest tests/test_app_batch.py tests/test_dependency_wiring_auth_settings.py -v`**.
- [ ] **Step 5: Commit** the UI adapter integration.

### Task 11: Scheduler orchestration integration

**Files:**
- Modify: `calls_analyser/runner.py`
- Test: `tests/test_runner_email.py`

- [ ] **Step 1: Rewrite/add failing runner tests** proving tenant-resolved settings, two Vertex rounds, verification cache reuse, partial-output retry isolation, usage modes, final email decisions/audit columns, counter logs, and no-email behavior when no valid primary decisions exist.
- [ ] **Step 2: Replace runner-owned cache/model/parsing logic** with `BatchAnalysisOrchestrator` plus `VertexBatchExecutor`; keep call discovery, filters, email transport, and tenant iteration outside.
- [ ] **Step 3: Run `pytest tests/test_runner_email.py -v`** and ensure existing off-mode single-round expectations remain valid.
- [ ] **Step 4: Commit** the scheduler adapter integration.

## Chunk 4: Integration and verification

### Task 12: Cross-executor parity and regression coverage

**Files:**
- Create: `tests/test_two_pass_integration.py`
- Modify: `tests/test_app_batch.py`
- Modify: `tests/test_runner_email.py`

- [ ] **Step 1: Add failing parity tests** feeding equivalent fake responses through sequential and Vertex executors and comparing ordered `BatchItemResult` decisions/statuses/counters.
- [ ] **Step 2: Add cache-identity tests** showing verification model/prompt version changes miss only verification cache.
- [ ] **Step 3: Add off/shadow/enforce saved-result regression cases**, including legacy cached off-mode behavior and strict fresh-output validation.
- [ ] **Step 4: Run the tests and verify they fail for any missing parity/cache/regression behavior**.
- [ ] **Step 5: Implement the minimal integration corrections in the owning service/adapter files**.
- [ ] **Step 6: Run `pytest tests/test_two_pass_integration.py tests/test_app_batch.py tests/test_runner_email.py -v`** and expect PASS.
- [ ] **Step 7: Commit** the cross-executor integration coverage and corrections.

### Task 13: Labeled-sample evaluation tooling

**Files:**
- Create: `calls_analyser/services/follow_up_evaluation.py`
- Create: `scripts/evaluate_follow_up_verification.py`
- Create: `tests/test_follow_up_evaluation.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing metric tests** for primary/final confusion matrices, disagreement rate, changed-to-false percentage, failure rate, precision/recall deltas, and threshold eligibility (strictly positive precision delta, recall loss at most 2 percentage points, and verification failures below 2%).
- [ ] **Step 2: Run `pytest tests/test_follow_up_evaluation.py -v`** and verify the missing module/API failure.
- [ ] **Step 3: Implement pure evaluation functions and a CSV CLI** accepting manual labels plus primary/final/verification columns and optional `primary_tokens`, `verification_tokens`, `primary_cost`, `verification_cost`, `primary_elapsed_seconds`, and `verification_elapsed_seconds`; emit JSON totals/deltas for incremental tokens, cost, and elapsed time alongside quality metrics.
- [ ] **Step 4: Add a failing CLI smoke test**, implement argument/error handling, and document the exact command. The tool must state that `enforce` remains a tenant configuration decision and must not auto-enable it.
- [ ] **Step 5: Run `pytest tests/test_follow_up_evaluation.py -v`** and a sample temporary CSV invocation; expect a report with `eligible_for_enforcement`.
- [ ] **Step 6: Commit** the evaluation tooling. A real tenant's labeled sample is external operational input and is explicitly required before enabling that tenant; implementation does not fabricate rollout evidence.

### Task 14: Full verification and cleanup

**Files:**
- Modify only files already listed if verification exposes defects.

- [ ] **Step 1: Run `python -m pytest -q`** and resolve all regressions through new failing tests when behavior changes are needed.
- [ ] **Step 2: Run `python -m compileall calls_analyser`** to catch import/syntax errors outside collected tests.
- [ ] **Step 3: Inspect `git diff --check` and `git status --short`** for whitespace issues and unintended files.
- [ ] **Step 4: Confirm acceptance criteria against the design**: shared orchestration, candidate gating, tenant/cache isolation, enforce authority, safe fallback, auditability, off compatibility, both adapters, output columns, counters, and usage modes.
- [ ] **Step 5: Commit any verification-only fixes**, keeping unrelated files out of the branch.
