# Two-Pass Follow-Up Verification

## Goal

Improve the precision of batch `needs_follow_up` decisions by independently re-analyzing calls that the first round marks as requiring follow-up. The second round uses a tenant-specific model and prompt. In enforcement mode, a successful second-round decision replaces the first-round decision.

The same decision pipeline must serve both interactive UI batch analysis and scheduled Vertex Batch analysis so that execution mode does not change business behavior.

## Scope and Non-Goals

This change covers batch analysis launched from the Gradio UI and the scheduled batch path in `calls_analyser/runner.py`. It includes tenant configuration, two-round orchestration, cache behavior, reporting, usage tracking, retries, and tests.

It does not change direct single-call analysis. It does not introduce a persistent job queue or a new batch-job database. It does not re-check first-round `false` decisions, so it can reduce false positives but cannot recover false negatives from the first round.

## Product Decisions

- Only valid first-round `needs_follow_up=true` results enter verification.
- Verification independently analyzes the original audio. The first-round decision and reason are not included in the verification prompt, avoiding confirmation bias.
- The verification model and prompt key are configured per tenant.
- A successful second-round decision is authoritative in enforcement mode.
- If verification cannot produce a valid decision, the safe fallback remains the first-round `true`; a failed verification must never become `false`.
- UI and scheduler use the same orchestrator and decision rules, with different execution adapters.

## Architecture

### `BatchAnalysisOrchestrator`

The orchestrator owns the two-round workflow. It accepts call entries, the resolved tenant and settings, a round executor, and an optional progress callback. It returns a `BatchRunResult` containing one `BatchItemResult` per requested call and aggregate counters.

The orchestrator is independent of Gradio, pandas, email, GCS, and Vertex job lifecycle details. Its responsibilities are:

1. resolve round specifications from tenant settings and prompts;
2. execute round one for all entries;
3. strictly parse and validate decisions;
4. select valid first-round positives;
5. execute verification when the tenant mode requires it;
6. apply the final-decision matrix;
7. expose progress, audit fields, and aggregate counters.

Call `unique_id` is the round-result key and must be unique within one orchestrator request. The orchestrator validates this before any cache or model work and rejects a duplicate-ID request with a clear `ValueError`. This prevents dictionary result collapse and preserves the one-result-per-input ordering contract.

### `BatchRoundExecutor`

Both execution paths implement a shared interface conceptually equivalent to:

```python
execute(
    entries,
    tenant,
    round_spec,
    *,
    bypass_cache=False,
    progress=None,
) -> dict[str, RoundExecutionResult]
```

`RoundSpec` contains the model key, prompt key, resolved prompt text and version, custom prompt fragment, language, usage mode, and stage name. `RoundExecutionResult` contains the raw text, provider/model identity, `execution_status`, `from_cache`, usage metadata, and an execution error if any.

The status contracts are closed sets:

- `execution_status`: `success`, `error`, or `missing`; cache hits are successful executions distinguished by `from_cache=true`;
- `decision_status`: `valid`, `invalid`, or `unavailable`;
- `final_status`: `pending`, `complete`, `fallback`, `error`, or `invalid`;
- `verification_status`: `not_requested`, `disabled`, `pending`, `shadow_complete`, `complete`, `failed`, or `config_error`.

If an executor omits a requested ID from its result dictionary, the orchestrator synthesizes `execution_status=missing`; omission is never interpreted as an empty successful response. Executors must return no unrequested IDs. The orchestrator rejects extras as an executor contract error.

- `SequentialBatchExecutor` uses `AnalysisService` for the UI and emits per-item progress.
- `VertexBatchExecutor` wraps `VertexBatchRunner` for the scheduler and submits one Vertex batch per non-empty round, chunked with the existing tenant batch size.

Executors perform model calls, cache access, and usage recording. They do not parse follow-up decisions or choose candidates for another round.

### `FollowUpDecisionParser`

One parser replaces the duplicated parsing rules in `calls_analyser/ui/handlers.py` and `calls_analyser/services/batch_results.py`. Its output is:

```python
FollowUpDecision(
    needs_follow_up: bool,
    reason: str,
)
```

The accepted new model output is a JSON object containing an actual JSON boolean in `needs_follow_up` and a non-empty string in `reason`. Markdown JSON fences may be stripped, but values such as `"false"`, `0`, missing fields, and free-form prose are invalid.

Default `off` mode retains a compatibility parser for historical cached results and therefore preserves current single-round behavior without a format-only reanalysis. The compatibility parser may read the existing `Needs follow-up: Yes/No` format, but only for a cached primary result in `off` mode. In `shadow` or `enforce`, primary and verification decisions must pass the strict parser; a legacy cached primary result is unusable for candidate selection and receives the bounded cache-bypassing retry. All newly produced responses in every mode are validated strictly before being marked `decision_valid=true`.

### Result Models

Each `BatchItemResult` contains:

- the call entry;
- first-round execution status, raw response, and parsed decision;
- optional verification execution status, raw response, and parsed decision;
- final decision and reason, when available;
- final status and verification status;
- whether either round came from cache.

`BatchRunResult` preserves input ordering and contains counters needed by UI, logs, and scheduled reporting.

A single row-projection function maps `BatchItemResult` into the canonical DataFrame fields used by UI, CSV, and email. Outer adapters may hide columns for display, but they do not reinterpret statuses or decisions.

## Tenant Configuration

`TenantRuntimeSettings` gains:

- `follow_up_verification_mode`: `off`, `shadow`, or `enforce`; default `off`;
- `follow_up_verification_model_key`: model used for the second round;
- `follow_up_verification_prompt_key`: prompt resolved through the existing tenant-aware `PromptService`.

Verification reuses the tenant's batch language and batch size. Separate verification language and chunk-size settings are excluded until a demonstrated need exists.

For `shadow` or `enforce`, configuration validation requires the model to exist in the AI registry, the resolved prompt body to be non-empty, and the verification prompt key to differ from the effective primary prompt key. The model may be the same, but the distinct prompt key guarantees a different existing cache identity and independently auditable records. A matching effective cache identity is also rejected defensively as `config_error`. Invalid verification configuration does not stop round one; affected first-round positives receive `verification_status=config_error` and keep their first-round `true` as the final safety fallback.

## Data Flow

1. Both callers resolve `TenantRuntimeSettings` through `TenantSettingsService`; the UI no longer uses dependency-level model/language values when tenant runtime values exist. Call filtering remains in the outer adapters.
2. The orchestrator builds the primary `RoundSpec` from the tenant's normal batch model, language, and prompt. A UI custom batch prompt overrides only the primary prompt text and participates in the existing `custom_fragment` cache identity exactly as today. Scheduler runs have no custom override. Verification always uses its tenant-configured prompt independently of a primary custom prompt.
3. The selected executor retrieves cached primary results and processes misses.
4. The orchestrator parses each primary response. Missing, failed, and invalid responses are not verification candidates.
5. Valid primary positives become verification candidates when mode is `shadow` or `enforce`.
6. The orchestrator builds the verification `RoundSpec` from the tenant-specific model and prompt.
7. The same executor retrieves cached verification results and processes misses using the original recordings.
8. The orchestrator parses verification responses and derives final decisions.
9. The UI adapter renders progressive and final DataFrames. The scheduler adapter builds the email DataFrame and sends the existing filtered report.

The UI remains sequential and reports phase-aware progress. The scheduler remains Vertex Batch-based and starts a second Vertex job only when at least one candidate is not already cached. In both paths, the resolved tenant model, language, prompt version, and verification settings are the same inputs to orchestration.

## Decision Matrix

| Primary result | Mode | Verification result | Final result | Final status | Verification status |
|---|---|---|---|---|---|
| valid `false` | any | not run | primary `false` | `complete` | `not_requested` |
| valid `true` | `off` | not run | primary `true` | `complete` | `disabled` |
| valid `true` | `shadow` | valid `true` or `false` | primary `true` | `complete` | `shadow_complete` |
| valid `true` | `enforce` | valid `true` | verification `true` | `complete` | `complete` |
| valid `true` | `enforce` | valid `false` | verification `false` | `complete` | `complete` |
| valid `true` | `shadow` or `enforce` | error or invalid | primary `true` | `fallback` | `failed` |
| valid `true` | `shadow` or `enforce` | invalid configuration | primary `true` | `fallback` | `config_error` |
| error | any | not run | no decision | `error` | `not_requested` |
| invalid | any | not run | no decision | `invalid` | `not_requested` |

In shadow mode, the verification decision is retained for comparison but never changes the final decision.

## Cache and Persistence

Primary and verification results use the existing `analysis_results` cache. The current cache identity includes tenant, call ID, prompt key/version, provider, model, and custom fragment. Verification requires a prompt key distinct from the effective primary prompt key, so the stages cannot reuse or overwrite one another even when they use the same model. `batch_stage` remains audit metadata rather than cache identity.

Saved metadata gains:

- `batch_stage`: `primary` or `verification`;
- `decision_valid`: boolean;
- `batch_execution`: `ui_sequential` or `vertex_batch`;
- existing usage metadata when available.

The orchestrator re-parses cached raw text rather than trusting metadata alone. In `shadow` and `enforce`, a cached invalid or legacy-format decision is unusable and retried once with `bypass_cache=True`. In `off`, the historical primary compatibility parser is used and a legacy-format cache hit does not trigger reanalysis. The latest invalid raw response may remain stored with `decision_valid=false` for diagnosis, but it cannot satisfy a later strict decision lookup. A successful retry upserts the valid response at the same cache identity.

The composite final result is derived from the two round records and is not stored as a third analysis record. This avoids stale composite data when a model or prompt version changes.

## Output and Auditability

The existing `Needs follow-up` and `Reason` columns contain the final decision. The full in-memory DataFrame, CSV export, and email attachment also contain:

- `Initial needs follow-up`;
- `Initial reason`;
- `Verification needs follow-up`;
- `Verification reason`;
- `Verification status`.

The visible UI table may continue hiding technical identity fields, but the full state retains them for row selection and export. Filtering by "Needs follow-up" uses only the final column.

The canonical row projection applies these mappings:

| Item state | `Status` | `Needs follow-up` | `Reason` |
|---|---|---|---|
| primary pending | `⏳ Primary analysis` | blank | blank |
| primary `true`, verification pending | `⏳ Verification` | blank | blank; initial reason is present only in the audit column |
| complete primary/verification decision | `✅` | final `Yes` or `No` | final reason |
| verification failed or config error with primary `true` fallback | `⚠️` | `Yes` | primary reason; technical detail remains in verification fields |
| primary execution error | `❌` | blank | execution error |
| primary invalid after retry | `❌` | blank | invalid-response explanation |

Progressive UI output never places a pending primary positive in the final `Needs follow-up` column. In shadow mode, a completed verification cannot alter the final primary value, but its audit columns and `shadow_complete` status are populated.

The final UI message and scheduler logs expose:

- `total`;
- `round_1_success`;
- `verification_requested`;
- `verification_success`;
- `verification_changed_to_false`;
- `verification_failed`;
- `final_follow_up`.

Counter definitions are identical for UI and scheduler:

- `total`: number of unique accepted input entries;
- `round_1_success`: entries with a valid primary decision under the active mode's parsing rules, not merely provider success;
- `verification_requested`: valid primary positives in `shadow` or `enforce`, including candidates later marked `config_error`;
- `verification_success`: requested entries with a valid verification decision;
- `verification_changed_to_false`: `enforce` entries whose valid primary `true` became final `false`; shadow disagreements are excluded and reported through disagreement metrics;
- `verification_failed`: requested entries ending in `failed` or `config_error`;
- `final_follow_up`: entries whose final decision is `true`, including safe fallbacks.

## Error Handling and Retry

- After the executor exhausts its internal provider/chunk transport retry, any primary or verification item with `execution_status=error`, `missing`, or `decision_status=invalid` receives at most one orchestrator-level retry with `bypass_cache=True`.
- A first-round error or missing response after that retry produces `final_status=error`; verification is not attempted. A first-round invalid response after that retry produces `final_status=invalid` and no final boolean.
- A verification error, missing response, or invalid response is retried once for that item. If it still fails, the final decision safely falls back to the primary `true` and the verification status is `failed`.
- A partial Vertex result does not discard successful items. Only missing, errored, or invalid items enter the one orchestrator-level retry.
- A chunk-level transport or Vertex job failure retains the existing bounded executor retry. The executor returns per-item terminal statuses after that policy is exhausted; orchestration then performs no more than its single cache-bypassing retry per affected item and never loops indefinitely.
- Invalid verification configuration is reported as `config_error`; round one and reporting continue.
- If all primary analyses fail, email sending follows the existing behavior and is skipped because there are no successful decisions.

Errors are represented per item so one recording cannot abort the complete batch unless call discovery or a shared prerequisite fails before orchestration starts.

## Usage Tracking and Observability

Verification usage is recorded separately with modes:

- `ui_mass_verify`;
- `scheduler_batch_verify`.

Primary modes remain unchanged. Logs and reports make verification volume, failures, cache hits, latency, tokens, and cost distinguishable by tenant, model, prompt key, and prompt version.

Quality evaluation tracks:

- percentage of primary positives changed to final false;
- primary-versus-verification disagreement rate;
- verification invalid/failure rate;
- precision and recall against manual labels;
- incremental tokens, cost, and elapsed time.

Because only primary positives are rechecked, rollout decisions prioritize improved precision while guarding against recall loss.

## Rollout

The default mode is `off`, preserving single-round execution and existing final-decision semantics for every tenant. Fresh model responses are nevertheless subject to the new strict validation, so a malformed new response is surfaced as invalid instead of being displayed as a successful undecided row.

For each tenant:

1. configure and version the verification prompt and model;
2. enable `shadow` and collect decisions without changing user-visible results;
3. review a manually labeled sample and operational metrics;
4. enable `enforce` only when precision improves over the primary baseline, recall decreases by no more than two percentage points, and verification failures remain below two percent in the evaluation sample.

These thresholds are an operational enablement policy, not a runtime block. Returning to `shadow` or `off` is a configuration-only rollback and does not delete cached verification results.

## Integration Changes

- Extract the common strict parser and row-building inputs from UI and scheduled reporting code.
- Route UI batch analysis through `BatchAnalysisOrchestrator` with `SequentialBatchExecutor`, preserving primary custom-prompt overrides and custom-fragment cache isolation.
- Route `run_batch_process` through the same orchestrator with `VertexBatchExecutor`.
- Keep Gradio updates, pandas display formatting, email sending, and scheduler tenant iteration in their existing outer adapters.
- Extend tenant settings resolution and Supabase tenant setting documentation with the three verification fields.
- Add a default verification prompt key/template only as a discoverable base; tenants in `shadow` or `enforce` must resolve a non-empty prompt and valid model.

No unrelated telephony, authentication, direct-analysis, or email transport refactoring is included.

## Testing

### Unit tests

- strict parser accepts valid booleans and fenced JSON;
- strict parser rejects string booleans, numbers, missing/empty reasons, and prose;
- the orchestrator selects only valid primary positives;
- every row of the decision matrix produces the specified final result and statuses;
- shadow mode never changes the final decision;
- verification receives original audio context but not the primary decision text;
- invalid cached responses trigger exactly one cache-bypassing retry;
- changed verification model or prompt version causes a verification cache miss without invalidating primary cache;
- aggregate counters are correct for mixed success, disagreement, and failure batches;
- input ordering is preserved.

### Integration tests

- tenant settings resolve verification mode/model/prompt with safe defaults;
- UI and scheduler both use tenant-resolved primary model and language values;
- UI sequential and scheduler Vertex executors produce identical `BatchItemResult` decisions for equivalent fake responses;
- partial Vertex output retries only affected candidates;
- primary and verification usage records use distinct modes;
- exported and emailed DataFrames use final decisions while retaining audit columns;
- existing tenants in default `off` mode retain single-round behavior.

### Regression evaluation

Run `off`, `shadow`, and `enforce` logic against a manually labeled saved-call sample. Record the primary and final confusion matrices, disagreement cases, latency, and incremental cost. The evaluation must demonstrate the rollout thresholds before enabling enforcement for a tenant.

## Acceptance Criteria

- UI and scheduled batches share one two-round decision implementation.
- Verification runs only for valid primary positives and only in `shadow` or `enforce`.
- Tenant-specific verification model and prompt/version participate in cache identity, and verification cannot share the effective primary cache identity.
- A valid verification decision controls the final result in `enforce`.
- Failed verification safely preserves the primary `true` and is visible in status and counters.
- Primary and verification raw outputs remain independently auditable.
- Default configuration produces no second-round calls and no verification-driven final-decision change; strict validation still applies to fresh responses.
- Default `off` mode continues to display legacy cached primary results without format-only model calls, while `shadow` and `enforce` require strict decisions.
- Tests cover the decision matrix, cache/retry behavior, both execution paths, tenant isolation, outputs, and usage tracking.
