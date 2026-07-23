# Scheduler / Vertex Batch Correctness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Надзейна прывязваць Vertex Batch-вынікі да іх званкоў, захоўваць частковы поспех чанкаў, кэшаваць толькі строга валідныя вынікі і запускаць адзін абаронены scheduler-job на tenant.

**Architecture:** `VertexBatchRunner` карэлюе радкі праз дакладную мапу `fileUri -> task.key` і вяртае item-level памылкі замест страты ўсяго batch. `run_batch_process` валідуюць вынікі праз адзін строгі parser, ізалюе persistence па item і вяртае кампактны `BatchRunResult`. Scheduler стварае асобныя startup-time jobs для tenant і перад кожным запускам атамарна claim-іць слот у новым Supabase-рэпазіторыі.

**Tech Stack:** Python 3.10+, pytest, Google GenAI Vertex Batch API, Google Cloud Storage, APScheduler, Supabase/PostgREST, PostgreSQL.

**Source design:** `docs/superpowers/specs/2026-07-23-scheduler-vertex-batch-correctness-design.md` at commit `70f68b6`.

---

## File map

**Create**

- `calls_analyser/adapters/storage/supabase_scheduler_runs.py` — адзіны Supabase adapter для atomic claim і фіналізацыі scheduler run.
- `tests/test_batch_results.py` — строгі parser для plain/fenced JSON і яго памылкі.
- `tests/test_supabase_scheduler_runs.py` — insert/update payloads, duplicate claim і не-дублікатныя памылкі.

**Modify**

- `calls_analyser/services/gemini_batch.py` — валідацыя task keys, URI correlation, item-level diagnostics, partial chunks і per-object cleanup.
- `calls_analyser/services/batch_results.py` — строгая мадэль распазнанага выніку і `BatchRunResult`.
- `calls_analyser/runner.py` — strict-before-cache, ізаляваныя cache/usage writes і вяртанне `BatchRunResult`.
- `calls_analyser/services/scheduler.py` — tenant job execution, timezone/date/bucket calculation, run identity і fail-closed guard.
- `calls_analyser/ui/dependencies.py` — wiring `scheduler_run_repository`.
- `app.py` — адзін job на tenant з tenant-specific trigger і агульнай timezone.
- `docs/supabase/multi_tenant_schema.sql` — `scheduler_runs`, composite uniqueness і service-role RLS.
- `README.md` — `SCHEDULER_TIMEZONE`, startup-only schedule refresh і fail-closed requirement.
- `tests/test_gemini_batch.py` — task validation, partial chunks і cleanup.
- `tests/test_gemini_usage_metadata.py` — URI-based output correlation, multiple files і usage attachment.
- `tests/test_runner_email.py` — strict validation, persistence isolation і `BatchRunResult`.
- `tests/test_scheduler_service.py` — timezone, slot identity, claim/skip/finalize і fail-closed behavior.
- `tests/test_app_scheduler.py` — асобныя cron/interval jobs на tenant і `max_instances=1`.
- `tests/test_dependency_wiring_auth_settings.py` — Supabase/local wiring новага рэпазіторыя.
- `tests/test_multi_tenant_schema.py` — schema/RLS/unique-key assertions і гарантыя, што `analysis_results` не мігруе.

**Intentionally unchanged**

- `calls_analyser/adapters/storage/supabase_storage.py` і структура `analysis_results` — існуючыя cache keys і гістарычныя радкі не змяняюцца.
- `calls_analyser/services/tenant_settings.py` — цяперашніх tenant scheduler settings дастаткова.
- Няма heartbeat, stale-run takeover, backfill або dynamic rescheduling.

## Chunk 1: Correct batch results and persistence

### Task 1: Add strict shared result parsing and the batch outcome contract

**Files:**

- Create: `tests/test_batch_results.py`
- Modify: `calls_analyser/services/batch_results.py`

- [ ] **Step 1: Specify the strict parser with focused tests**

  Add parametrized tests for:

  - plain JSON;
  - fenced ` ```json ... ``` ` and unlabelled fences;
  - `needs_follow_up` equal to `true` and `false`;
  - rejection of missing keys;
  - rejection of string/integer/null `needs_follow_up`;
  - rejection of non-string/null `reason`;
  - rejection of trailing prose or the old `Needs follow-up:` format.

  The public contract should be explicit:

  ```python
  @dataclass(frozen=True)
  class FollowUpResult:
      needs_follow_up: bool
      reason: str


  def parse_follow_up_result(text: str) -> FollowUpResult:
      """Parse one complete plain/fenced JSON object or raise ValueError."""
  ```

- [ ] **Step 2: Implement one parser used for both validation and report fields**

  Strip only a complete outer Markdown fence, decode the complete remaining string with `json.loads`, require a JSON object, then check exact Python types (`type(value) is bool`, `isinstance(reason, str)`). Do not retain the permissive prose fallback.

  Update `build_success_row` to accept a pre-parsed `FollowUpResult` so `runner.py` does not parse the same model response twice:

  ```python
  def build_success_row(entry, tenant, result: FollowUpResult) -> dict[str, object]:
      return {
          # existing call columns...
          "Needs follow-up": "Yes" if result.needs_follow_up else "No",
          "Reason": result.reason,
          "Status": "✅",
      }
  ```

- [ ] **Step 3: Define the scheduler-facing batch summary in the same focused module**

  Add:

  ```python
  BatchRunStatus = Literal["success", "partial", "failed"]


  @dataclass(frozen=True)
  class BatchRunResult:
      status: BatchRunStatus
      total_count: int
      success_count: int
      failure_count: int
      cached_count: int = 0

      @classmethod
      def from_counts(
          cls,
          *,
          total_count: int,
          success_count: int,
          failure_count: int,
          cached_count: int = 0,
      ) -> "BatchRunResult":
          status = (
              "success"
              if failure_count == 0
              else "partial"
              if success_count > 0
              else "failed"
          )
          return cls(status, total_count, success_count, failure_count, cached_count)
  ```

  Count every cache hit in `cached_count`. A cached row that passes the same strict parser is also a successful item; an invalid historical cached row is a failed item but remains untouched and is not reprocessed. A run with zero calls is `success` with all counters at zero.

### Task 2: Correlate Vertex output by file URI

**Files:**

- Modify: `calls_analyser/services/gemini_batch.py`
- Modify: `tests/test_gemini_batch.py`
- Modify: `tests/test_gemini_usage_metadata.py`

- [ ] **Step 1: Add tests that reproduce positional mis-correlation**

  Build fake output blobs whose rows:

  - arrive in reverse order;
  - are split across two `.jsonl` files;
  - preserve each original audio URI under `request.contents[*].parts[*].fileData.fileUri`;
  - contain visibly different text and usage counts.

  Assert both text and `usage_metadata` land under the key mapped to that URI, regardless of file or row order.

- [ ] **Step 2: Add task-key preflight tests**

  Assert `run_batch_results` rejects the whole input before `_run_single_batch` when any `BatchTask.key` is blank/whitespace or duplicated. Use a stable `ValueError` message containing the invalid/duplicate key.

- [ ] **Step 3: Preserve both directions of the upload mapping**

  Add a small immutable upload result:

  ```python
  @dataclass(frozen=True)
  class UploadedBatchInputs:
      uri_by_key: dict[str, str]
      key_by_uri: dict[str, str]
  ```

  `_upload_audio_to_gcs` fills both maps from the actual URI it generated. `_write_jsonl_to_gcs` uses `uri_by_key`; `_run_single_batch` passes `key_by_uri` into `_read_output_jsonl`. Never recover a key by parsing a blob name.

- [ ] **Step 4: Parse and classify every output row independently**

  Introduce a helper that extracts `fileUri` only from the row's preserved request:

  ```python
  def _request_file_uri(row: dict[str, Any]) -> str | None:
      request = row.get("request")
      # Walk contents/parts and return the single fileData.fileUri.
  ```

  In `_read_output_jsonl`:

  - log malformed JSONL with blob name and line number;
  - log and ignore rows with missing or unknown URIs;
  - detect a repeated known URI across all output files and return an item error for that key;
  - map `status` or `error` from a known URI to `BatchAnalysisResult(text="Error: ...")`;
  - map a valid response and its usage to the key from `key_by_uri`;
  - after all files, return `Error: missing response` for every expected key with no exactly-one usable row.

  A bad row must never consume or shift another task's result.

### Task 3: Preserve successful chunks and make cleanup truly best-effort

**Files:**

- Modify: `calls_analyser/services/gemini_batch.py`
- Modify: `tests/test_gemini_batch.py`

- [ ] **Step 1: Cover a failed middle chunk and a successful later chunk**

  Use three one-item chunks. Make chunk 2 fail for every configured attempt. Assert the returned mapping contains:

  - chunk 1 success;
  - chunk 2 `Error: batch chunk failed: ...`;
  - chunk 3 success;
  - the expected retry count only for chunk 2.

- [ ] **Step 2: Convert exhausted chunk exceptions to item errors**

  Keep the retry loop, but after the last failure populate a `BatchAnalysisResult` error for every task in that chunk and continue the outer loop. Do not raise after earlier chunks have completed.

- [ ] **Step 3: Isolate GCS deletion per object**

  Move `try/except` inside the blob loop. Log the failing blob name, continue deleting remaining objects, and also retain an outer guard for `list_blobs` failure.

  Extend the cleanup test so the first fake blob raises from `delete()` and the second is still deleted.

### Task 4: Validate before cache and isolate persistence per call

**Files:**

- Modify: `calls_analyser/runner.py`
- Modify: `tests/test_runner_email.py`

- [ ] **Step 1: Add runner tests for invalid model output**

  Feed a mix of:

  - valid plain JSON;
  - valid fenced JSON;
  - invalid JSON;
  - wrong field types.

  Assert only valid items are written to cache, invalid items become error report rows, and the returned summary is `partial` with exact counters.

  Add a separate mixed-cache case with one valid and one invalid historical cached text. Assert both increment `cached_count`, only the valid row increments `success_count`, the invalid row increments `failure_count`, neither cache row is deleted/overwritten, and no Vertex task is created for either cache hit.

- [ ] **Step 2: Add cache and usage failure isolation tests**

  Make one cache write raise and verify later calls still persist. Separately make one usage write raise and verify later usage writes still run. Cache failure makes that item failed; usage failure is logged as an auxiliary persistence error but does not invalidate an already cached analysis result.

- [ ] **Step 3: Restructure the per-item result loop**

  First run every cache hit through `parse_follow_up_result` for report construction only. Put valid parsed values in `parsed_result_by_id`; turn invalid historical text into an item error without deleting, invalidating or reprocessing it.

  Then, for each uncached task:

  1. reject missing/`Error:` batch results;
  2. call `parse_follow_up_result`;
  3. build `AnalysisResult`;
  4. write cache in its own `try/except`;
  5. write usage in a separate `try/except`;
  6. only then add the parsed result to report state.

  Do not let a parser, cache, or usage exception escape the item loop. Build every success report row from `parsed_result_by_id`, so cached and newly analyzed rows compose with the new `build_success_row(..., FollowUpResult)` contract.

- [ ] **Step 4: Resolve one immutable execution context for identity and work**

  Add a frozen `BatchExecutionContext` plus a public resolver in `runner.py`. It must contain every value that could otherwise diverge between guard claim and cache write: resolved tenant, model key, provider name, batch size/language, exact merged prompt body, prompt version, and resolved call filters.

  ```python
  @dataclass(frozen=True)
  class BatchExecutionContext:
      tenant: Any
      prompt_key: str
      batch_model_key: str
      provider_name: str
      batch_size: int
      batch_language: Any
      merged_prompt: str
      prompt_version: int
      time_from: Any
      time_to: Any
      call_type: Any


  def resolve_batch_execution_context(...) -> BatchExecutionContext:
      """Resolve scheduler/cache identity and all matching execution inputs once."""
  ```

  Extend `run_batch_process(..., execution_context: BatchExecutionContext | None = None)`: CLI/manual callers resolve internally when omitted; the scheduler passes a pre-resolved context and the function must not re-read tenant settings, tenant ID, prompt key, model or prompt in that path. Every cache and usage key must use `execution_context.tenant.tenant_id`, `execution_context.prompt_key`, `execution_context.prompt_version`, `execution_context.provider_name`, and `execution_context.batch_model_key`.

- [ ] **Step 5: Return `BatchRunResult` instead of the DataFrame**

  Keep DataFrame construction and email sending internal. Return exact counters on every exit path, including disabled configuration, invalid filters, fetch failure, no calls, all-cached calls and model lookup failure.

  Use failures for actual requested items. Configuration/fetch failures before item discovery return `failed` with zero item counters and a logged cause. The CLI remains unchanged because it ignores the return value; scheduler consumes the new result.

## Chunk 2: Guarded per-tenant scheduler

### Task 5: Add the `scheduler_runs` schema and Supabase adapter

**Files:**

- Modify: `docs/supabase/multi_tenant_schema.sql`
- Modify: `tests/test_multi_tenant_schema.py`
- Create: `calls_analyser/adapters/storage/supabase_scheduler_runs.py`
- Create: `tests/test_supabase_scheduler_runs.py`

- [ ] **Step 1: Specify the schema without touching historical results**

  Add `public.scheduler_runs` with:

  ```sql
  tenant_id text not null references public.tenants(id) on delete cascade,
  scheduled_for timestamptz not null,
  prompt_key text not null,
  prompt_version integer not null,
  model_key text not null,
  status text not null check (status in ('running', 'success', 'partial', 'failed')),
  total_count integer not null default 0,
  success_count integer not null default 0,
  failure_count integer not null default 0,
  cached_count integer not null default 0,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  primary key (tenant_id, scheduled_for, prompt_key, prompt_version, model_key)
  ```

  Enable RLS and add one service-role `for all` policy. Extend the schema test to assert these exact columns/key/policy and that no data-changing statement targets `analysis_results`.

- [ ] **Step 2: Define repository data contracts and fake-table tests**

  In the adapter add:

  ```python
  @dataclass(frozen=True)
  class SchedulerRunKey:
      tenant_id: str
      scheduled_for: datetime
      prompt_key: str
      prompt_version: int
      model_key: str


  class SupabaseSchedulerRunRepository:
      def claim(self, key: SchedulerRunKey) -> bool: ...
      def finish(self, key: SchedulerRunKey, result: BatchRunResult) -> None: ...
  ```

  Tests must verify UTC ISO timestamps, `status="running"` insert, composite-key filters on update, result counters, and `finished_at`.

  Add a genuinely concurrent atomic-claim test, not two sequential calls: synchronize two workers with `threading.Barrier` and use either a PostgreSQL/Supabase integration fixture or a repository-capable fake whose unique insert is protected by one lock. Assert the two returned claim values are exactly `[False, True]` in any order and only one `running` row exists.

- [ ] **Step 3: Implement atomic claim and strict duplicate handling**

  `claim` performs one plain insert. Return `False` only for a Postgres uniqueness violation (`23505`/the concrete PostgREST duplicate response); re-raise authentication, network, missing-table and all other errors so the scheduler fails closed.

  `finish` updates only the row identified by all five key columns. No heartbeat, lease, stale-run takeover or retry mutation is added.

### Task 6: Execute one guarded scheduler slot for one tenant

**Files:**

- Modify: `calls_analyser/services/scheduler.py`
- Modify: `tests/test_scheduler_service.py`

- [ ] **Step 1: Replace multi-tenant dispatch tests with one-tenant job tests**

  Cover:

  - global timezone parsing with `ZoneInfo`;
  - target day computed as local `now.date() - 1 day`;
  - delayed cron execution still derives the latest planned tenant cron instant, not the wall-clock invocation minute, then stores it as UTC;
  - interval `scheduled_for` floored to the current interval bucket then stored as UTC;
  - two simultaneous attempts with the same five-part identity call the runner once;
  - duplicate claim skips without calling runner or `finish`;
  - unavailable repository and non-duplicate claim errors fail closed;
  - runner `success`, `partial`, `failed`, and raised exception finalize the claimed row correctly.

- [ ] **Step 2: Add explicit time/identity helpers**

  Keep pure helpers independently testable:

  ```python
  def scheduler_timezone(value: str | None) -> ZoneInfo: ...

  def cron_scheduled_for(now: datetime, cron_time: time) -> datetime: ...

  def interval_scheduled_for(now: datetime, minutes: int) -> datetime: ...
  ```

  `cron_scheduled_for` chooses the latest planned occurrence of the configured tenant cron time at or before `now`; this keeps delayed/misfired jobs on their intended slot. Immediately before claim, build one `BatchExecutionContext` using the startup-resolved tenant scheduler settings and one exact prompt lookup (fallback version `1`). Build every identity field in `SchedulerRunKey` from that context (`tenant.tenant_id`, `prompt_key`, `prompt_version`, `batch_model_key`), then pass the same context object into `run_batch_process`. The runner must never resolve a second tenant/prompt/model identity for that guarded execution.

- [ ] **Step 3: Implement the guarded tenant executor**

  Add a single entry point similar to:

  ```python
  def run_scheduled_batch_for_tenant(
      *,
      tenant_id: str,
      runtime_settings,
      scheduled_for: datetime,
      now: datetime,
      run_repository,
      runner,
      deps,
  ) -> BatchRunResult | None:
      """Claim once, run yesterday for this tenant, finish once; None means duplicate."""
  ```

  Claim before listing/downloading calls. On a runner exception, finalize the claimed row as `failed` with zero counters and re-raise/log. If claim cannot be protected, do not invoke the runner.

  Add a mutation test where `claim()` changes the underlying tenant setting, `deps.batch_prompt_key`, and active prompt before returning. Assert the runner receives the already frozen context and that the tenant ID, prompt key/version, and model key used by cache/usage writes remain identical to the claimed `SchedulerRunKey`.

### Task 7: Wire the run repository and register one startup job per tenant

**Files:**

- Modify: `calls_analyser/ui/dependencies.py`
- Modify: `tests/test_dependency_wiring_auth_settings.py`
- Modify: `app.py`
- Modify: `tests/test_app_scheduler.py`
- Modify: `README.md`

- [ ] **Step 1: Wire the repository only for complete Supabase credentials**

  Add `scheduler_run_repository: Any = None` to `AppDependencies`. Import and construct `SupabaseSchedulerRunRepository(supabase_url, supabase_key)` beside the cache/usage repositories. Keep it `None` for local storage, incomplete credentials, missing optional imports or constructor failure.

  Extend dependency tests to assert the Supabase instance is wired and local/minimal fallbacks remain `None`.

- [ ] **Step 2: Redesign scheduler registration around tenant snapshots**

  `_register_scheduler_jobs_if_available` should:

  - read `SCHEDULER_TIMEZONE` once, defaulting to `UTC`;
  - list enabled tenants once at startup;
  - resolve each tenant's settings once for trigger construction;
  - add one closure/job per tenant;
  - use a stable ID such as `scheduler:{tenant_id}`;
  - set `replace_existing=True` and `max_instances=1`;
  - register `cron` with that tenant's `scheduler_cron_time`, or `interval` with its `scheduler_interval_minutes`;
  - start APScheduler once after all jobs are added.

  The closure computes `now`, the tenant's planned `scheduled_for` and yesterday's target date inside execution, then delegates to `run_scheduled_batch_for_tenant`. For cron it passes the startup-resolved hour/minute to `cron_scheduled_for`; for interval it passes the startup-resolved interval size. Add a delayed cron test (for example invocation at `02:37` for a `02:30` job) asserting the guard key remains `02:30`, not `02:37`.

- [ ] **Step 3: Cover tenant-specific registration**

  Update `RecordingScheduler` tests to assert two tenants can receive different trigger modes/settings while sharing the configured timezone. Assert no enabled tenants means no scheduler start, and no run repository never falls through to an unguarded batch call.

- [ ] **Step 4: Document the operational contract**

  In `README.md`, document:

  - `SCHEDULER_TIMEZONE` as one IANA timezone (default `UTC`);
  - schedule changes require application restart;
  - scheduled execution requires Supabase plus the applied `scheduler_runs` schema and fails closed otherwise;
  - interrupted `running` slots require manual intervention;
  - there is intentionally no heartbeat, stale takeover, backfill or dynamic rescheduling.

## Chunk 3: One overall verification and review

### Task 8: Verify the complete implementation once

**Files:**

- Review all files listed in the file map.
- Do not modify historical `analysis_results` rows or add data-repair scripts.

- [ ] **Step 1: Run all focused correctness tests together**

  Run:

  ```powershell
  pytest tests/test_batch_results.py tests/test_gemini_batch.py tests/test_gemini_usage_metadata.py tests/test_runner_email.py tests/test_supabase_scheduler_runs.py tests/test_scheduler_service.py tests/test_app_scheduler.py tests/test_dependency_wiring_auth_settings.py tests/test_multi_tenant_schema.py -v
  ```

  Expected: all focused tests pass; no unexpected xfail/skip.

- [ ] **Step 2: Run scheduler, runner, cache and Vertex regression tests together**

  Run:

  ```powershell
  pytest tests/test_supabase_storage.py tests/test_supabase_usage.py tests/test_tenant_settings.py tests/test_app_batch.py tests/test_runner_email.py tests/test_scheduler_service.py tests/test_app_scheduler.py tests/test_gemini_batch.py tests/test_gemini_usage_metadata.py -v
  ```

  Expected: all regression tests pass.

- [ ] **Step 3: Run the complete suite**

  Run:

  ```powershell
  pytest
  ```

  Expected: the complete suite passes.

- [ ] **Step 4: Perform one final scope/code review**

  Compare the diff with `docs/superpowers/specs/2026-07-23-scheduler-vertex-batch-correctness-design.md` and verify:

  - every output is correlated only by preserved `fileUri`;
  - failed chunks do not erase or stop other chunks;
  - invalid JSON never reaches cache;
  - one item's parser/cache/usage failure does not stop later items;
  - one job exists per enabled tenant with `max_instances=1`;
  - every scheduled run claims before work and fails closed without the guard;
  - no historical cache rows, backfill, heartbeat, stale takeover or dynamic rescheduling were introduced.

- [ ] **Step 5: Run the small manual Vertex smoke test**

  Use a tenant with two or three calls whose durations/content are obviously different. Run one small scheduler batch, inspect the generated report, `analysis_results`, `analysis_usage`, and `scheduler_runs`, and confirm each text/token count stays with its call and the run finishes with the expected counters/status. Trigger the same slot identity a second time and confirm it is skipped.

- [ ] **Step 6: Commit the verified implementation**

  ```powershell
  git add `
    calls_analyser/adapters/storage/supabase_scheduler_runs.py `
    calls_analyser/services/gemini_batch.py `
    calls_analyser/services/batch_results.py `
    calls_analyser/runner.py `
    calls_analyser/services/scheduler.py `
    calls_analyser/ui/dependencies.py `
    app.py `
    docs/supabase/multi_tenant_schema.sql `
    docs/superpowers/plans/2026-07-23-scheduler-vertex-batch-correctness.md `
    README.md `
    tests/test_batch_results.py `
    tests/test_supabase_scheduler_runs.py `
    tests/test_gemini_batch.py `
    tests/test_gemini_usage_metadata.py `
    tests/test_runner_email.py `
    tests/test_scheduler_service.py `
    tests/test_app_scheduler.py `
    tests/test_dependency_wiring_auth_settings.py `
    tests/test_multi_tenant_schema.py
  git commit -m "fix: make scheduled Vertex batches deterministic"
  ```
