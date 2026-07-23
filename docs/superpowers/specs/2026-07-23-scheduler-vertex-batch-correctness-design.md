# Scheduler / Vertex Batch Correctness Design

## Goal

Fix the current scheduler and Vertex Batch code so every model response is
attached to the call that produced it, successful work is not discarded after
a later chunk failure, and duplicate scheduler processes do not run the same
tenant batch concurrently.

Historical analysis results are explicitly out of scope and will not be read,
changed, invalidated, or reprocessed.

## Simplified Design

### Vertex result correlation

`VertexBatchRunner` will stop pairing output rows with input tasks by list
position. Upload will build an exact `fileUri -> task.key` map and pass it to
the output reader. Each output row will use the audio `fileUri` preserved in its
`request` to look up the expected call key for the current chunk. The code will
not derive a key by parsing a blob filename.

Empty and duplicate task keys will be rejected before upload. Unknown or
duplicate output URIs and malformed JSONL rows will be logged as batch
diagnostics. Expected calls without one valid matched row will receive an
explicit `missing response` item error. Vertex `status` errors with a known URI
will become errors for that item. None of these conditions may shift the
mapping of later rows.

### Chunk failure handling

`run_batch_results` will retain completed chunk results. If a chunk still fails
after its configured retries, the calls in that chunk receive error results and
processing continues with the next chunk. The caller therefore receives all
successful results plus explicit failures instead of losing earlier work
through one raised exception.

The existing GCS cleanup remains best-effort. Deletion failures are handled per
object so one undeletable object does not prevent cleanup of the rest.

### Result validation and persistence

Before caching a new scheduler result, the runner will use one shared parser
that accepts plain or markdown-fenced JSON and requires:

- `needs_follow_up` as a boolean;
- `reason` as a string.

Invalid responses are reported as item errors and are not cached. Cache and
usage writes are isolated per item so one storage failure does not abort the
remaining calls. `run_batch_process` returns a small `BatchRunResult` containing
`success`, `partial`, or `failed` and the item counters needed by the scheduler.
Existing cache keys and historical rows remain unchanged.

### Simple scheduler guard

The application will keep APScheduler but register one job per enabled tenant
at application startup, using that tenant's cron or interval settings and one
explicit global `SCHEDULER_TIMEZONE`. The target date is calculated inside the
job in that timezone. A settings change takes effect after an application
restart; per-tenant timezones and dynamic rescheduling are intentionally out of
scope.

A new Supabase `scheduler_runs` table provides an atomic run guard. Its unique
key is:

`tenant_id + scheduled_for + prompt_key + prompt_version + model_key`

The process that inserts the row runs the batch; a uniqueness conflict skips
the duplicate. `scheduled_for` is the planned cron instant or the current
interval bucket, stored in UTC, so interval mode can run more than once per
day. Rows store `running`, `success`, `partial`, or `failed` plus timestamps and
counters. There is no automatic stale-run takeover; an interrupted slot is
retried manually. APScheduler also uses `max_instances=1`.

A small Supabase run repository owns the insert and final status update and is
wired through `build_dependencies()`. The schema includes the unique
constraint and service-role RLS policy. When Supabase or the run guard is
unavailable, scheduled execution fails closed instead of running without
duplicate protection.

The schema addition does not alter `analysis_results`.

## Verification

Focused tests will cover reversed Vertex output, multiple output files,
malformed rows, unknown/duplicate/missing keys, Vertex status errors, partial
chunk failure, invalid model JSON, isolated persistence failures, tenant
schedules, and two simultaneous attempts to claim the same scheduler run.

The existing scheduler, runner, cache, and Vertex tests will run as regression
checks, followed by the complete test suite. A final manual smoke test will use
a small batch of calls with clearly different durations and verify that each
result and token count stays attached to its own call.
