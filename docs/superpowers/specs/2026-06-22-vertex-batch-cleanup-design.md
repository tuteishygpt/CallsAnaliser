# Vertex Batch Cleanup Design

## Goal

Ensure every Vertex Batch attempt removes its GCS staging prefix, regardless of
whether upload, request creation, polling, or output parsing succeeds.

## Design

`VertexBatchRunner._run_single_batch` will retain the existing lifecycle and
return values, but execute that lifecycle inside `try/finally`. The `finally`
block will call the existing best-effort `_cleanup_gcs` method. Cleanup failures
will remain warnings and will not replace the original batch exception.

The configured bucket itself is not owned by the runner and will not be deleted.
Only objects under the generated `batch_<job-id>/` prefix are in scope.

## Verification

Unit tests will construct the runner without external Google clients and verify
that cleanup is called with the generated prefix after success and after
failures at upload, job creation, polling, and output reading. Existing tests
will then be run as a regression check.
