"""Helpers for running Gemini processing via Vertex AI.

**VertexBatchRunner** — true Vertex AI Batch API via GCS:
1. Upload audio files to a GCS bucket.
2. Write a JSONL request file referencing ``gs://`` URIs.
3. Submit ``batches.create`` with ``gcs_uri`` source/destination.
4. Poll until the job reaches a terminal state.
5. Download and parse the JSONL output from GCS.

The legacy ``GeminiBatchRunner`` (sequential one-by-one calls) is
commented out below but retained for reference.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from calls_analyser.domain.exceptions import AIModelError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency wiring
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    types = None  # type: ignore

try:  # pragma: no cover
    from google.cloud import storage as gcs_storage
except Exception:  # pragma: no cover
    gcs_storage = None  # type: ignore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BatchTask:
    """Represents a single audio file queued for processing."""

    key: str
    path: str
    mime_type: str


@dataclass(frozen=True)
class BatchAnalysisResult:
    """Text and optional usage metadata returned for one batch item."""

    text: str
    usage_metadata: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Legacy GeminiBatchRunner (sequential, one-by-one generate_content calls).
# Commented out — replaced by VertexBatchRunner which uses the real Batch API.
# ---------------------------------------------------------------------------
#
# class GeminiBatchRunner:
#     ...  (see git history for full implementation)


# ---------------------------------------------------------------------------
# Vertex AI Batch runner (GCS-based batches.create API) — default
# ---------------------------------------------------------------------------

_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
})


class VertexBatchRunner:
    """Submit audio analysis as a Vertex AI *batch* job via GCS.

    Required:
    * ``GOOGLE_APPLICATION_CREDENTIALS`` — path to service-account JSON
      (or ``GOOGLE_SERVICE_ACCOUNT_JSON`` env var with raw JSON content,
      which ``app.py`` / ``runner.py`` write to a temp file on startup).
    * ``GCS_BATCH_BUCKET`` — GCS bucket name for staging audio and JSONL.
    * ``GOOGLE_CLOUD_PROJECT`` — GCP project id.

    Workflow:
    1. Upload audio files to ``gs://{bucket}/batch_{job_id}/audio/``.
    2. Build JSONL with one ``generateContent`` request per line,
       each referencing the ``gs://`` URI of its audio file.
    3. Upload JSONL to ``gs://{bucket}/batch_{job_id}/input.jsonl``.
    4. Call ``client.batches.create(src=gcs_input_uri, dest=gcs_output_prefix)``.
    5. Poll ``client.batches.get`` until terminal state.
    6. Download output JSONL from GCS, parse responses.
    7. Clean up GCS staging prefix.
    """

    def __init__(
        self,
        model: str,
        project: Optional[str] = None,
        location: Optional[str] = None,
        bucket: Optional[str] = None,
        poll_interval: float = 30.0,
        poll_timeout: float = 3600.0,
    ) -> None:
        if genai is None:
            raise AIModelError("google-genai library is not available")
        if gcs_storage is None:
            raise AIModelError(
                "google-cloud-storage is not installed. "
                "Run: pip install google-cloud-storage"
            )

        self._project = project or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", "canvas-genius-492412-c3",
        )
        self._location = location or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", "global",
        )
        self._bucket_name = bucket or os.environ.get("GCS_BATCH_BUCKET", "")
        if not self._bucket_name:
            raise AIModelError(
                "GCS_BATCH_BUCKET is not configured. "
                "Set it to an existing GCS bucket name."
            )

        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout

        if not model.startswith("publishers/"):
            base_model = model.replace("models/", "").replace("publishers/google/models/", "")
            self._model = f"publishers/google/models/{base_model}"
        else:
            self._model = model

        self._client = genai.Client(
            vertexai=True,
            project=self._project,
            location=self._location,
        )
        self._gcs = gcs_storage.Client(project=self._project)
        self._gcs_bucket = self._gcs.bucket(self._bucket_name)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run_batch(
        self,
        tasks: Iterable[BatchTask],
        prompt_text: str,
        *,
        chunk_size: int = 25,
        max_attempts: int = 2,
    ) -> Dict[str, str]:
        """Upload to GCS, submit batch job(s), poll, return results."""
        results = self.run_batch_results(
            tasks,
            prompt_text,
            chunk_size=chunk_size,
            max_attempts=max_attempts,
        )
        return {
            key: result.text if isinstance(result, BatchAnalysisResult) else str(result)
            for key, result in results.items()
        }

    def run_batch_results(
        self,
        tasks: Iterable[BatchTask],
        prompt_text: str,
        *,
        chunk_size: int = 25,
        max_attempts: int = 2,
    ) -> Dict[str, BatchAnalysisResult]:
        """Upload to GCS, submit batch job(s), poll, return text and usage."""
        pending = list(tasks)
        if not pending:
            return {}

        all_results: Dict[str, BatchAnalysisResult] = {}
        attempts_per_chunk = max(1, max_attempts)

        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start : chunk_start + chunk_size]
            chunk_num = chunk_start // chunk_size + 1
            total_chunks = (len(pending) + chunk_size - 1) // chunk_size
            logger.info(
                "Batch chunk %d/%d (%d tasks)", chunk_num, total_chunks, len(chunk),
            )
            chunk_results: Dict[str, BatchAnalysisResult] = {}
            for attempt in range(1, attempts_per_chunk + 1):
                try:
                    chunk_results = self._run_single_batch(chunk, prompt_text)
                    break
                except Exception as exc:
                    if attempt >= attempts_per_chunk:
                        logger.error(
                            "Batch chunk %d/%d failed after %d attempt(s): %s",
                            chunk_num,
                            total_chunks,
                            attempts_per_chunk,
                            exc,
                        )
                        raise
                    logger.warning(
                        "Batch chunk %d/%d failed on attempt %d/%d: %s. Restarting batch chunk.",
                        chunk_num,
                        total_chunks,
                        attempt,
                        attempts_per_chunk,
                        exc,
                    )
            all_results.update(chunk_results)

        return all_results

    # ------------------------------------------------------------------ #
    # Internal: single batch job lifecycle
    # ------------------------------------------------------------------ #
    def _run_single_batch(
        self,
        tasks: List[BatchTask],
        prompt_text: str,
    ) -> Dict[str, BatchAnalysisResult]:
        job_id = uuid.uuid4().hex[:12]
        gcs_prefix = f"batch_{job_id}"

        try:
            audio_uris = self._upload_audio_to_gcs(tasks, gcs_prefix)
            input_uri = self._write_jsonl_to_gcs(
                tasks,
                audio_uris,
                prompt_text,
                gcs_prefix,
            )
            output_prefix = f"gs://{self._bucket_name}/{gcs_prefix}/output/"

            logger.info(
                "Submitting batch job (input=%s, output=%s)…",
                input_uri,
                output_prefix,
            )
            job = self._client.batches.create(
                model=self._model,
                src=input_uri,
                config=types.CreateBatchJobConfig(
                    dest=output_prefix,
                    display_name=f"calls-analyser-{job_id}",
                ),
            )
            job_name = job.name
            logger.info("Batch job created: %s", job_name)

            self._poll_until_done(job_name)

            return self._read_output_jsonl(tasks, gcs_prefix)
        finally:
            self._cleanup_gcs(gcs_prefix)

    # ------------------------------------------------------------------ #
    # Step 1: upload audio files to GCS
    # ------------------------------------------------------------------ #
    def _upload_audio_to_gcs(
        self,
        tasks: List[BatchTask],
        gcs_prefix: str,
    ) -> Dict[str, str]:
        """Upload local audio → GCS. Returns {task_key: gs://uri}."""
        uris: Dict[str, str] = {}
        for i, task in enumerate(tasks, start=1):
            ext = os.path.splitext(task.path)[1] or ".bin"
            blob_name = f"{gcs_prefix}/audio/{task.key}{ext}"
            logger.info("Uploading %d/%d to GCS: %s", i, len(tasks), blob_name)
            blob = self._gcs_bucket.blob(blob_name)
            blob.upload_from_filename(task.path, content_type=task.mime_type)
            uris[task.key] = f"gs://{self._bucket_name}/{blob_name}"
        return uris

    # ------------------------------------------------------------------ #
    # Step 2: write JSONL request file to GCS
    # ------------------------------------------------------------------ #
    def _write_jsonl_to_gcs(
        self,
        tasks: List[BatchTask],
        audio_uris: Dict[str, str],
        prompt_text: str,
        gcs_prefix: str,
    ) -> str:
        """Build JSONL and upload to GCS. Returns gs:// URI of the JSONL."""
        clean_prompt = (prompt_text or "").strip()
        lines: list[str] = []

        for task in tasks:
            gcs_uri = audio_uris.get(task.key)
            if not gcs_uri:
                continue

            audio_part = {
                "fileData": {
                    "mimeType": task.mime_type or "audio/wav",
                    "fileUri": gcs_uri,
                }
            }

            parts = [audio_part]
            if clean_prompt:
                parts.append({"text": clean_prompt})

            row = {
                "request": {
                    "contents": [{"role": "user", "parts": parts}],
                },
            }

            lines.append(json.dumps(row, ensure_ascii=False))

        jsonl_content = "\n".join(lines)
        blob_name = f"{gcs_prefix}/input.jsonl"
        blob = self._gcs_bucket.blob(blob_name)
        blob.upload_from_string(jsonl_content, content_type="application/jsonl")
        input_uri = f"gs://{self._bucket_name}/{blob_name}"
        logger.info("Wrote %d requests to %s", len(lines), input_uri)
        return input_uri

    # ------------------------------------------------------------------ #
    # Step 3: poll until terminal state
    # ------------------------------------------------------------------ #
    def _poll_until_done(self, job_name: str) -> Any:
        start = time.monotonic()

        while True:
            job = self._client.batches.get(name=job_name)
            state = getattr(job.state, "value", str(job.state)) if job.state else "UNKNOWN"
            elapsed = time.monotonic() - start

            stats = getattr(job, "completion_stats", None)
            stats_str = ""
            if stats:
                stats_str = (
                    f" (ok={getattr(stats, 'successful_count', '?')}"
                    f" fail={getattr(stats, 'failed_count', '?')})"
                )

            logger.info(
                "Batch %s — state: %s (%.0fs)%s",
                job_name, state, elapsed, stats_str,
            )

            if state in _TERMINAL_STATES:
                if state == "JOB_STATE_FAILED":
                    error = getattr(job, "error", None)
                    raise AIModelError(f"Batch job failed: {error}")
                return job

            if elapsed >= self._poll_timeout:
                raise AIModelError(
                    f"Batch job {job_name} timed out after {self._poll_timeout}s"
                )

            time.sleep(self._poll_interval)

    # ------------------------------------------------------------------ #
    # Step 4: read output JSONL from GCS
    # ------------------------------------------------------------------ #
    def _read_output_jsonl(
        self,
        tasks: List[BatchTask],
        gcs_prefix: str,
    ) -> Dict[str, BatchAnalysisResult]:
        results: Dict[str, BatchAnalysisResult] = {}

        output_prefix = f"{gcs_prefix}/output/"
        blobs = list(self._gcs_bucket.list_blobs(prefix=output_prefix))
        jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]

        if not jsonl_blobs:
            logger.warning("No output JSONL files found under %s", output_prefix)
            for task in tasks:
                results[task.key] = BatchAnalysisResult(text="Error: no output file")
            return results

        output_rows: list[dict] = []
        for blob in jsonl_blobs:
            content = blob.download_as_text(encoding="utf-8")
            for line in content.strip().splitlines():
                if not line.strip():
                    continue
                try:
                    output_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        for idx, row in enumerate(output_rows):
            task_key = tasks[idx].key if idx < len(tasks) else ""

            response = row.get("response", {})
            error = row.get("error")
            if error:
                if task_key:
                    results[task_key] = BatchAnalysisResult(text=f"Error: {error}")
                continue

            text = self._extract_text_from_response(response)
            if task_key:
                usage_metadata = self._extract_usage_metadata_from_response(response)
                results[task_key] = BatchAnalysisResult(
                    text=text or "Error: no text in response",
                    usage_metadata=usage_metadata,
                )

        for task in tasks:
            if task.key not in results:
                results[task.key] = BatchAnalysisResult(text="Error: no response received")

        logger.info("Parsed %d results from output JSONL", len(results))
        return results

    @staticmethod
    def _extract_text_from_response(response: dict) -> str:
        candidates = response.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                text = part.get("text")
                if text:
                    return text
        return ""

    @staticmethod
    def _extract_usage_metadata_from_response(response: dict) -> dict[str, int] | None:
        usage = response.get("usageMetadata") or response.get("usage_metadata")
        if not isinstance(usage, dict):
            return None
        keys = (
            "promptTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
            "thoughtsTokenCount",
        )
        result: dict[str, int] = {}
        for key in keys:
            value = usage.get(key)
            if value is None:
                continue
            try:
                result[key] = int(value)
            except (TypeError, ValueError):
                continue
        return result or None

    # ------------------------------------------------------------------ #
    # Cleanup: delete GCS staging prefix
    # ------------------------------------------------------------------ #
    def _cleanup_gcs(self, gcs_prefix: str) -> None:
        """Best-effort deletion of all blobs under the batch prefix."""
        try:
            blobs = list(self._gcs_bucket.list_blobs(prefix=f"{gcs_prefix}/"))
            for blob in blobs:
                blob.delete()
            logger.info("Cleaned up %d GCS objects under %s/", len(blobs), gcs_prefix)
        except Exception as exc:
            logger.warning("GCS cleanup failed for %s: %s", gcs_prefix, exc)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def guess_mime_type(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "application/octet-stream"
