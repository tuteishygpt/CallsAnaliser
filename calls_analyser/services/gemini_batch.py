"""Gemini BATCH API helper."""
from __future__ import annotations
"""Helpers for running Gemini BATCH API jobs."""

from dataclasses import dataclass
import json
import mimetypes
import os
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

from calls_analyser.domain.exceptions import AIModelError

try:  # pragma: no cover - optional dependency wiring
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency wiring
    genai = None  # type: ignore
    types = None  # type: ignore


@dataclass
class BatchTask:
    """Represents a single audio file queued for BATCH processing."""

    key: str
    path: str
    mime_type: str
    file_uri: Optional[str] = None


class GeminiBatchRunner:
    """Create and poll Gemini BATCH jobs for multiple audio files."""

    def __init__(self, api_key: Optional[str], model: str) -> None:
        if genai is None:
            raise AIModelError("google-genai library is not available")
        if not api_key:
            raise AIModelError("GOOGLE_API_KEY is not configured")
        self._api_key = api_key
        self._model = model
        self._client = genai.Client(
            vertexai=True,
    def __init__(self, api_key: str, model: str, project: str, location: str = "global") -> None:
        if genai is None:
            raise AIModelError("google-genai library is not available")
        if not project:
            raise AIModelError("GOOGLE_CLOUD_PROJECT is not configured")
        self._api_key = api_key
        self._model = model
        self._project = project
        self._location = location
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            api_key=api_key,
        )

    def run_batch(
        self,
        tasks: Iterable[BatchTask],
        prompt_text: str,
        *,
        chunk_size: int = 20,
    ) -> Dict[str, str]:
        """Run batch jobs and return mapping ``key -> text``."""

        pending = list(tasks)
        if not pending:
            return {}

        results: Dict[str, str] = {}
        normalized_chunk_size = max(1, int(chunk_size))

        for chunk_idx in range(0, len(pending), normalized_chunk_size):
            chunk = pending[chunk_idx : chunk_idx + normalized_chunk_size]
            self._process_chunk(chunk, chunk_idx // normalized_chunk_size, prompt_text, results)

        return results

    def _process_chunk(
        self, chunk: List[BatchTask], chunk_index: int, prompt_text: str, results: Dict[str, str]
    ) -> None:
        if not chunk:
            return

        uploaded_file_names: List[str] = []
        uploaded_jsonl_name: Optional[str] = None

        try:
            for task in chunk:
                uploaded = self._client.files.upload(file=task.path)
                task.file_uri = uploaded.uri
                task.mime_type = uploaded.mime_type or task.mime_type
                uploaded_file_names.append(uploaded.name)

            with tempfile.TemporaryDirectory() as tmpdir:
                jsonl_path = os.path.join(tmpdir, f"batch_input_{chunk_index:03}.jsonl")
                self._prepare_chunk_jsonl(chunk, jsonl_path, prompt_text, chunk_index)
                uploaded_jsonl = self._upload_jsonl(jsonl_path, f"batch-input-{chunk_index:03}")
                uploaded_jsonl_name = uploaded_jsonl.name

            batch_name = self._create_batch_job_rest(
                model_id=self._model,
                input_file_name=uploaded_jsonl_name,
                display_name=f"audio-batch-{chunk_index:03}",
            )
            dest_file_name = self._poll_batch_job(batch_name)
            file_content = self._client.files.download(file=dest_file_name)
            self._process_results_jsonl_bytes(file_content, results)
        finally:
            for name in uploaded_file_names:
                try:
                    self._client.files.delete(name=name)
                except Exception:  # pragma: no cover - cleanup best effort
                    pass
            if uploaded_jsonl_name:
                try:
                    self._client.files.delete(name=uploaded_jsonl_name)
                except Exception:  # pragma: no cover - cleanup best effort
                    pass

    # ------------------------- JSONL helpers -------------------------
    def _prepare_chunk_jsonl(
        self, tasks_chunk: List[BatchTask], jsonl_path: str, prompt_text: str, chunk_index: int
    ) -> None:
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, task in enumerate(tasks_chunk):
                unique_key = task.key or f"chunk{chunk_index:03}_batch_{i:03}"
                parts = self._build_parts_for_task(task, prompt_text)
                request_entry = {
                    "key": unique_key,
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": parts,
                            }
                        ]
                    },
                }
                f.write(json.dumps(request_entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _build_parts_for_task(task: BatchTask, prompt_text: str) -> List[Dict[str, Any]]:
        clean_prompt = (prompt_text or "").strip()
        parts: List[Dict[str, Any]] = []
        if clean_prompt:
            parts.append({"text": clean_prompt})
        parts.append(
            {
                "file_data": {
                    "mime_type": task.mime_type,
                    "file_uri": task.file_uri,
                }
            }
        )
        return parts

    def _upload_jsonl(self, jsonl_path: str, display_name: str):
        try:
            return self._client.files.upload(
                file=jsonl_path,
                config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
            )
        except Exception:
            return self._client.files.upload(
                file=jsonl_path,
                config=types.UploadFileConfig(display_name=display_name),
            )

    # ------------------------- REST helpers --------------------------
    @staticmethod
    def _rest_model_name(model_id: str) -> str:
        return model_id.replace("models/", "")

    def _create_batch_job_rest(self, model_id: str, input_file_name: str, display_name: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._rest_model_name(model_id)}:batchGenerateContent"
        )
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "batch": {
                "display_name": display_name,
                "input_config": {"file_name": input_file_name},
            }
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if not resp.ok:
            raise AIModelError(f"REST create failed: {resp.status_code} {resp.text}")

        data = resp.json()
        name = data.get("name")
        if not name and isinstance(data.get("batch"), dict):
            name = data["batch"].get("name")
        if not name:
            raise AIModelError(f"REST create succeeded but no batch name found. Response: {data}")

        return name

    def _get_batch_job_rest(self, name: str) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/{name}"
        headers = {"x-goog-api-key": self._api_key}
        resp = requests.get(url, headers=headers, timeout=60)
        if not resp.ok:
            raise AIModelError(f"REST get failed: {resp.status_code} {resp.text}")
        return resp.json()

    @staticmethod
    def _extract_state(rest_obj: Dict[str, Any]) -> Optional[str]:
        return rest_obj.get("state") or (rest_obj.get("metadata") or {}).get("state") or (rest_obj.get("batch") or {}).get("state")

    @staticmethod
    def _extract_result_file_name(rest_obj: Dict[str, Any]) -> Optional[str]:
        resp = rest_obj.get("response") or {}
        dest = resp.get("dest") or {}
        return (
            dest.get("file_name")
            or dest.get("fileName")
            or resp.get("file_name")
            or resp.get("fileName")
            or resp.get("responsesFile")
            or resp.get("responses_file")
        )

    def _poll_batch_job(self, batch_name: str) -> str:
        completed_states = {
            "BATCH_STATE_SUCCEEDED",
            "BATCH_STATE_FAILED",
            "BATCH_STATE_CANCELLED",
            "BATCH_STATE_EXPIRED",
            "BATCH_STATE_PAUSED",
        }

        while True:
            rest_job = self._get_batch_job_rest(batch_name)
            state = self._extract_state(rest_job)
            if state in completed_states:
                break
            time.sleep(30)

        if state != "BATCH_STATE_SUCCEEDED":
            err = rest_job.get("error") or (rest_job.get("response") or {}).get("error")
            raise AIModelError(f"Batch job failed with state {state}: {err}")

        result_file_name = self._extract_result_file_name(rest_job)
        if not result_file_name:
            raise AIModelError("Could not locate result file name in REST response")
        return result_file_name

    # ------------------------- Results processing --------------------
    @staticmethod
    def _process_results_jsonl_bytes(content_bytes: bytes, results: Dict[str, str]) -> None:
        content_str = content_bytes.decode("utf-8", errors="replace")

        for line in content_str.splitlines():
            if not line.strip():
                continue
            try:
                result = json.loads(line)
            except Exception:
                continue

            key = result.get("key")
            if not key:
                continue

            response_wrapper = result.get("response", {})
            if "error" in response_wrapper:
                results[key] = f"Error: {response_wrapper['error']}"
                continue

            candidates = response_wrapper.get("candidates", [])
            text: Optional[str] = None
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                for part in parts:
                    if isinstance(part, dict) and part.get("text"):
                        text = part["text"]
                        break

            if text is None:
                continue

            results[key] = text


def guess_mime_type(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "application/octet-stream"
