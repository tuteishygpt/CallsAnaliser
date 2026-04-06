"""Helpers for running Gemini BATCH processing via Vertex AI.

Processes audio files sequentially using ``types.Part.from_bytes`` and
``client.models.generate_content`` — no GCS or Batch API required.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from calls_analyser.domain.exceptions import AIModelError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency wiring
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency wiring
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency wiring
    genai = None  # type: ignore
    types = None  # type: ignore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BatchTask:
    """Represents a single audio file queued for processing."""

    key: str
    path: str
    mime_type: str


# ---------------------------------------------------------------------------
# Batch runner  (sequential, inline bytes, Vertex AI only)
# ---------------------------------------------------------------------------
class GeminiBatchRunner:
    """Process multiple audio files via Vertex AI.

    Each file is sent inline as ``types.Part.from_bytes`` — exactly the
    same approach as a single-file call.  No GCS bucket or Batch API is
    needed.

    Required environment / params:
    * ``GOOGLE_API_KEY`` – Vertex AI API key (or pass *api_key*).
    * ``GOOGLE_CLOUD_PROJECT`` – GCP project id (optional, read from env).
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        project: Optional[str] = None,
        location: str = "global",
    ) -> None:
        if genai is None:
            raise AIModelError("google-genai library is not available")
        if not api_key:
            raise AIModelError(
                "GOOGLE_API_KEY is not configured. "
                "Vertex AI requires a valid API key."
            )

        self._api_key = api_key
        self._model = model
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location = location

        # Vertex AI only — no Developer API
        client_kwargs: dict[str, Any] = {
            "vertexai": True,
            "api_key": api_key,
        }
        if self._project:
            client_kwargs["project"] = self._project
        if self._location:
            client_kwargs["location"] = self._location

        logger.info(
            "Creating Vertex AI client (project=%s, location=%s)",
            self._project or "<env>",
            self._location,
        )
        self._client = genai.Client(**client_kwargs)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run_batch(
        self,
        tasks: Iterable[BatchTask],
        prompt_text: str,
        *,
        chunk_size: int = 20,
        max_retries: int = 5,
    ) -> Dict[str, str]:
        """Process *tasks* sequentially and return ``{key: text}``.

        *chunk_size* is accepted for API compatibility but is not used —
        all files are processed one-by-one via inline bytes.
        """
        pending = list(tasks)
        if not pending:
            return {}

        results: Dict[str, str] = {}

        for i, task in enumerate(pending, start=1):
            logger.info(
                "Processing %d/%d: %s", i, len(pending), task.key,
            )
            text = self._process_single(task, prompt_text, max_retries)
            if text is not None:
                results[task.key] = text

        return results

    # ------------------------------------------------------------------ #
    # Single-file processing (with retries)
    # ------------------------------------------------------------------ #
    def _process_single(
        self,
        task: BatchTask,
        prompt_text: str,
        max_retries: int,
    ) -> Optional[str]:
        """Send one audio file to Gemini and return the response text."""
        for attempt in range(max_retries):
            try:
                with open(task.path, "rb") as f:
                    audio_bytes = f.read()

                audio_part = types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=task.mime_type or "audio/wav",
                )

                clean_prompt = (prompt_text or "").strip()
                contents: list[Any] = [audio_part]
                if clean_prompt:
                    contents.append(clean_prompt)

                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                )

                text = getattr(response, "text", None)
                if text:
                    return text

                logger.warning(
                    "Empty response for %s (attempt %d/%d)",
                    task.key, attempt + 1, max_retries,
                )
                return None

            except Exception as exc:
                if self._is_retryable(exc) and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(
                        "Retryable error for %s (attempt %d/%d): %s. "
                        "Waiting %ds…",
                        task.key, attempt + 1, max_retries, exc, wait_time,
                    )
                    time.sleep(wait_time)
                    continue

                error_msg = f"Error: {exc}"
                logger.error(
                    "Failed %s after %d attempt(s): %s",
                    task.key, attempt + 1, exc,
                )
                return error_msg

        return None

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        msg = str(exc)
        return "429" in msg or "503" in msg or "UNAVAILABLE" in msg


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def guess_mime_type(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "application/octet-stream"
