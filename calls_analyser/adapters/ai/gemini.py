"""Google Gemini AI adapter."""
from __future__ import annotations

import importlib
import logging
import os
import time
from typing import Any, Callable, Mapping, Optional

from calls_analyser.domain.exceptions import AIModelError
from calls_analyser.domain.models import AnalysisResult, Language
from calls_analyser.ports.ai import AIModelPort, AudioSource

logger = logging.getLogger(__name__)

_genai_module = importlib.util.find_spec("google.genai")
if _genai_module is not None:  # pragma: no cover - optional dependency
    genai = importlib.import_module("google.genai")  # type: ignore
else:  # pragma: no cover - optional dependency
    genai = None  # type: ignore


class GeminiAIAdapter(AIModelPort):
    """Adapter around the google-genai client."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        client_factory: Optional[Callable[[Optional[str]], Any]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        self._client_factory = client_factory or self._default_factory
        self._client = self._client_factory(self._api_key)

    def _default_factory(
        self,
        api_key: Optional[str],
    ) -> Any:  # pragma: no cover - requires dependency
        if genai is None:
            raise AIModelError("google-genai library is not available")
        if not api_key:
            raise AIModelError("GOOGLE_API_KEY is not configured")

        return genai.Client(
            vertexai=True,
            api_key=api_key,
        )

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        message = str(exc)
        return "429" in message or "503" in message or "UNAVAILABLE" in message

    def analyze(
        self,
        audio: AudioSource,
        prompt: str,
        lang: Language,
        options: Mapping[str, Any] | None = None,
    ) -> AnalysisResult:
        if not getattr(audio, "path", None) and not getattr(audio, "content", None):
            raise AIModelError("Audio source must provide either a path or content")

        client = self._client
        max_retries = 5

        for attempt in range(max_retries):
            try:
                audio_bytes = getattr(audio, "content", None)
                if not audio_bytes and getattr(audio, "path", None):
                    with open(getattr(audio, "path"), "rb") as f:
                        audio_bytes = f.read()

                if not audio_bytes:
                    raise AIModelError("Audio source must provide either a path or content")

                # Pass audio as inline bytes (limited to 20MB in Vertex AI)
                audio_part = {"inline_data": {"data": audio_bytes, "mime_type": "audio/mpeg"}}

                system_instruction = self._system_instruction(lang)
                merged_prompt = f"[SYSTEM INSTRUCTION: {system_instruction}]\n\n{prompt}"
                
                response = client.models.generate_content(
                    model=self._model,
                    contents=[audio_part, merged_prompt],
                )
                text = getattr(response, "text", None)
                if not text:
                    raise AIModelError("Model returned no text")

                return AnalysisResult(
                    text=text,
                    model=self._model,
                    provider=self.provider_name,
                    metadata={"lang": lang.value, "tenant": (options or {}).get("tenant_id")},
                )
            except Exception as exc:  # pragma: no cover - passthrough in tests via fakes
                is_retryable = self._is_retryable_error(exc)
                if is_retryable and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(
                        "Gemini API error (attempt %s/%s): %s. Retrying in %ss...",
                        attempt + 1,
                        max_retries,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue

                if is_retryable:
                    logger.error("Gemini API failed after %s attempts: %s", max_retries, exc)
                raise AIModelError(f"Gemini call failed: {exc}") from exc

    @staticmethod
    def _system_instruction(lang: Language) -> str:
        if lang is Language.BELARUSIAN:
            return "Reply in Belarusian."
        if lang is Language.RUSSIAN:
            return "Reply in Russian."
        if lang is Language.ENGLISH:
            return "Reply in English."
        return "Reply in the caller's language; if unclear, use concise professional English."
