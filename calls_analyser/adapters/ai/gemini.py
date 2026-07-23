"""Google Gemini AI adapter – Vertex AI only."""
from __future__ import annotations

import importlib
import logging
import os
import time
from typing import Any, Callable, Mapping, Optional

from calls_analyser.domain.exceptions import AIModelError
from calls_analyser.google_credentials import load_google_credentials
from calls_analyser.domain.models import AnalysisResult, Language
from calls_analyser.ports.ai import AIModelPort, AudioSource
from calls_analyser.services.usage import extract_usage_metadata, usage_metadata_to_dict

logger = logging.getLogger(__name__)

_genai_module = importlib.util.find_spec("google.genai")
if _genai_module is not None:  # pragma: no cover - optional dependency
    genai = importlib.import_module("google.genai")  # type: ignore
    genai_types = importlib.import_module("google.genai.types")  # type: ignore
else:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    genai_types = None  # type: ignore


class GeminiAIAdapter(AIModelPort):
    """Adapter around the google-genai client.

    Only Vertex AI mode is supported (``vertexai=True``).
    Using the Developer-API mode will raise ``AIModelError``.
    """

    provider_name = "gemini"

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        client_factory: Optional[Callable[[Optional[str]], Any]] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        if genai is None:
            raise AIModelError("google-genai library is not available")

        self._api_key = api_key
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "canvas-genius-492412-c3")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

        self._credentials = None if api_key else load_google_credentials()
        has_adc = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

        if api_key and self._project and self._location and not model.startswith("projects/"):
            base_model = model.replace("models/", "")
            self._model = f"projects/{self._project}/locations/{self._location}/publishers/google/models/{base_model}"
        elif not api_key and not model.startswith("publishers/"):
            base_model = model.replace("models/", "")
            self._model = f"publishers/google/models/{base_model}"
        else:
            self._model = model

        if not self._api_key and self._credentials is None and not has_adc:
            raise AIModelError("No Google credentials are configured.")

        self._client_factory = client_factory or self._default_factory
        self._client = self._client_factory(self._api_key)

    def _default_factory(
        self,
        api_key: Optional[str],
    ) -> Any:  # pragma: no cover - requires dependency
        """Create a genai.Client in Vertex AI mode.

        Uses ``api_key`` when available, otherwise falls back to ADC
        (service account via ``GOOGLE_APPLICATION_CREDENTIALS``).
        """
        if api_key:
            logger.info("Creating Vertex AI client with api_key")
            return genai.Client(vertexai=True, api_key=api_key)

        if self._credentials is not None:
            logger.info(
                "Creating Vertex AI client with in-memory service-account credentials "
                "(project=%s, location=%s)",
                self._project,
                self._location,
            )
            return genai.Client(
                vertexai=True,
                project=self._project,
                location=self._location,
                credentials=self._credentials,
            )

        logger.info(
            "Creating Vertex AI client with ADC (project=%s, location=%s)",
            self._project, self._location,
        )
        return genai.Client(
            vertexai=True,
            project=self._project,
            location=self._location,
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

                # Pass audio as inline bytes via Vertex AI (limited to 20 MB)
                audio_part = genai_types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav",
                )

                system_instruction = self._system_instruction(lang)
                merged_prompt = f"[SYSTEM INSTRUCTION: {system_instruction}]\n\n{prompt}"

                response = client.models.generate_content(
                    model=self._model,
                    contents=[audio_part, merged_prompt],
                )
                text = getattr(response, "text", None)
                if not text:
                    raise AIModelError("Model returned no text")

                usage = extract_usage_metadata(
                    getattr(response, "usage_metadata", None)
                    or getattr(response, "usageMetadata", None)
                )
                metadata = {"lang": lang.value, "tenant": (options or {}).get("tenant_id")}
                usage_dict = usage_metadata_to_dict(usage)
                if usage_dict:
                    metadata["usage_metadata"] = usage_dict

                return AnalysisResult(
                    text=text,
                    model=self._model,
                    provider=self.provider_name,
                    metadata=metadata,
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
