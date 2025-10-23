"""Google Gemini AI adapter."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import importlib

_genai_module = importlib.util.find_spec("google.genai")
if _genai_module is not None:  # pragma: no cover - optional dependency
    genai = importlib.import_module("google.genai")  # type: ignore
else:  # pragma: no cover - optional dependency
    genai = None  # type: ignore

from calls_analyser.domain.exceptions import AIModelError
from calls_analyser.domain.models import AnalysisResult, Language
from calls_analyser.ports.ai import AIModelPort, AudioSource


class GeminiAIAdapter(AIModelPort):
    """Adapter around the google-genai client."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        client_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client_factory = client_factory or self._default_factory
        self._client = self._client_factory(api_key)

    def _default_factory(self, api_key: str) -> Any:  # pragma: no cover - requires dependency
        if genai is None:
            raise AIModelError("google-genai library is not available")
        return genai.Client(api_key=api_key)

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
        uploaded_name: Optional[str] = None
        try:
            if getattr(audio, "path", None):
                uploaded = client.files.upload(file=getattr(audio, "path"))
            else:
                uploaded = client.files.upload_bytes(
                    file=getattr(audio, "content"),
                    mime_type="audio/mpeg",
                )
            uploaded_name = getattr(uploaded, "name", None)
            system_instruction = self._system_instruction(lang)
            merged_prompt = f"[SYSTEM INSTRUCTION: {system_instruction}]\n\n{prompt}"
            response = client.models.generate_content(
                model=self._model,
                contents=[uploaded, merged_prompt],
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
            raise AIModelError(f"Gemini call failed: {exc}") from exc
        finally:
            if uploaded_name:
                try:
                    client.files.delete(name=uploaded_name)
                except Exception:  # pragma: no cover - cleanup best effort
                    pass

    @staticmethod
    def _system_instruction(lang: Language) -> str:
        if lang is Language.BELARUSIAN:
            return "Reply in Belarusian."
        if lang is Language.RUSSIAN:
            return "Reply in Russian."
        if lang is Language.ENGLISH:
            return "Reply in English."
        return "Reply in the caller's language; if unclear, use concise professional English."
