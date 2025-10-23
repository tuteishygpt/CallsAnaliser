"""Analysis service coordinating adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping, Optional

from calls_analyser.domain.models import AnalysisResult, Language
from calls_analyser.ports.ai import AIModelPort
from calls_analyser.services.prompt import PromptService
from calls_analyser.services.registry import ProviderRegistry
from calls_analyser.services.tenant import TenantConfig
from calls_analyser.services.call_log import CallLogService


@dataclass
class AnalysisOptions:
    """Options passed to the analysis service."""

    model_key: str
    prompt_key: str
    custom_prompt: Optional[str] = None


@dataclass
class FileAudioSource:
    """Simple audio source implementation for adapters."""

    path: str
    content: bytes | None = None


CacheKey = tuple[str, str, str, str, str, str]


class AnalysisService:
    """Coordinates telephony, storage and AI providers."""

    def __init__(
        self,
        call_log_service: CallLogService,
        ai_registry: ProviderRegistry[AIModelPort],
        prompt_service: PromptService,
        cache: MutableMapping[CacheKey, AnalysisResult] | None = None,
    ) -> None:
        self._call_log_service = call_log_service
        self._ai_registry = ai_registry
        self._prompt_service = prompt_service
        self._cache: MutableMapping[CacheKey, AnalysisResult] = cache if cache is not None else {}

    def analyze_call(
        self,
        unique_id: str,
        tenant: TenantConfig,
        lang: Language,
        options: AnalysisOptions,
    ) -> AnalysisResult:
        """Return an analysis of the call ensuring idempotency."""

        provider = self._ai_registry.get(options.model_key)
        provider_name = getattr(provider, "provider_name", options.model_key)
        custom_fragment = (options.custom_prompt or "").strip()
        cache_key: CacheKey = (
            tenant.tenant_id,
            unique_id,
            options.prompt_key,
            provider_name,
            options.model_key,
            custom_fragment,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        handle = self._call_log_service.ensure_recording(unique_id, tenant)

        prompt_template = self._prompt_service.get_prompt(options.prompt_key)
        prompt_body = options.custom_prompt.strip() if options.custom_prompt else prompt_template.body

        audio_source = FileAudioSource(path=handle.local_uri)
        result = provider.analyze(audio_source, prompt_body, lang, options={"tenant_id": tenant.tenant_id})
        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        """Remove cached analysis results."""

        self._cache.clear()
