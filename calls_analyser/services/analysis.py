"""Analysis service coordinating adapters."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, MutableMapping, Optional

from calls_analyser.domain.models import AnalysisResult, Language
from calls_analyser.ports.ai import AIModelPort
from calls_analyser.services.prompt import PromptService
from calls_analyser.services.registry import ProviderRegistry
from calls_analyser.services.tenant import TenantConfig
from calls_analyser.services.call_log import CallLogService
from calls_analyser.services.usage import extract_usage_metadata


@dataclass
class AnalysisOptions:
    """Options passed to the analysis service."""

    model_key: str
    prompt_key: str
    custom_prompt: Optional[str] = None
    mode: str = "ui_direct"
    call_entry: Any = None
    bypass_cache: bool = False
    batch_stage: str | None = None
    batch_execution: str | None = None


@dataclass
class FileAudioSource:
    """Simple audio source implementation for adapters."""

    path: str
    content: bytes | None = None


CacheKey = tuple[str, str, str, int, str, str, str]


class AnalysisService:
    """Coordinates telephony, storage and AI providers."""

    def __init__(
        self,
        call_log_service: CallLogService,
        ai_registry: ProviderRegistry[AIModelPort],
        prompt_service: PromptService,
        cache: MutableMapping[CacheKey, AnalysisResult] | None = None,
        usage_tracker: Any = None,
    ) -> None:
        self._call_log_service = call_log_service
        self._ai_registry = ai_registry
        self._prompt_service = prompt_service
        self._cache: MutableMapping[CacheKey, AnalysisResult] = cache if cache is not None else {}
        self._usage_tracker = usage_tracker

    def analyze_call(
        self,
        unique_id: str,
        tenant: TenantConfig,
        lang: Language,
        options: AnalysisOptions,
    ) -> AnalysisResult:
        """Return an analysis of the call ensuring idempotency."""

        result, _from_cache, _cache_key = self.analyze_call_with_status(
            unique_id, tenant, lang, options,
        )
        return result

    def analyze_call_with_status(
        self,
        unique_id: str,
        tenant: TenantConfig,
        lang: Language,
        options: AnalysisOptions,
    ) -> tuple[AnalysisResult, bool, CacheKey]:
        """Analyze a call and expose cache status for execution adapters."""

        provider = self._ai_registry.get(options.model_key)
        provider_name = getattr(provider, "provider_name", options.model_key)
        custom_fragment = (options.custom_prompt or "").strip()
        prompt_template = self._prompt_service.get_prompt(
            options.prompt_key,
            tenant_id=tenant.tenant_id,
        )
        cache_key: CacheKey = (
            tenant.tenant_id,
            unique_id,
            options.prompt_key,
            prompt_template.version,
            provider_name,
            options.model_key,
            custom_fragment,
        )
        if not options.bypass_cache and cache_key in self._cache:
            return self._cache[cache_key], True, cache_key

        handle = self._call_log_service.ensure_recording(unique_id, tenant)

        prompt_body = custom_fragment or prompt_template.body

        audio_source = FileAudioSource(path=handle.local_uri)
        result = provider.analyze(audio_source, prompt_body, lang, options={"tenant_id": tenant.tenant_id})
        if options.batch_stage is not None:
            result.metadata["batch_stage"] = options.batch_stage
            result.metadata["decision_valid"] = False
        if options.batch_execution is not None:
            result.metadata["batch_execution"] = options.batch_execution
        self._cache[cache_key] = result
        self._record_usage(
            result=result,
            options=options,
            tenant=tenant,
            unique_id=unique_id,
            provider_name=provider_name,
            custom_fragment=custom_fragment,
            cache_key=cache_key,
        )
        return result, False, cache_key

    def clear_cache(self) -> None:
        """Remove cached analysis results."""

        self._cache.clear()

    def _record_usage(
        self,
        *,
        result: AnalysisResult,
        options: AnalysisOptions,
        tenant: TenantConfig,
        unique_id: str,
        provider_name: str,
        custom_fragment: str,
        cache_key: CacheKey,
    ) -> None:
        if self._usage_tracker is None:
            return
        usage = extract_usage_metadata(result.metadata.get("usage_metadata"))
        if usage is None:
            return
        entry = options.call_entry or SimpleNamespace(unique_id=unique_id, raw={})
        self._usage_tracker.record(
            entry=entry,
            tenant=tenant,
            prompt_key=options.prompt_key,
            custom_fragment=custom_fragment,
            provider_name=provider_name,
            model_key=options.model_key,
            mode=options.mode,
            usage=usage,
            cache_key=cache_key,
        )
