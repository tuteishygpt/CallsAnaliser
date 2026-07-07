from __future__ import annotations

from types import SimpleNamespace

from calls_analyser.domain.models import AnalysisResult, Language, RecordingHandle
from calls_analyser.services.analysis import CacheKey, AnalysisOptions, AnalysisService
from calls_analyser.services.prompt import PromptService, PromptTemplate
from calls_analyser.services.registry import ProviderRegistry
from calls_analyser.ports.ai import AIModelPort, AudioSource
from calls_analyser.services.call_log import CallLogService
from calls_analyser.services.tenant import TenantConfig


class StubCallLogService(CallLogService):  # type: ignore[misc]
    def __init__(self) -> None:
        self.calls = 0

    def ensure_recording(self, unique_id: str, tenant: TenantConfig) -> RecordingHandle:
        self.calls += 1
        return RecordingHandle(unique_id=unique_id, local_uri="/tmp/recording.mp3", source_uri="remote")


class FakeAIModel(AIModelPort):
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt: str | None = None

    def analyze(self, audio: AudioSource, prompt: str, lang: Language, options=None) -> AnalysisResult:
        self.calls += 1
        self.last_prompt = prompt
        return AnalysisResult(text=f"result-{self.calls}", model="fake-model", provider=self.provider_name)


class UsageAIModel(FakeAIModel):
    def analyze(self, audio: AudioSource, prompt: str, lang: Language, options=None) -> AnalysisResult:
        self.calls += 1
        return AnalysisResult(
            text=f"result-{self.calls}",
            model="fake-model",
            provider=self.provider_name,
            metadata={
                "usage_metadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                    "thoughtsTokenCount": 0,
                }
            },
        )


class RecordingUsageTracker:
    def __init__(self) -> None:
        self.calls = []

    def record(self, **kwargs) -> None:  # noqa: ANN003
        self.calls.append(kwargs)


PROMPTS = {
    "simple": PromptTemplate(key="simple", title="simple", body="default prompt"),
}


def test_analysis_service_is_idempotent() -> None:
    registry: ProviderRegistry[AIModelPort] = ProviderRegistry()
    ai = FakeAIModel()
    registry.register("fake-model", ai)
    prompt_service = PromptService(PROMPTS)
    call_log_service = StubCallLogService()
    service = AnalysisService(call_log_service, registry, prompt_service)
    tenant = TenantConfig(tenant_id="tenant", vochi_base_url="https://api")

    options = AnalysisOptions(model_key="fake-model", prompt_key="simple", custom_prompt="custom")
    result1 = service.analyze_call("abc", tenant, Language.ENGLISH, options)
    result2 = service.analyze_call("abc", tenant, Language.ENGLISH, options)

    assert result1.text == "result-1"
    assert result2.text == "result-1"
    assert ai.calls == 1
    assert ai.last_prompt == "custom"
    assert call_log_service.calls == 1


def test_analysis_service_accepts_external_cache() -> None:
    registry: ProviderRegistry[AIModelPort] = ProviderRegistry()
    ai = FakeAIModel()
    registry.register("fake-model", ai)
    prompt_service = PromptService(PROMPTS)
    call_log_service = StubCallLogService()
    cache: dict[CacheKey, AnalysisResult] = {}
    service = AnalysisService(call_log_service, registry, prompt_service, cache=cache)
    tenant = TenantConfig(tenant_id="tenant", vochi_base_url="https://api")

    options = AnalysisOptions(model_key="fake-model", prompt_key="simple", custom_prompt="custom ")
    service.analyze_call("abc", tenant, Language.BELARUSIAN, options)

    expected_key: CacheKey = (
        tenant.tenant_id,
        "abc",
        options.prompt_key,
        1,
        ai.provider_name,
        options.model_key,
        "custom",
    )
    assert expected_key in cache
    assert cache[expected_key].text == "result-1"


def test_analysis_service_records_usage_for_uncached_model_call_only() -> None:
    registry: ProviderRegistry[AIModelPort] = ProviderRegistry()
    ai = UsageAIModel()
    registry.register("fake-model", ai)
    prompt_service = PromptService(PROMPTS)
    call_log_service = StubCallLogService()
    usage_tracker = RecordingUsageTracker()
    service = AnalysisService(
        call_log_service,
        registry,
        prompt_service,
        cache={},
        usage_tracker=usage_tracker,
    )
    tenant = TenantConfig(tenant_id="tenant", vochi_base_url="https://api")
    entry = SimpleNamespace(unique_id="abc", duration_seconds=42, raw={"user": "agent"})
    options = AnalysisOptions(
        model_key="fake-model",
        prompt_key="simple",
        custom_prompt="custom",
        mode="ui_mass",
        call_entry=entry,
    )

    service.analyze_call("abc", tenant, Language.ENGLISH, options)
    service.analyze_call("abc", tenant, Language.ENGLISH, options)

    assert len(usage_tracker.calls) == 1
    call = usage_tracker.calls[0]
    assert call["entry"] is entry
    assert call["tenant"] is tenant
    assert call["prompt_key"] == "simple"
    assert call["custom_fragment"] == "custom"
    assert call["provider_name"] == "fake"
    assert call["model_key"] == "fake-model"
    assert call["mode"] == "ui_mass"
    assert call["usage"].total_token_count == 15
    assert call["cache_key"] == (
        "tenant",
        "abc",
        "simple",
        1,
        "fake",
        "fake-model",
        "custom",
    )
