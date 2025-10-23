from __future__ import annotations

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
    tenant = TenantConfig(tenant_id="tenant", vochi_base_url="https://api", vochi_client_id="client")

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
    tenant = TenantConfig(tenant_id="tenant", vochi_base_url="https://api", vochi_client_id="client")

    options = AnalysisOptions(model_key="fake-model", prompt_key="simple", custom_prompt="custom ")
    service.analyze_call("abc", tenant, Language.BELARUSIAN, options)

    expected_key: CacheKey = (
        tenant.tenant_id,
        "abc",
        options.prompt_key,
        ai.provider_name,
        options.model_key,
        "custom",
    )
    assert expected_key in cache
    assert cache[expected_key].text == "result-1"
