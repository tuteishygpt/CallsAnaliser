from __future__ import annotations

from calls_analyser.domain.models import AnalysisResult, CallLogEntry, Language
from calls_analyser.ports.ai import AIModelPort, AudioSource
from calls_analyser.services.analysis import AnalysisService
from calls_analyser.services.batch_executors import SequentialBatchExecutor
from calls_analyser.services.batch_orchestrator import RoundSpec
from calls_analyser.services.prompt import PromptService, PromptTemplate
from calls_analyser.services.registry import ProviderRegistry
from calls_analyser.services.tenant import TenantConfig

from test_analysis_service import RecordingUsageTracker, StubCallLogService


class QueueAI(AIModelPort):
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, audio: AudioSource, prompt: str, lang: Language, options=None) -> AnalysisResult:
        self.calls += 1
        if "bad" in audio.path:
            raise RuntimeError("provider failed")
        return AnalysisResult(
            text=f"raw-{self.calls}", model="resolved-model", provider="fake",
            metadata={"usage_metadata": {"totalTokenCount": 3}},
        )


class EntryCallLog(StubCallLogService):
    def ensure_recording(self, unique_id, tenant):  # noqa: ANN001
        handle = super().ensure_recording(unique_id, tenant)
        if unique_id == "bad":
            return handle.model_copy(update={"local_uri": "/tmp/bad.mp3"})
        return handle


def make_executor():  # noqa: ANN201
    registry: ProviderRegistry[AIModelPort] = ProviderRegistry()
    ai = QueueAI()
    registry.register("model", ai)
    usage = RecordingUsageTracker()
    service = AnalysisService(
        EntryCallLog(), registry,
        PromptService({
            "prompt": PromptTemplate(key="prompt", title="p", body="body"),
            "simple": PromptTemplate(key="simple", title="s", body="fallback"),
        }),
        cache={}, usage_tracker=usage,
    )
    return SequentialBatchExecutor(service), ai, usage


def spec(*, stage="primary", mode="ui_mass"):  # noqa: ANN201
    return RoundSpec(
        model_key="model", prompt_key="prompt", prompt_text="body", prompt_version=1,
        custom_fragment="", language="en", usage_mode=mode, stage_name=stage,
        provider="fake", model_identity="resolved-model", cache_identity="identity",
    )


def test_sequential_executor_returns_raw_results_cache_hits_errors_and_progress() -> None:
    executor, ai, _usage = make_executor()
    tenant = TenantConfig(tenant_id="one", vochi_base_url="https://api")
    entries = [CallLogEntry(unique_id="ok"), CallLogEntry(unique_id="bad")]
    progress = []

    first = executor.execute(entries, tenant, spec(), progress=lambda *args: progress.append(args))
    cached = executor.execute(entries[:1], tenant, spec())
    executor.record_validation(spec(), {"ok": True})

    assert first["ok"].raw_text == "raw-1"
    assert first["ok"].execution_status == "success"
    assert first["ok"].from_cache is False
    assert first["bad"].execution_status == "error"
    assert first["bad"].execution_error == "provider failed"
    assert cached["ok"].from_cache is True
    assert executor._latest_results["primary"]["ok"].metadata["batch_stage"] == "primary"  # noqa: SLF001
    assert executor._latest_results["primary"]["ok"].metadata["batch_execution"] == "ui_sequential"  # noqa: SLF001
    assert executor._latest_results["primary"]["ok"].metadata["decision_valid"] is True  # noqa: SLF001
    assert ai.calls == 2
    assert [(p[0], p[2], p[3]) for p in progress] == [("ok", 1, 2), ("bad", 2, 2)]


def test_sequential_executor_records_stage_modes_and_validation_acknowledgement() -> None:
    executor, _ai, usage = make_executor()
    tenant = TenantConfig(tenant_id="one", vochi_base_url="https://api")
    primary = executor.execute([CallLogEntry(unique_id="a")], tenant, spec())
    verify_spec = spec(stage="verification", mode="ui_mass_verify")
    verification = executor.execute([CallLogEntry(unique_id="b")], tenant, verify_spec)

    executor.record_validation(spec(), {"a": True})
    executor.record_validation(verify_spec, {"b": False})

    assert [call["mode"] for call in usage.calls] == ["ui_mass", "ui_mass_verify"]
    assert primary["a"].raw_text == "raw-1"
    assert verification["b"].raw_text == "raw-2"
    assert primary["a"].cache_identity == "identity"
    assert verification["b"].usage_metadata == {"totalTokenCount": 3}
    assert executor._latest_results["primary"]["a"].metadata == {  # noqa: SLF001
        "usage_metadata": {"totalTokenCount": 3},
        "batch_stage": "primary",
        "batch_execution": "ui_sequential",
        "decision_valid": True,
    }
    assert executor._latest_results["verification"]["b"].metadata["decision_valid"] is False  # noqa: SLF001


def test_sequential_executor_isolates_same_identity_between_tenants() -> None:
    executor, ai, _usage = make_executor()
    entry = [CallLogEntry(unique_id="same")]

    one = executor.execute(entry, TenantConfig("one", "https://api"), spec())
    two = executor.execute(entry, TenantConfig("two", "https://api"), spec())

    assert ai.calls == 2
    assert one["same"].cache_key != two["same"].cache_key
