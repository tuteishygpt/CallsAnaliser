from __future__ import annotations

from calls_analyser.domain.models import AnalysisResult, CallLogEntry, Language
from calls_analyser.ports.ai import AIModelPort, AudioSource
from calls_analyser.services.analysis import AnalysisService
from calls_analyser.services.batch_executors import SequentialBatchExecutor
from calls_analyser.services.batch_orchestrator import BatchAnalysisOrchestrator, RoundSpec
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


def make_executor(*, cache=None):  # noqa: ANN001, ANN201
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
        cache={} if cache is None else cache, usage_tracker=usage,
    )
    return SequentialBatchExecutor(service), ai, usage


class RecordingCache(dict):
    def __init__(self) -> None:
        super().__init__()
        self.writes = []

    def __setitem__(self, key, value) -> None:  # noqa: ANN001
        self.writes.append((key, value.metadata.copy()))
        super().__setitem__(key, value)


def spec(*, stage="primary", mode="ui_mass"):  # noqa: ANN201
    return RoundSpec(
        model_key="model", prompt_key="prompt", prompt_text="body", prompt_version=1,
        custom_fragment="", language="en", usage_mode=mode, stage_name=stage,
        provider="fake", model_identity="resolved-model", cache_identity="identity",
    )


def test_sequential_executor_returns_raw_results_cache_hits_errors_and_progress() -> None:
    cache = RecordingCache()
    executor, ai, _usage = make_executor(cache=cache)
    tenant = TenantConfig(tenant_id="one", vochi_base_url="https://api")
    entries = [CallLogEntry(unique_id="ok"), CallLogEntry(unique_id="bad")]
    progress = []
    round_spec = spec()

    first = executor.execute(entries, tenant, round_spec, progress=lambda *args: progress.append(args))
    cached = executor.execute(entries[:1], tenant, round_spec)
    executor.record_validation(round_spec, {"ok": True})

    assert first["ok"].raw_text == "raw-1"
    assert first["ok"].execution_status == "success"
    assert first["ok"].from_cache is False
    assert first["bad"].execution_status == "error"
    assert first["bad"].execution_error == "provider failed"
    assert cached["ok"].from_cache is True
    cached_result = cache[cached["ok"].cache_key]
    assert cached_result.metadata["batch_stage"] == "primary"
    assert cached_result.metadata["batch_execution"] == "ui_sequential"
    assert cached_result.metadata["decision_valid"] is True
    assert ai.calls == 2
    assert [(p[0], p[2], p[3]) for p in progress] == [("ok", 1, 2), ("bad", 2, 2)]


def test_sequential_executor_records_stage_modes_and_validation_acknowledgement() -> None:
    cache = RecordingCache()
    executor, _ai, usage = make_executor(cache=cache)
    tenant = TenantConfig(tenant_id="one", vochi_base_url="https://api")
    primary_spec = spec()
    primary = executor.execute([CallLogEntry(unique_id="a")], tenant, primary_spec)
    verify_spec = spec(stage="verification", mode="ui_mass_verify")
    verification = executor.execute([CallLogEntry(unique_id="b")], tenant, verify_spec)

    executor.record_validation(primary_spec, {"a": True})
    executor.record_validation(verify_spec, {"b": False})

    assert [call["mode"] for call in usage.calls] == ["ui_mass", "ui_mass_verify"]
    assert primary["a"].raw_text == "raw-1"
    assert verification["b"].raw_text == "raw-2"
    assert primary["a"].cache_identity == "identity"
    assert verification["b"].usage_metadata == {"totalTokenCount": 3}
    assert cache[primary["a"].cache_key].metadata == {
        "usage_metadata": {"totalTokenCount": 3},
        "batch_stage": "primary",
        "batch_execution": "ui_sequential",
        "decision_valid": True,
    }
    assert cache[verification["b"].cache_key].metadata["decision_valid"] is False


def test_validation_acknowledgement_upserts_metadata_to_durable_cache() -> None:
    cache = RecordingCache()
    executor, _ai, _usage = make_executor(cache=cache)
    tenant = TenantConfig(tenant_id="one", vochi_base_url="https://api")
    round_spec = spec()

    executor.execute([CallLogEntry(unique_id="a")], tenant, round_spec)
    assert len(cache.writes) == 1

    executor.record_validation(round_spec, {"a": True})

    assert len(cache.writes) == 2
    assert cache.writes[-1][1]["decision_valid"] is True


def test_interleaved_validation_is_scoped_to_its_round_execution() -> None:
    cache = RecordingCache()
    executor, _ai, _usage = make_executor(cache=cache)
    entry = [CallLogEntry(unique_id="same")]
    first_spec = spec()
    second_spec = spec()

    executor.execute(entry, TenantConfig("one", "https://api"), first_spec)
    executor.execute(entry, TenantConfig("two", "https://api"), second_spec)
    executor.record_validation(first_spec, {"same": True})
    executor.record_validation(second_spec, {"same": False})

    validation_writes = cache.writes[2:]
    assert [write[0][0] for write in validation_writes] == ["one", "two"]
    assert [write[1]["decision_valid"] for write in validation_writes] == [True, False]


def test_same_spec_out_of_order_validation_targets_concrete_execution() -> None:
    cache = RecordingCache()
    executor, _ai, _usage = make_executor(cache=cache)
    entry = [CallLogEntry(unique_id="same")]
    shared_spec = spec()

    first = executor.execute(entry, TenantConfig("one", "https://api"), shared_spec)
    second = executor.execute(entry, TenantConfig("two", "https://api"), shared_spec)
    parse = lambda result: (  # noqa: E731
        None,
        "invalid" if result.raw_text == "raw-2" else "valid",
    )
    second_validation = BatchAnalysisOrchestrator._validation_mapping(second, parse)  # noqa: SLF001
    first_validation = BatchAnalysisOrchestrator._validation_mapping(first, parse)  # noqa: SLF001

    executor.record_validation(shared_spec, second_validation)
    executor.record_validation(shared_spec, first_validation)

    validation_writes = cache.writes[2:]
    assert [write[0][0] for write in validation_writes] == ["two", "one"]
    assert [write[1]["decision_valid"] for write in validation_writes] == [False, True]


def test_sequential_executor_isolates_same_identity_between_tenants() -> None:
    executor, ai, _usage = make_executor()
    entry = [CallLogEntry(unique_id="same")]

    one = executor.execute(entry, TenantConfig("one", "https://api"), spec())
    two = executor.execute(entry, TenantConfig("two", "https://api"), spec())

    assert ai.calls == 2
    assert one["same"].cache_key != two["same"].cache_key
