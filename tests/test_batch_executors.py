from __future__ import annotations

from calls_analyser.domain.models import AnalysisResult, CallLogEntry, Language
from calls_analyser.ports.ai import AIModelPort, AudioSource
from calls_analyser.services.analysis import AnalysisService
from calls_analyser.services.batch_executors import SequentialBatchExecutor, VertexBatchExecutor
from calls_analyser.services.batch_orchestrator import BatchAnalysisOrchestrator, RoundSpec
from calls_analyser.services.gemini_batch import BatchAnalysisResult
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


class BulkCache(RecordingCache):
    def __init__(self) -> None:
        super().__init__()
        self.bulk_calls = []

    def get_many(self, keys):  # noqa: ANN001
        keys = list(keys)
        self.bulk_calls.append(keys)
        return {key: self[key] for key in keys if key in self}

    def get(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Vertex executor must use a bulk cache lookup")


class VertexCallLog:
    def __init__(self) -> None:
        self.prepared = []

    def ensure_recording(self, unique_id, tenant):  # noqa: ANN001
        self.prepared.append((tenant.tenant_id, unique_id))
        if unique_id == "prep-error":
            raise RuntimeError("audio unavailable")
        return type("Handle", (), {"local_uri": f"/audio/{unique_id}.mp3"})()


class VertexRunner:
    instances = []
    outputs = {}

    def __init__(self, model):  # noqa: ANN001
        self.model = model
        self.calls = []
        self.__class__.instances.append(self)

    def run_batch_results(self, tasks, prompt, *, chunk_size):  # noqa: ANN001
        tasks = list(tasks)
        self.calls.append((tasks, prompt, chunk_size))
        return {key: value for key, value in self.outputs.items() if key in {t.key for t in tasks}}


def make_vertex_executor(cache, call_log, usage, *, batch_size=7):  # noqa: ANN001, ANN201
    VertexRunner.instances = []
    VertexRunner.outputs = {}
    service = make_executor(cache=cache)[0]._analysis_service  # noqa: SLF001
    service._call_log_service = call_log  # noqa: SLF001
    service._usage_tracker = usage  # noqa: SLF001
    return VertexBatchExecutor(
        service,
        runner_factory=VertexRunner,
        batch_size_resolver=lambda _tenant: batch_size,
    )


def test_vertex_executor_bulk_partitions_prepares_and_batches_with_language() -> None:
    cache = BulkCache()
    call_log = VertexCallLog()
    usage = RecordingUsageTracker()
    executor = make_vertex_executor(cache, call_log, usage, batch_size=3)
    tenant = TenantConfig("one", "https://api")
    round_spec = spec(mode="scheduler_batch")
    cached_key = ("one", "cached", "prompt", 1, "fake", "model", "")
    cache[cached_key] = AnalysisResult(text="cached-text", model="old", provider="fake")
    VertexRunner.outputs = {
        "fresh": BatchAnalysisResult("fresh-text", {"totalTokenCount": 9}),
    }

    results = executor.execute(
        [CallLogEntry(unique_id="cached"), CallLogEntry(unique_id="fresh")],
        tenant,
        round_spec,
    )

    assert len(cache.bulk_calls) == 1
    assert call_log.prepared == [("one", "fresh")]
    assert results["cached"].from_cache is True
    assert results["fresh"].raw_text == "fresh-text"
    assert len(VertexRunner.instances) == 1
    tasks, prompt, chunk_size = VertexRunner.instances[0].calls[0]
    assert [(task.key, task.path, task.mime_type) for task in tasks] == [
        ("fresh", "/audio/fresh.mp3", "audio/mpeg"),
    ]
    assert prompt == "[SYSTEM INSTRUCTION: Reply in English.]\n\nbody"
    assert chunk_size == 3
    assert usage.calls[0]["mode"] == "scheduler_batch"


def test_vertex_executor_partial_terminal_errors_bypass_and_verification_mode() -> None:
    cache = BulkCache()
    call_log = VertexCallLog()
    usage = RecordingUsageTracker()
    executor = make_vertex_executor(cache, call_log, usage)
    tenant = TenantConfig("one", "https://api")
    cached_key = ("one", "cached", "prompt", 1, "fake", "model", "")
    cache[cached_key] = AnalysisResult(text="stale", model="old", provider="fake")
    VertexRunner.outputs = {
        "cached": BatchAnalysisResult("new", {"totalTokenCount": 2}),
        "terminal": BatchAnalysisResult("Error: rejected"),
    }
    verify = spec(stage="verification", mode="scheduler_batch_verify")

    results = executor.execute(
        [
            CallLogEntry(unique_id="cached"),
            CallLogEntry(unique_id="terminal"),
            CallLogEntry(unique_id="missing"),
            CallLogEntry(unique_id="prep-error"),
        ],
        tenant,
        verify,
        bypass_cache=True,
    )

    assert set(results) == {"cached", "terminal", "prep-error"}
    assert results["cached"].from_cache is False
    assert results["terminal"].execution_status == "error"
    assert results["terminal"].execution_error == "Error: rejected"
    assert results["prep-error"].execution_error == "audio unavailable"
    assert [call["mode"] for call in usage.calls] == ["scheduler_batch_verify"]


def test_vertex_executor_validation_is_durable_and_execution_scoped() -> None:
    cache = BulkCache()
    call_log = VertexCallLog()
    executor = make_vertex_executor(cache, call_log, RecordingUsageTracker())
    shared_spec = spec(mode="scheduler_batch")
    VertexRunner.outputs = {"same": BatchAnalysisResult("first")}
    first = executor.execute([CallLogEntry(unique_id="same")], TenantConfig("one", "url"), shared_spec)
    VertexRunner.outputs = {"same": BatchAnalysisResult("second")}
    second = executor.execute([CallLogEntry(unique_id="same")], TenantConfig("two", "url"), shared_spec)

    executor.record_validation(
        shared_spec,
        BatchAnalysisOrchestrator._validation_mapping(second, lambda _r: (None, "invalid")),  # noqa: SLF001
    )
    executor.record_validation(
        shared_spec,
        BatchAnalysisOrchestrator._validation_mapping(first, lambda _r: (None, "valid")),  # noqa: SLF001
    )

    validation_writes = cache.writes[2:]
    assert [key[0] for key, _metadata in validation_writes] == ["two", "one"]
    assert [metadata["decision_valid"] for _key, metadata in validation_writes] == [False, True]
    assert all(metadata["batch_execution"] == "vertex_batch" for _key, metadata in validation_writes)


def test_vertex_executor_turns_runner_construction_failure_into_per_item_errors() -> None:
    cache = BulkCache()
    call_log = VertexCallLog()
    service = make_executor(cache=cache)[0]._analysis_service  # noqa: SLF001
    service._call_log_service = call_log  # noqa: SLF001

    def unavailable_runner(_model):  # noqa: ANN001, ANN202
        raise RuntimeError("Vertex unavailable")

    executor = VertexBatchExecutor(service, runner_factory=unavailable_runner)

    results = executor.execute(
        [CallLogEntry(unique_id="one"), CallLogEntry(unique_id="two")],
        TenantConfig("tenant", "https://api"),
        spec(mode="scheduler_batch"),
    )

    assert set(results) == {"one", "two"}
    assert all(result.execution_status == "error" for result in results.values())
    assert all(result.execution_error == "Vertex unavailable" for result in results.values())
