from __future__ import annotations

from dataclasses import astuple

import pytest

from calls_analyser.domain.models import AnalysisResult, CallLogEntry, Language
from calls_analyser.ports.ai import AIModelPort, AudioSource
from calls_analyser.services.analysis import AnalysisService
from calls_analyser.services.batch_executors import SequentialBatchExecutor, VertexBatchExecutor
from calls_analyser.services.batch_orchestrator import BatchAnalysisOrchestrator, RoundSpec
from calls_analyser.services.gemini_batch import BatchAnalysisResult
from calls_analyser.services.prompt import PromptService, PromptTemplate
from calls_analyser.services.registry import ProviderRegistry
from calls_analyser.services.tenant import TenantConfig


PRIMARY_TRUE = '{"needs_follow_up": true, "reason": "initial yes"}'
PRIMARY_FALSE = '{"needs_follow_up": false, "reason": "initial no"}'
VERIFY_TRUE = '{"needs_follow_up": true, "reason": "confirmed"}'
VERIFY_FALSE = '{"needs_follow_up": false, "reason": "cleared"}'
LEGACY_TRUE = "Needs follow-up: Yes\nSummary: historical"


class _CallLog:
    def ensure_recording(self, unique_id, _tenant):  # noqa: ANN001
        return type("Handle", (), {"local_uri": f"/audio/{unique_id}.mp3"})()


class _PromptAI(AIModelPort):
    provider_name = "fake"

    def __init__(self, responses):  # noqa: ANN001
        self.responses = responses
        self.calls = []

    def analyze(self, audio: AudioSource, prompt: str, lang: Language, options=None):  # noqa: ANN001, ANN201
        unique_id = audio.path.rsplit("/", 1)[-1].removesuffix(".mp3")
        self.calls.append((unique_id, prompt))
        return AnalysisResult(
            text=self.responses[(prompt, unique_id)],
            model="resolved-model",
            provider=self.provider_name,
        )


class _VertexRunner:
    responses = {}
    calls = []

    def __init__(self, model):  # noqa: ANN001
        self.model = model

    def run_batch_results(self, tasks, prompt, *, chunk_size):  # noqa: ANN001
        stage_prompt = prompt.rsplit("\n\n", 1)[-1]
        tasks = list(tasks)
        self.__class__.calls.append((self.model, stage_prompt, [task.key for task in tasks]))
        return {
            task.key: BatchAnalysisResult(self.responses[(stage_prompt, task.key)])
            for task in tasks
        }


def _spec(stage: str, *, model="model", version=1):  # noqa: ANN001, ANN201
    prompt_key = stage
    return RoundSpec(
        model_key=model,
        prompt_key=prompt_key,
        prompt_text=f"{stage}-v{version}",
        prompt_version=version,
        custom_fragment="",
        language="en",
        usage_mode="ui_mass" if stage == "primary" else "ui_mass_verify",
        stage_name=stage,
        provider="fake",
        model_identity="resolved-model",
        cache_identity=f"{model}:{prompt_key}:{version}",
    )


def _service(responses, cache, *, verification_version=1):  # noqa: ANN001, ANN201
    registry = ProviderRegistry()
    ai = _PromptAI(responses)
    registry.register("model", ai)
    registry.register("verify-model", ai)
    prompts = PromptService({
        "simple": PromptTemplate("simple", "Fallback", "fallback", version=1),
        "primary": PromptTemplate("primary", "Primary", "primary-v1", version=1),
        "verification": PromptTemplate(
            "verification", "Verification", f"verification-v{verification_version}",
            version=verification_version,
        ),
    })
    return AnalysisService(_CallLog(), registry, prompts, cache=cache), ai


def _run(executor, mode, entries, primary, verification=None):  # noqa: ANN001, ANN201
    return BatchAnalysisOrchestrator(executor).run(
        entries,
        TenantConfig("tenant", "https://api"),
        primary,
        verification_mode=mode,
        verification_spec=verification,
    )


def _decision_view(result):  # noqa: ANN001, ANN201
    return (
        [
            (
                item.entry.unique_id,
                item.primary_decision_status,
                item.verification_decision_status,
                item.final_decision,
                item.final_reason,
                item.final_status,
                item.verification_status,
            )
            for item in result.items
        ],
        astuple(result)[1:],
    )


@pytest.mark.parametrize("mode", ["off", "shadow", "enforce"])
def test_real_sequential_and_vertex_executors_have_ordered_decision_parity(mode) -> None:
    entries = [CallLogEntry(unique_id="yes"), CallLogEntry(unique_id="no")]
    responses = {
        ("primary-v1", "yes"): PRIMARY_TRUE,
        ("primary-v1", "no"): PRIMARY_FALSE,
        ("verification-v1", "yes"): VERIFY_FALSE,
    }
    sequential_service, _ai = _service(responses, {})
    vertex_service, _unused = _service(responses, {})
    _VertexRunner.responses = responses
    _VertexRunner.calls = []
    primary = _spec("primary")
    verification = _spec("verification") if mode != "off" else None

    sequential = _run(
        SequentialBatchExecutor(sequential_service), mode, entries, primary, verification,
    )
    vertex = _run(
        VertexBatchExecutor(vertex_service, runner_factory=_VertexRunner),
        mode,
        entries,
        primary,
        verification,
    )

    assert _decision_view(sequential) == _decision_view(vertex)
    assert [item.entry.unique_id for item in sequential.items] == ["yes", "no"]


@pytest.mark.parametrize("executor_kind", ["sequential", "vertex"])
@pytest.mark.parametrize(
    ("changed_model", "changed_version", "expected_final"),
    [("verify-model", 1, True), ("model", 2, False)],
)
def test_verification_model_or_prompt_version_change_misses_only_verification_cache(
    executor_kind, changed_model, changed_version, expected_final,
) -> None:
    cache = {}
    entries = [CallLogEntry(unique_id="yes")]
    responses = {
        ("primary-v1", "yes"): PRIMARY_TRUE,
        ("verification-v1", "yes"): VERIFY_TRUE,
        ("verification-v2", "yes"): VERIFY_FALSE,
    }
    service, ai = _service(responses, cache, verification_version=1)
    if executor_kind == "sequential":
        executor = SequentialBatchExecutor(service)
    else:
        _VertexRunner.responses = responses
        _VertexRunner.calls = []
        executor = VertexBatchExecutor(service, runner_factory=_VertexRunner)

    first = _run(executor, "enforce", entries, _spec("primary"), _spec("verification"))
    service._prompt_service = _service(  # noqa: SLF001
        responses, cache, verification_version=changed_version,
    )[0]._prompt_service  # noqa: SLF001
    second = _run(
        executor,
        "enforce",
        entries,
        _spec("primary"),
        _spec("verification", model=changed_model, version=changed_version),
    )

    assert first.items[0].primary.from_cache is False
    assert second.items[0].primary.from_cache is True
    assert second.items[0].verification.from_cache is False
    assert second.items[0].final_decision is expected_final
    if executor_kind == "sequential":
        assert ai.calls == [
            ("yes", "primary-v1"),
            ("yes", "verification-v1"),
            ("yes", f"verification-v{changed_version}"),
        ]
    else:
        assert _VertexRunner.calls == [
            ("model", "primary-v1", ["yes"]),
            ("model", "verification-v1", ["yes"]),
            (
                changed_model,
                f"verification-v{changed_version}",
                ["yes"],
            ),
        ]


@pytest.mark.parametrize(
    ("mode", "cached", "fresh", "expected", "status"),
    [
        ("off", LEGACY_TRUE, PRIMARY_FALSE, True, "disabled"),
        ("shadow", PRIMARY_TRUE, VERIFY_FALSE, True, "shadow_complete"),
        ("enforce", PRIMARY_TRUE, VERIFY_FALSE, False, "complete"),
    ],
)
def test_saved_results_preserve_off_shadow_and_enforce_semantics(
    mode, cached, fresh, expected, status,
) -> None:
    cache = {}
    responses = {
        ("primary-v1", "saved"): fresh,
        ("verification-v1", "saved"): VERIFY_FALSE,
    }
    service, ai = _service(responses, cache)
    primary = _spec("primary")
    primary_key = ("tenant", "saved", "primary", 1, "fake", "model", "")
    cache[primary_key] = AnalysisResult(
        text=cached, model="resolved-model", provider="fake",
    )

    result = _run(
        SequentialBatchExecutor(service),
        mode,
        [CallLogEntry(unique_id="saved")],
        primary,
        _spec("verification") if mode != "off" else None,
    )

    item = result.items[0]
    assert item.final_decision is expected
    assert item.verification_status == status
    if mode == "off":
        assert ai.calls == []


def test_fresh_legacy_output_is_strictly_invalid_even_in_off_mode() -> None:
    responses = {("primary-v1", "fresh"): LEGACY_TRUE}
    service, ai = _service(responses, {})

    result = _run(
        SequentialBatchExecutor(service),
        "off",
        [CallLogEntry(unique_id="fresh")],
        _spec("primary"),
    )

    assert result.items[0].primary_decision_status == "invalid"
    assert result.items[0].final_status == "invalid"
    assert len(ai.calls) == 2
