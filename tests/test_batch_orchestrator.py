from __future__ import annotations

from dataclasses import replace
from typing import get_type_hints

import pytest

from calls_analyser.domain.models import CallLogEntry
from calls_analyser.services.analysis import CacheKey
from calls_analyser.services.batch_orchestrator import (
    BatchAnalysisOrchestrator,
    BatchExecutorContractError,
    BatchItemResult,
    BatchProgressEvent,
    DECISION_STATUSES,
    EXECUTION_STATUSES,
    FINAL_STATUSES,
    RoundExecutionResult,
    RoundSpec,
    VERIFICATION_STATUSES,
)
from calls_analyser.services.tenant import TenantConfig


def _entry(unique_id: str) -> CallLogEntry:
    return CallLogEntry(unique_id=unique_id)


def _round_spec() -> RoundSpec:
    return RoundSpec(
        model_key="primary-model",
        prompt_key="follow-up",
        prompt_text="Return JSON.",
        prompt_version=1,
        custom_fragment="",
        language="en",
        usage_mode="ui_mass",
        stage_name="primary",
        provider="google",
        model_identity="gemini-primary",
        cache_identity="follow-up:1:google:gemini-primary",
    )


def _verification_spec() -> RoundSpec:
    return replace(
        _round_spec(),
        model_key="verification-model",
        prompt_key="follow-up-verification",
        prompt_text="Verify and return JSON.",
        stage_name="verification",
        model_identity="gemini-verifier",
        cache_identity="follow-up-verification:1:google:gemini-verifier",
    )


def _success(text: str = "ok") -> RoundExecutionResult:
    return RoundExecutionResult(
        raw_text=text,
        provider="google",
        model="gemini-primary",
        execution_status="success",
    )


class _Executor:
    def __init__(self, results: dict[str, RoundExecutionResult]) -> None:
        self.results = results
        self.execute_calls: list[tuple] = []
        self.validation_calls: list[tuple] = []

    def execute(
        self,
        entries,
        tenant,
        round_spec,
        *,
        bypass_cache=False,
        progress=None,
    ) -> dict[str, RoundExecutionResult]:
        self.execute_calls.append(
            (list(entries), tenant, round_spec, bypass_cache, progress),
        )
        return self.results

    def record_validation(self, round_spec, validated_results) -> None:
        self.validation_calls.append((round_spec, validated_results))


class _RoundExecutor(_Executor):
    def __init__(self, results_by_stage) -> None:
        super().__init__({})
        self.results_by_stage = results_by_stage

    def execute(
        self,
        entries,
        tenant,
        round_spec,
        *,
        bypass_cache=False,
        progress=None,
    ) -> dict[str, RoundExecutionResult]:
        self.execute_calls.append(
            (list(entries), tenant, round_spec, bypass_cache, progress),
        )
        return self.results_by_stage[round_spec.stage_name]


def _decision(value: bool, reason: str = "A reason") -> RoundExecutionResult:
    return _success(
        f'{{"needs_follow_up": {str(value).lower()}, "reason": "{reason}"}}',
    )


def _error() -> RoundExecutionResult:
    return replace(_success(), execution_status="error", execution_error="boom")


@pytest.fixture
def tenant() -> TenantConfig:
    return TenantConfig(tenant_id="tenant-a", vochi_base_url="https://example.test")


def test_duplicate_input_ids_are_rejected_before_executor_work(tenant) -> None:
    executor = _Executor({})
    orchestrator = BatchAnalysisOrchestrator(executor)

    with pytest.raises(ValueError, match="duplicate.*duplicate-id"):
        orchestrator.run(
            [_entry("duplicate-id"), _entry("duplicate-id")],
            tenant,
            _round_spec(),
        )

    assert executor.execute_calls == []
    assert executor.validation_calls == []


def test_batch_result_preserves_input_order_not_executor_mapping_order(tenant) -> None:
    executor = _Executor(
        {
            "third": _success("third result"),
            "first": _success("first result"),
            "second": _success("second result"),
        },
    )

    result = BatchAnalysisOrchestrator(executor).run(
        [_entry("first"), _entry("second"), _entry("third")],
        tenant,
        _round_spec(),
    )

    assert [item.entry.unique_id for item in result.items] == [
        "first",
        "second",
        "third",
    ]
    assert [item.primary.raw_text for item in result.items] == [
        "first result",
        "second result",
        "third result",
    ]


def test_executor_omissions_are_synthesized_as_missing(tenant) -> None:
    result = BatchAnalysisOrchestrator(_Executor({"first": _success()})).run(
        [_entry("first"), _entry("omitted")],
        tenant,
        _round_spec(),
    )

    omitted = result.items[1].primary
    assert omitted.execution_status == "missing"
    assert omitted.execution_error == "executor omitted requested result"
    assert omitted.provider == "google"
    assert omitted.model == "gemini-primary"


def test_unrequested_executor_ids_raise_contract_error(tenant) -> None:
    executor = _Executor(
        {
            "requested": _success(),
            "unrequested": _success(),
        },
    )

    with pytest.raises(BatchExecutorContractError, match="unrequested"):
        BatchAnalysisOrchestrator(executor).run(
            [_entry("requested")],
            tenant,
            _round_spec(),
        )


@pytest.mark.parametrize("status", ["cached", "failed", ""])
def test_round_execution_status_is_a_closed_set(status) -> None:
    with pytest.raises(ValueError, match="execution_status"):
        replace(_success(), execution_status=status)


def test_round_execution_result_retains_repository_cache_key() -> None:
    cache_key: CacheKey = (
        "tenant-a",
        "call-1",
        "follow-up",
        3,
        "google",
        "gemini-primary",
        "custom fragment",
    )

    result = replace(_success(), cache_key=cache_key)

    assert result.cache_key is cache_key
    assert get_type_hints(RoundExecutionResult)["cache_key"] == CacheKey | None


def test_round_spec_prompt_version_matches_cache_key_integer_shape() -> None:
    spec = _round_spec()

    assert spec.prompt_version == 1
    assert get_type_hints(RoundSpec)["prompt_version"] is int


def test_status_contracts_match_the_approved_closed_sets_exactly() -> None:
    assert EXECUTION_STATUSES == {"success", "error", "missing"}
    assert DECISION_STATUSES == {"valid", "invalid", "unavailable"}
    assert FINAL_STATUSES == {"pending", "complete", "fallback", "error", "invalid"}
    assert VERIFICATION_STATUSES == {
        "not_requested",
        "disabled",
        "pending",
        "shadow_complete",
        "complete",
        "failed",
        "config_error",
    }


@pytest.mark.parametrize(
    ("field", "invalid_status"),
    [
        ("primary_decision_status", "unknown"),
        ("verification_decision_status", "unknown"),
        ("final_status", "done"),
        ("verification_status", "skipped"),
    ],
)
def test_item_statuses_are_closed_sets(field, invalid_status) -> None:
    item = BatchItemResult(entry=_entry("one"), primary=_success())

    with pytest.raises(ValueError, match=field):
        replace(item, **{field: invalid_status})


@pytest.mark.parametrize(
    (
        "case",
        "mode",
        "primary_result",
        "verification_result",
        "config_error",
        "expected",
        "expected_reason",
    ),
    [
        ("false-off", "off", _decision(False, "resolved"), None, None,
         (False, "complete", "not_requested", "valid", "unavailable"), "resolved"),
        ("false-shadow", "shadow", _decision(False, "resolved"), None, None,
         (False, "complete", "not_requested", "valid", "unavailable"), "resolved"),
        ("false-enforce", "enforce", _decision(False, "resolved"), None, None,
         (False, "complete", "not_requested", "valid", "unavailable"), "resolved"),
        ("true-off", "off", _decision(True, "call"), None, None,
         (True, "complete", "disabled", "valid", "unavailable"), "call"),
        ("shadow-true", "shadow", _decision(True, "call"), _decision(True, "yes"), None,
         (True, "complete", "shadow_complete", "valid", "valid"), "call"),
        ("shadow-false", "shadow", _decision(True, "call"), _decision(False, "no"), None,
         (True, "complete", "shadow_complete", "valid", "valid"), "call"),
        ("enforce-true", "enforce", _decision(True, "call"), _decision(True, "yes"), None,
         (True, "complete", "complete", "valid", "valid"), "yes"),
        ("enforce-false", "enforce", _decision(True, "call"), _decision(False, "no"), None,
         (False, "complete", "complete", "valid", "valid"), "no"),
        ("shadow-error", "shadow", _decision(True, "call"), _error(), None,
         (True, "fallback", "failed", "valid", "unavailable"), "call"),
        ("enforce-missing", "enforce", _decision(True, "call"), None, None,
         (True, "fallback", "failed", "valid", "unavailable"), "call"),
        ("shadow-invalid", "shadow", _decision(True, "call"), _success("not json"), None,
         (True, "fallback", "failed", "valid", "invalid"), "call"),
        ("shadow-config", "shadow", _decision(True, "call"), None, "missing verifier",
         (True, "fallback", "config_error", "valid", "unavailable"), "call"),
        ("enforce-config", "enforce", _decision(True, "call"), None, "missing verifier",
         (True, "fallback", "config_error", "valid", "unavailable"), "call"),
        ("primary-error", "enforce", _error(), None, None,
         (None, "error", "not_requested", "unavailable", "unavailable"), "boom"),
        ("primary-missing", "off", None, None, None,
         (None, "error", "not_requested", "unavailable", "unavailable"),
         "executor omitted requested result"),
        ("primary-invalid", "shadow", _success("not json"), None, None,
         (None, "invalid", "not_requested", "invalid", "unavailable"), None),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_approved_decision_matrix(
    tenant,
    case,
    mode,
    primary_result,
    verification_result,
    config_error,
    expected,
    expected_reason,
) -> None:
    del case
    primary_results = {} if primary_result is None else {"one": primary_result}
    by_stage = {"primary": primary_results}
    if verification_result is not None:
        by_stage["verification"] = {"one": verification_result}
    elif mode in {"shadow", "enforce"} and config_error is None:
        by_stage["verification"] = {}
    executor = _RoundExecutor(by_stage)

    result = BatchAnalysisOrchestrator(executor).run(
        [_entry("one")],
        tenant,
        _round_spec(),
        verification_mode=mode,
        verification_spec=(
            _verification_spec()
            if mode in {"shadow", "enforce"} and config_error is None
            else None
        ),
        verification_config_error=config_error,
    )

    item = result.items[0]
    assert (
        item.final_decision,
        item.final_status,
        item.verification_status,
        item.primary_decision_status,
        item.verification_decision_status,
    ) == expected
    assert item.final_reason == expected_reason
    if item.verification_decision_status == "valid":
        assert item.verification_decision is not None


def test_only_valid_primary_positives_are_sent_to_verification_as_original_entries(tenant) -> None:
    entries = [_entry("yes"), _entry("no"), _entry("invalid"), _entry("error")]
    executor = _RoundExecutor(
        {
            "primary": {
                "yes": _decision(True),
                "no": _decision(False),
                "invalid": _success("Needs follow-up: Yes\nSummary: old"),
                "error": _error(),
            },
            "verification": {"yes": _decision(False)},
        },
    )

    BatchAnalysisOrchestrator(executor).run(
        entries,
        tenant,
        _round_spec(),
        verification_mode="enforce",
        verification_spec=_verification_spec(),
    )

    verification_entries = executor.execute_calls[1][0]
    assert verification_entries == [entries[0]]
    assert verification_entries[0] is entries[0]
    assert not hasattr(executor.execute_calls[1][2], "primary_raw_text")
    assert not hasattr(executor.execute_calls[1][2], "primary_decision_text")


def test_parsing_compatibility_is_limited_to_cached_primary_in_off_mode(tenant) -> None:
    legacy = "Needs follow-up: Yes\nSummary: Call back"

    def run(mode: str, from_cache: bool):
        executor = _RoundExecutor(
            {"primary": {"one": replace(_success(legacy), from_cache=from_cache)}},
        )
        return BatchAnalysisOrchestrator(executor).run(
            [_entry("one")], tenant, _round_spec(), verification_mode=mode,
            verification_config_error=("not configured" if mode != "off" else None),
        ).items[0]

    assert run("off", True).primary_decision_status == "valid"
    assert run("off", False).primary_decision_status == "invalid"
    assert run("shadow", True).primary_decision_status == "invalid"
    assert run("enforce", True).primary_decision_status == "invalid"


def test_verification_always_uses_strict_parser_even_for_cached_results(tenant) -> None:
    executor = _RoundExecutor(
        {
            "primary": {"one": _decision(True)},
            "verification": {
                "one": replace(
                    _success("Needs follow-up: No\nSummary: old"),
                    from_cache=True,
                ),
            },
        },
    )

    item = BatchAnalysisOrchestrator(executor).run(
        [_entry("one")], tenant, _round_spec(), verification_mode="enforce",
        verification_spec=_verification_spec(),
    ).items[0]

    assert item.verification_decision_status == "invalid"
    assert item.final_decision is True
    assert item.final_status == "fallback"


def test_rejects_unsupported_verification_mode_and_missing_active_configuration(tenant) -> None:
    orchestrator = BatchAnalysisOrchestrator(_Executor({}))

    with pytest.raises(ValueError, match="verification_mode.*off.*shadow.*enforce"):
        orchestrator.run([], tenant, _round_spec(), verification_mode="audit")
    with pytest.raises(ValueError, match="verification_spec.*configuration error"):
        orchestrator.run(
            [], tenant, _round_spec(), verification_mode="shadow",
        )


def test_mixed_batch_counters_are_derived_from_finalized_items(tenant) -> None:
    entries = [_entry(name) for name in (
        "primary-no", "kept", "changed", "verify-failed", "primary-invalid",
    )]
    executor = _RoundExecutor(
        {
            "primary": {
                "primary-no": _decision(False),
                "kept": _decision(True),
                "changed": _decision(True),
                "verify-failed": _decision(True),
                "primary-invalid": _success("bad"),
            },
            "verification": {
                "kept": _decision(True),
                "changed": _decision(False),
                "verify-failed": _error(),
            },
        },
    )

    result = BatchAnalysisOrchestrator(executor).run(
        entries, tenant, _round_spec(), verification_mode="enforce",
        verification_spec=_verification_spec(),
    )

    assert (
        result.total,
        result.round_1_success,
        result.verification_requested,
        result.verification_success,
        result.verification_changed_to_false,
        result.verification_failed,
        result.final_follow_up,
    ) == (5, 4, 3, 2, 1, 1, 2)


def test_config_errors_count_as_requested_and_failed_without_round_two(tenant) -> None:
    executor = _RoundExecutor(
        {"primary": {"yes": _decision(True), "no": _decision(False)}},
    )

    result = BatchAnalysisOrchestrator(executor).run(
        [_entry("yes"), _entry("no")], tenant, _round_spec(),
        verification_mode="shadow", verification_config_error="bad prompt",
    )

    assert len(executor.execute_calls) == 1
    assert result.verification_requested == 1
    assert result.verification_failed == 1
    assert result.final_follow_up == 1
    assert result.items[0].verification is not None
    assert result.items[0].verification.execution_status == "error"
    assert result.items[0].verification.execution_error == "bad prompt"


def test_progress_events_are_deterministic_and_primary_items_remain_pending(tenant) -> None:
    events: list[BatchProgressEvent] = []
    entries = [_entry("yes"), _entry("no")]
    executor = _RoundExecutor(
        {
            "primary": {"yes": _decision(True), "no": _decision(False)},
            "verification": {"yes": _decision(False)},
        },
    )

    BatchAnalysisOrchestrator(executor).run(
        entries, tenant, _round_spec(), verification_mode="enforce",
        verification_spec=_verification_spec(), progress=events.append,
    )

    assert [event.event for event in events] == [
        "primary_started",
        "primary_complete",
        "primary_complete",
        "verification_started",
        "verification_complete",
        "run_complete",
    ]
    assert [(event.completed, event.total) for event in events] == [
        (0, 2), (1, 2), (2, 2), (0, 1), (1, 1), (2, 2),
    ]
    primary_events = [event for event in events if event.event == "primary_complete"]
    assert [event.unique_id for event in primary_events] == ["yes", "no"]
    assert all(event.item.final_decision is None for event in primary_events)
    assert all(event.item.final_status == "pending" for event in primary_events)
    assert primary_events[0].item.verification_status == "pending"
    assert primary_events[1].item.verification_status == "not_requested"
    assert callable(executor.execute_calls[0][4])
    assert callable(executor.execute_calls[1][4])


def test_executor_item_progress_is_emitted_before_execute_returns(tenant) -> None:
    events: list[BatchProgressEvent] = []

    class LiveExecutor(_Executor):
        def execute(
            self,
            entries,
            tenant,
            round_spec,
            *,
            bypass_cache=False,
            progress=None,
        ) -> dict[str, RoundExecutionResult]:
            del tenant, bypass_cache
            result = (
                _decision(True, "primary")
                if round_spec.stage_name == "primary"
                else _decision(False, "verified")
            )
            assert progress is not None
            progress(entries[0].unique_id, result, 1, 1)
            expected_event = (
                "primary_complete"
                if round_spec.stage_name == "primary"
                else "verification_complete"
            )
            assert events[-1].event == expected_event
            return {entries[0].unique_id: result}

    result = BatchAnalysisOrchestrator(LiveExecutor({})).run(
        [_entry("one")], tenant, _round_spec(), verification_mode="enforce",
        verification_spec=_verification_spec(), progress=events.append,
    )

    assert [event.event for event in events] == [
        "primary_started",
        "primary_complete",
        "verification_started",
        "verification_complete",
        "run_complete",
    ]
    assert result.items[0].final_decision is False


def test_progress_callback_exceptions_are_logged_and_ignored(tenant, caplog) -> None:
    def broken_callback(event: BatchProgressEvent) -> None:
        raise RuntimeError(f"cannot handle {event.event}")

    result = BatchAnalysisOrchestrator(
        _RoundExecutor({"primary": {"one": _decision(False)}}),
    ).run(
        [_entry("one")], tenant, _round_spec(), progress=broken_callback,
    )

    assert result.items[0].final_decision is False
    assert "progress callback failed" in caplog.text
