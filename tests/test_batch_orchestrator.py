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
    RoundExecutionResult,
    RoundSpec,
)
from calls_analyser.services.tenant import TenantConfig


def _entry(unique_id: str) -> CallLogEntry:
    return CallLogEntry(unique_id=unique_id)


def _round_spec() -> RoundSpec:
    return RoundSpec(
        model_key="primary-model",
        prompt_key="follow-up",
        prompt_text="Return JSON.",
        prompt_version="v1",
        custom_fragment="",
        language="en",
        usage_mode="ui_mass",
        stage_name="primary",
        provider="google",
        model_identity="gemini-primary",
        cache_identity="follow-up:v1:google:gemini-primary",
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
