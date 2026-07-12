"""Shared contracts for batch analysis orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from calls_analyser.domain.models import CallLogEntry
from calls_analyser.services.analysis import CacheKey
from calls_analyser.services.follow_up import FollowUpDecision
from calls_analyser.services.tenant import TenantConfig


EXECUTION_STATUSES = frozenset({"success", "error", "missing"})
DECISION_STATUSES = frozenset({"valid", "invalid", "unavailable"})
FINAL_STATUSES = frozenset({"pending", "complete", "fallback", "error", "invalid"})
VERIFICATION_STATUSES = frozenset(
    {
        "not_requested",
        "disabled",
        "pending",
        "shadow_complete",
        "complete",
        "failed",
        "config_error",
    },
)


def _validate_status(field_name: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {choices}; got {value!r}")


@dataclass(frozen=True, slots=True)
class RoundSpec:
    model_key: str
    prompt_key: str
    prompt_text: str
    prompt_version: int
    custom_fragment: str
    language: str
    usage_mode: str
    stage_name: str
    provider: str
    model_identity: str
    cache_identity: str


@dataclass(frozen=True, slots=True)
class RoundExecutionResult:
    raw_text: str = ""
    provider: str = ""
    model: str = ""
    execution_status: str = "success"
    from_cache: bool = False
    usage_metadata: Mapping[str, Any] | None = None
    execution_error: str | None = None
    cache_key: CacheKey | None = None
    cache_identity: Mapping[str, str] | str | None = None

    def __post_init__(self) -> None:
        _validate_status("execution_status", self.execution_status, EXECUTION_STATUSES)


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    entry: CallLogEntry
    primary: RoundExecutionResult
    primary_decision: FollowUpDecision | None = None
    primary_decision_status: str = "unavailable"
    verification: RoundExecutionResult | None = None
    verification_decision: FollowUpDecision | None = None
    verification_decision_status: str = "unavailable"
    final_decision: bool | None = None
    final_reason: str | None = None
    final_status: str = "pending"
    verification_status: str = "not_requested"

    def __post_init__(self) -> None:
        _validate_status(
            "primary_decision_status",
            self.primary_decision_status,
            DECISION_STATUSES,
        )
        _validate_status(
            "verification_decision_status",
            self.verification_decision_status,
            DECISION_STATUSES,
        )
        _validate_status("final_status", self.final_status, FINAL_STATUSES)
        _validate_status(
            "verification_status",
            self.verification_status,
            VERIFICATION_STATUSES,
        )

    @property
    def from_cache(self) -> bool:
        """Whether either execution round came from cache."""
        return self.primary.from_cache or bool(
            self.verification is not None and self.verification.from_cache,
        )


@dataclass(frozen=True, slots=True)
class BatchRunResult:
    items: tuple[BatchItemResult, ...]
    total: int
    round_1_success: int = 0
    verification_requested: int = 0
    verification_success: int = 0
    verification_changed_to_false: int = 0
    verification_failed: int = 0
    final_follow_up: int = 0


@dataclass(frozen=True, slots=True)
class BatchProgressEvent:
    event: str
    stage_name: str
    completed: int
    total: int
    unique_id: str | None = None
    item: BatchItemResult | None = None


ExecutorProgress = Callable[[str, int, int], None]


class BatchRoundExecutor(Protocol):
    def execute(
        self,
        entries: Sequence[CallLogEntry],
        tenant: TenantConfig,
        round_spec: RoundSpec,
        *,
        bypass_cache: bool = False,
        progress: ExecutorProgress | None = None,
    ) -> dict[str, RoundExecutionResult]: ...

    def record_validation(
        self,
        round_spec: RoundSpec,
        validated_results: Mapping[str, bool],
    ) -> None: ...


class BatchExecutorContractError(RuntimeError):
    """Raised when an executor violates the round-result mapping contract."""


class BatchAnalysisOrchestrator:
    def __init__(self, executor: BatchRoundExecutor) -> None:
        self._executor = executor

    def run(
        self,
        entries: Sequence[CallLogEntry],
        tenant: TenantConfig,
        round_spec: RoundSpec,
        *,
        progress: ExecutorProgress | None = None,
    ) -> BatchRunResult:
        ordered_entries = tuple(entries)
        requested_ids: set[str] = set()
        for entry in ordered_entries:
            if entry.unique_id in requested_ids:
                raise ValueError(
                    f"duplicate input unique_id: {entry.unique_id!r}",
                )
            requested_ids.add(entry.unique_id)

        execution_results = self._executor.execute(
            ordered_entries,
            tenant,
            round_spec,
            progress=progress,
        )
        extra_ids = set(execution_results).difference(requested_ids)
        if extra_ids:
            extras = ", ".join(sorted(extra_ids))
            raise BatchExecutorContractError(
                f"executor returned unrequested result IDs: {extras}",
            )

        items = tuple(
            BatchItemResult(
                entry=entry,
                primary=execution_results.get(entry.unique_id)
                or self._missing_result(round_spec),
            )
            for entry in ordered_entries
        )
        return BatchRunResult(items=items, total=len(items))

    @staticmethod
    def _missing_result(round_spec: RoundSpec) -> RoundExecutionResult:
        return RoundExecutionResult(
            provider=round_spec.provider,
            model=round_spec.model_identity,
            execution_status="missing",
            execution_error="executor omitted requested result",
            cache_identity=round_spec.cache_identity,
        )
