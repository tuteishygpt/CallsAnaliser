"""Shared contracts for batch analysis orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from calls_analyser.domain.models import CallLogEntry
from calls_analyser.services.analysis import CacheKey
from calls_analyser.services.follow_up import FollowUpDecision, FollowUpDecisionParser
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
VERIFICATION_MODES = frozenset({"off", "shadow", "enforce"})

LOGGER = logging.getLogger(__name__)


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


ExecutorProgress = Callable[[str, RoundExecutionResult, int, int], None]
BatchProgressCallback = Callable[[BatchProgressEvent], None]


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
        verification_mode: str = "off",
        verification_spec: RoundSpec | None = None,
        verification_config_error: str | None = None,
        progress: BatchProgressCallback | None = None,
    ) -> BatchRunResult:
        self._validate_verification_configuration(
            verification_mode,
            verification_spec,
            verification_config_error,
        )
        ordered_entries = tuple(entries)
        requested_ids: set[str] = set()
        for entry in ordered_entries:
            if entry.unique_id in requested_ids:
                raise ValueError(
                    f"duplicate input unique_id: {entry.unique_id!r}",
                )
            requested_ids.add(entry.unique_id)

        self._emit_progress(
            progress,
            BatchProgressEvent(
                event="primary_started",
                stage_name=round_spec.stage_name,
                completed=0,
                total=len(ordered_entries),
            ),
        )
        entries_by_id = {entry.unique_id: entry for entry in ordered_entries}
        reported_primary_ids: set[str] = set()
        primary_completed = 0

        def emit_primary_complete(
            unique_id: str,
            execution: RoundExecutionResult,
            _executor_completed: int,
            _executor_total: int,
        ) -> None:
            nonlocal primary_completed
            entry = entries_by_id.get(unique_id)
            if entry is None:
                raise BatchExecutorContractError(
                    f"executor reported progress for unrequested result ID: {unique_id}",
                )
            if unique_id in reported_primary_ids:
                raise BatchExecutorContractError(
                    f"executor reported duplicate progress for result ID: {unique_id}",
                )
            reported_primary_ids.add(unique_id)
            primary_completed += 1
            item = self._build_primary_item(entry, execution, verification_mode)
            self._emit_progress(
                progress,
                BatchProgressEvent(
                    event="primary_complete",
                    stage_name=round_spec.stage_name,
                    completed=primary_completed,
                    total=len(ordered_entries),
                    unique_id=unique_id,
                    item=item,
                ),
            )

        execution_results = self._execute_round(
            ordered_entries,
            tenant,
            round_spec,
            progress=emit_primary_complete,
        )

        primary_items: dict[str, BatchItemResult] = {}
        candidates: list[CallLogEntry] = []
        for entry in ordered_entries:
            primary = execution_results.get(entry.unique_id) or self._missing_result(
                round_spec,
            )
            item = self._build_primary_item(entry, primary, verification_mode)
            primary_items[entry.unique_id] = item
            if (
                verification_mode in {"shadow", "enforce"}
                and item.primary_decision is not None
                and item.primary_decision.needs_follow_up
            ):
                candidates.append(entry)
            if entry.unique_id not in reported_primary_ids:
                emit_primary_complete(
                    entry.unique_id,
                    primary,
                    0,
                    0,
                )

        verification_results: Mapping[str, RoundExecutionResult] = {}
        verification_completed = 0
        if candidates:
            verification_stage = (
                verification_spec.stage_name
                if verification_spec is not None
                else "verification"
            )
            self._emit_progress(
                progress,
                BatchProgressEvent(
                    event="verification_started",
                    stage_name=verification_stage,
                    completed=0,
                    total=len(candidates),
                ),
            )
            if verification_spec is not None:
                candidates_by_id = {
                    entry.unique_id: entry for entry in candidates
                }
                reported_verification_ids: set[str] = set()

                def emit_verification_complete(
                    unique_id: str,
                    execution: RoundExecutionResult,
                    _executor_completed: int,
                    _executor_total: int,
                ) -> None:
                    nonlocal verification_completed
                    if unique_id not in candidates_by_id:
                        raise BatchExecutorContractError(
                            "executor reported verification progress for "
                            f"unrequested result ID: {unique_id}",
                        )
                    if unique_id in reported_verification_ids:
                        raise BatchExecutorContractError(
                            "executor reported duplicate verification progress "
                            f"for result ID: {unique_id}",
                        )
                    reported_verification_ids.add(unique_id)
                    verification_completed += 1
                    progress_item = self._apply_verification(
                        primary_items[unique_id],
                        execution,
                        verification_mode,
                    )
                    self._emit_progress(
                        progress,
                        BatchProgressEvent(
                            event="verification_complete",
                            stage_name=verification_spec.stage_name,
                            completed=verification_completed,
                            total=len(candidates),
                            unique_id=unique_id,
                            item=progress_item,
                        ),
                    )

                verification_results = self._execute_round(
                    tuple(candidates),
                    tenant,
                    verification_spec,
                    progress=emit_verification_complete,
                )
            else:
                reported_verification_ids = set()

        finalized: dict[str, BatchItemResult] = {}
        candidate_ids = {entry.unique_id for entry in candidates}
        for entry in ordered_entries:
            item = primary_items[entry.unique_id]
            if entry.unique_id in candidate_ids:
                if verification_config_error is not None:
                    item = self._apply_verification_config_error(
                        item,
                        verification_config_error,
                    )
                else:
                    assert verification_spec is not None
                    verification = verification_results.get(entry.unique_id)
                    if verification is None:
                        verification = self._missing_result(verification_spec)
                    item = self._apply_verification(
                        item,
                        verification,
                        verification_mode,
                    )
                if entry.unique_id not in reported_verification_ids:
                    verification_completed += 1
                    self._emit_progress(
                        progress,
                        BatchProgressEvent(
                            event="verification_complete",
                            stage_name=(
                                verification_spec.stage_name
                                if verification_spec is not None
                                else "verification"
                            ),
                            completed=verification_completed,
                            total=len(candidates),
                            unique_id=entry.unique_id,
                            item=item,
                        ),
                    )
            else:
                item = self._finalize_without_verification(item, verification_mode)
            finalized[entry.unique_id] = item

        items = tuple(finalized[entry.unique_id] for entry in ordered_entries)
        result = self._build_result(items, verification_mode)
        self._emit_progress(
            progress,
            BatchProgressEvent(
                event="run_complete",
                stage_name="run",
                completed=len(items),
                total=len(items),
            ),
        )
        return result

    def _execute_round(
        self,
        entries: Sequence[CallLogEntry],
        tenant: TenantConfig,
        round_spec: RoundSpec,
        *,
        progress: ExecutorProgress | None,
    ) -> Mapping[str, RoundExecutionResult]:
        execution_results = self._executor.execute(
            entries,
            tenant,
            round_spec,
            progress=progress,
        )
        requested_ids = {entry.unique_id for entry in entries}
        extra_ids = set(execution_results).difference(requested_ids)
        if extra_ids:
            extras = ", ".join(sorted(extra_ids))
            raise BatchExecutorContractError(
                f"executor returned unrequested result IDs: {extras}",
            )
        return execution_results

    @staticmethod
    def _validate_verification_configuration(
        verification_mode: str,
        verification_spec: RoundSpec | None,
        verification_config_error: str | None,
    ) -> None:
        if verification_mode not in VERIFICATION_MODES:
            raise ValueError(
                "verification_mode must be one of: off, shadow, enforce; "
                f"got {verification_mode!r}",
            )
        if verification_spec is not None and verification_config_error is not None:
            raise ValueError(
                "provide verification_spec or a verification configuration error, not both",
            )
        if (
            verification_mode in {"shadow", "enforce"}
            and verification_spec is None
            and verification_config_error is None
        ):
            raise ValueError(
                "active verification requires verification_spec or a configuration error marker",
            )

    @staticmethod
    def _parse_primary(
        result: RoundExecutionResult,
        verification_mode: str,
    ) -> tuple[FollowUpDecision | None, str]:
        if result.execution_status != "success":
            return None, "unavailable"
        if verification_mode == "off" and result.from_cache:
            decision = FollowUpDecisionParser.parse_compatibility(result.raw_text)
        else:
            decision = FollowUpDecisionParser.parse_strict(result.raw_text)
        return (decision, "valid") if decision is not None else (None, "invalid")

    @classmethod
    def _build_primary_item(
        cls,
        entry: CallLogEntry,
        primary: RoundExecutionResult,
        verification_mode: str,
    ) -> BatchItemResult:
        decision, decision_status = cls._parse_primary(primary, verification_mode)
        return BatchItemResult(
            entry=entry,
            primary=primary,
            primary_decision=decision,
            primary_decision_status=decision_status,
            verification_status=(
                "pending"
                if verification_mode in {"shadow", "enforce"}
                and decision is not None
                and decision.needs_follow_up
                else "not_requested"
            ),
        )

    @staticmethod
    def _finalize_without_verification(
        item: BatchItemResult,
        verification_mode: str,
    ) -> BatchItemResult:
        if item.primary.execution_status != "success":
            return replace(
                item,
                final_reason=item.primary.execution_error,
                final_status="error",
            )
        if item.primary_decision is None:
            return replace(item, final_status="invalid")
        verification_status = (
            "disabled"
            if verification_mode == "off" and item.primary_decision.needs_follow_up
            else "not_requested"
        )
        return replace(
            item,
            final_decision=item.primary_decision.needs_follow_up,
            final_reason=item.primary_decision.reason,
            final_status="complete",
            verification_status=verification_status,
        )

    @staticmethod
    def _apply_verification_config_error(
        item: BatchItemResult,
        error_message: str,
    ) -> BatchItemResult:
        assert item.primary_decision is not None
        return replace(
            item,
            verification=RoundExecutionResult(
                execution_status="error",
                execution_error=error_message,
            ),
            final_decision=True,
            final_reason=item.primary_decision.reason,
            final_status="fallback",
            verification_status="config_error",
        )

    @staticmethod
    def _apply_verification(
        item: BatchItemResult,
        verification: RoundExecutionResult,
        verification_mode: str,
    ) -> BatchItemResult:
        assert item.primary_decision is not None
        verification_decision = None
        verification_decision_status = "unavailable"
        if verification.execution_status == "success":
            verification_decision = FollowUpDecisionParser.parse_strict(
                verification.raw_text,
            )
            verification_decision_status = (
                "valid" if verification_decision is not None else "invalid"
            )
        if verification_decision is None:
            return replace(
                item,
                verification=verification,
                verification_decision_status=verification_decision_status,
                final_decision=True,
                final_reason=item.primary_decision.reason,
                final_status="fallback",
                verification_status="failed",
            )
        if verification_mode == "shadow":
            return replace(
                item,
                verification=verification,
                verification_decision=verification_decision,
                verification_decision_status="valid",
                final_decision=True,
                final_reason=item.primary_decision.reason,
                final_status="complete",
                verification_status="shadow_complete",
            )
        return replace(
            item,
            verification=verification,
            verification_decision=verification_decision,
            verification_decision_status="valid",
            final_decision=verification_decision.needs_follow_up,
            final_reason=verification_decision.reason,
            final_status="complete",
            verification_status="complete",
        )

    @staticmethod
    def _build_result(
        items: tuple[BatchItemResult, ...],
        verification_mode: str,
    ) -> BatchRunResult:
        return BatchRunResult(
            items=items,
            total=len(items),
            round_1_success=sum(
                item.primary_decision_status == "valid" for item in items
            ),
            verification_requested=sum(
                item.verification_status
                in {"shadow_complete", "complete", "failed", "config_error"}
                for item in items
            ),
            verification_success=sum(
                item.verification_status in {"shadow_complete", "complete"}
                for item in items
            ),
            verification_changed_to_false=sum(
                verification_mode == "enforce"
                and item.primary_decision is not None
                and item.primary_decision.needs_follow_up
                and item.verification_decision is not None
                and not item.verification_decision.needs_follow_up
                for item in items
            ),
            verification_failed=sum(
                item.verification_status in {"failed", "config_error"}
                for item in items
            ),
            final_follow_up=sum(item.final_decision is True for item in items),
        )

    @staticmethod
    def _emit_progress(
        callback: BatchProgressCallback | None,
        event: BatchProgressEvent,
    ) -> None:
        if callback is None:
            return
        try:
            callback(event)
        except Exception:  # noqa: BLE001 - client callbacks must not abort a batch
            LOGGER.exception("progress callback failed for %s", event.event)

    @staticmethod
    def _missing_result(round_spec: RoundSpec) -> RoundExecutionResult:
        return RoundExecutionResult(
            provider=round_spec.provider,
            model=round_spec.model_identity,
            execution_status="missing",
            execution_error="executor omitted requested result",
            cache_identity=round_spec.cache_identity,
        )
