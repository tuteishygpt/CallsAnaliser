"""Execution adapters for batch analysis rounds."""
from __future__ import annotations

from collections import deque
from itertools import count
from typing import Mapping, Sequence

from calls_analyser.domain.models import AnalysisResult, CallLogEntry, Language
from calls_analyser.services.analysis import AnalysisOptions, AnalysisService, CacheKey
from calls_analyser.services.batch_orchestrator import (
    ExecutorProgress,
    RoundExecutionResult,
    RoundSpec,
    ValidationResults,
)
from calls_analyser.services.tenant import TenantConfig


class SequentialBatchExecutor:
    """Run a batch round item-by-item through ``AnalysisService``."""

    def __init__(self, analysis_service: AnalysisService) -> None:
        self._analysis_service = analysis_service
        self._pending_runs: dict[
            int,
            deque[
                tuple[
                    RoundSpec,
                    int,
                    dict[str, AnalysisResult],
                    dict[str, CacheKey],
                ]
            ],
        ] = {}
        self._execution_ids = count(1)

    def execute(
        self,
        entries: Sequence[CallLogEntry],
        tenant: TenantConfig,
        round_spec: RoundSpec,
        *,
        bypass_cache: bool = False,
        progress: ExecutorProgress | None = None,
    ) -> dict[str, RoundExecutionResult]:
        results: dict[str, RoundExecutionResult] = {}
        saved_results: dict[str, AnalysisResult] = {}
        saved_cache_keys: dict[str, CacheKey] = {}
        execution_id = next(self._execution_ids)
        total = len(entries)
        language = Language(round_spec.language)
        for completed, entry in enumerate(entries, start=1):
            try:
                analysis, from_cache, cache_key = (
                    self._analysis_service.analyze_call_with_status(
                        entry.unique_id,
                        tenant,
                        language,
                        AnalysisOptions(
                            model_key=round_spec.model_key,
                            prompt_key=round_spec.prompt_key,
                            custom_prompt=round_spec.custom_fragment or None,
                            mode=round_spec.usage_mode,
                            call_entry=entry,
                            bypass_cache=bypass_cache,
                            batch_stage=round_spec.stage_name,
                            batch_execution="ui_sequential",
                        ),
                    )
                )
                saved_results[entry.unique_id] = analysis
                saved_cache_keys[entry.unique_id] = cache_key
                result = RoundExecutionResult(
                    raw_text=analysis.text,
                    provider=analysis.provider,
                    model=analysis.model,
                    from_cache=from_cache,
                    usage_metadata=analysis.metadata.get("usage_metadata"),
                    cache_key=cache_key,
                    cache_identity=round_spec.cache_identity,
                    execution_id=execution_id,
                )
            except Exception as exc:  # noqa: BLE001 - errors are per batch item
                result = RoundExecutionResult(
                    provider=round_spec.provider,
                    model=round_spec.model_identity,
                    execution_status="error",
                    execution_error=str(exc),
                    cache_identity=round_spec.cache_identity,
                    execution_id=execution_id,
                )
            results[entry.unique_id] = result
            if progress is not None:
                progress(entry.unique_id, result, completed, total)
        self._pending_runs.setdefault(id(round_spec), deque()).append(
            (round_spec, execution_id, saved_results, saved_cache_keys),
        )
        return results

    def record_validation(
        self,
        round_spec: RoundSpec,
        validated_results: Mapping[str, bool],
    ) -> None:
        pending = self._pending_runs.get(id(round_spec))
        if not pending:
            return
        requested_execution_id = (
            validated_results.execution_id
            if isinstance(validated_results, ValidationResults)
            else None
        )
        pending_index = 0
        if requested_execution_id is not None:
            matching_index = next(
                (
                    index
                    for index, run in enumerate(pending)
                    if run[1] == requested_execution_id
                ),
                None,
            )
            if matching_index is None:
                return
            pending_index = matching_index
        _retained_spec, _execution_id, saved_results, cache_keys = pending[pending_index]
        del pending[pending_index]
        if not pending:
            del self._pending_runs[id(round_spec)]
        for unique_id, decision_valid in validated_results.items():
            result = saved_results.get(unique_id)
            if result is not None:
                result.metadata["batch_stage"] = round_spec.stage_name
                result.metadata["batch_execution"] = "ui_sequential"
                result.metadata["decision_valid"] = decision_valid
                self._analysis_service.persist_cached_result(
                    cache_keys[unique_id], result,
                )
