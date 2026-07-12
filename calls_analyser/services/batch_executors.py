"""Execution adapters for batch analysis rounds."""
from __future__ import annotations

from typing import Mapping, Sequence

from calls_analyser.domain.models import AnalysisResult, CallLogEntry, Language
from calls_analyser.services.analysis import AnalysisOptions, AnalysisService
from calls_analyser.services.batch_orchestrator import (
    ExecutorProgress,
    RoundExecutionResult,
    RoundSpec,
)
from calls_analyser.services.tenant import TenantConfig


class SequentialBatchExecutor:
    """Run a batch round item-by-item through ``AnalysisService``."""

    def __init__(self, analysis_service: AnalysisService) -> None:
        self._analysis_service = analysis_service
        self._latest_results: dict[str, dict[str, AnalysisResult]] = {}

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
                result = RoundExecutionResult(
                    raw_text=analysis.text,
                    provider=analysis.provider,
                    model=analysis.model,
                    from_cache=from_cache,
                    usage_metadata=analysis.metadata.get("usage_metadata"),
                    cache_key=cache_key,
                    cache_identity=round_spec.cache_identity,
                )
            except Exception as exc:  # noqa: BLE001 - errors are per batch item
                result = RoundExecutionResult(
                    provider=round_spec.provider,
                    model=round_spec.model_identity,
                    execution_status="error",
                    execution_error=str(exc),
                    cache_identity=round_spec.cache_identity,
                )
            results[entry.unique_id] = result
            if progress is not None:
                progress(entry.unique_id, result, completed, total)
        self._latest_results[round_spec.stage_name] = saved_results
        return results

    def record_validation(
        self,
        round_spec: RoundSpec,
        validated_results: Mapping[str, bool],
    ) -> None:
        saved_results = self._latest_results.get(round_spec.stage_name, {})
        for unique_id, decision_valid in validated_results.items():
            result = saved_results.get(unique_id)
            if result is not None:
                result.metadata["batch_stage"] = round_spec.stage_name
                result.metadata["batch_execution"] = "ui_sequential"
                result.metadata["decision_valid"] = decision_valid
