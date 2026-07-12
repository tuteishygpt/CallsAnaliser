"""Execution adapters for batch analysis rounds."""
from __future__ import annotations

from collections import deque
from itertools import count
from threading import Lock
from typing import Any, Callable, Mapping, Sequence

from calls_analyser.domain.models import AnalysisResult, CallLogEntry, Language
from calls_analyser.services.analysis import AnalysisOptions, AnalysisService, CacheKey
from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
from calls_analyser.services.batch_orchestrator import (
    ExecutorProgress,
    RoundExecutionResult,
    RoundExecutionResults,
    RoundSpec,
    ValidationResults,
)
from calls_analyser.services.tenant import TenantConfig
from calls_analyser.services.gemini_batch import (
    BatchAnalysisResult,
    BatchTask,
    VertexBatchRunner,
    guess_mime_type,
)
from calls_analyser.services.usage import extract_usage_metadata


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


class VertexBatchExecutor:
    """Run one scheduler round through the Vertex batch API."""

    def __init__(
        self,
        analysis_service: AnalysisService,
        *,
        runner_factory: Callable[[str], Any] = VertexBatchRunner,
        batch_size_resolver: Callable[[TenantConfig], int] | None = None,
    ) -> None:
        self._analysis_service = analysis_service
        self._runner_factory = runner_factory
        self._batch_size_resolver = batch_size_resolver or (lambda _tenant: 25)
        self._pending_runs: dict[
            int,
            deque[tuple[RoundSpec, int, dict[str, AnalysisResult], dict[str, CacheKey]]],
        ] = {}
        self._execution_ids = count(1)
        self._pending_lock = Lock()

    def execute(
        self,
        entries: Sequence[CallLogEntry],
        tenant: TenantConfig,
        round_spec: RoundSpec,
        *,
        bypass_cache: bool = False,
        progress: ExecutorProgress | None = None,
    ) -> dict[str, RoundExecutionResult]:
        with self._pending_lock:
            execution_id = next(self._execution_ids)
        cache_keys = {
            entry.unique_id: self._cache_key(entry, tenant, round_spec)
            for entry in entries
        }
        cached = {} if bypass_cache else self._get_many(cache_keys.values())
        results = RoundExecutionResults(execution_id=execution_id)
        persisted: dict[str, AnalysisResult] = {}
        tasks: list[BatchTask] = []
        entries_by_id = {entry.unique_id: entry for entry in entries}

        for entry in entries:
            cache_key = cache_keys[entry.unique_id]
            cached_result = cached.get(cache_key)
            if cached_result is not None:
                persisted[entry.unique_id] = cached_result
                results[entry.unique_id] = self._round_result(
                    cached_result,
                    round_spec,
                    cache_key,
                    execution_id,
                    from_cache=True,
                )
                continue
            try:
                handle = self._analysis_service._call_log_service.ensure_recording(  # noqa: SLF001
                    entry.unique_id,
                    tenant,
                )
                tasks.append(BatchTask(
                    key=entry.unique_id,
                    path=handle.local_uri,
                    mime_type=guess_mime_type(handle.local_uri),
                ))
            except Exception as exc:  # noqa: BLE001 - preparation is per item
                results[entry.unique_id] = RoundExecutionResult(
                    provider=round_spec.provider,
                    model=round_spec.model_identity,
                    execution_status="error",
                    execution_error=str(exc),
                    cache_identity=round_spec.cache_identity,
                    execution_id=execution_id,
                )

        if tasks:
            instruction = GeminiAIAdapter._system_instruction(Language(round_spec.language))  # noqa: SLF001
            prompt = f"[SYSTEM INSTRUCTION: {instruction}]\n\n{round_spec.prompt_text}".strip()
            try:
                runner = self._runner_factory(round_spec.model_key)
                batch_results = runner.run_batch_results(
                    tasks,
                    prompt,
                    chunk_size=self._batch_size_resolver(tenant),
                )
            except Exception as exc:  # noqa: BLE001 - batch failure is terminal per item
                batch_results = {}
                for task in tasks:
                    results[task.key] = RoundExecutionResult(
                        provider=round_spec.provider,
                        model=round_spec.model_identity,
                        execution_status="error",
                        execution_error=str(exc),
                        cache_identity=round_spec.cache_identity,
                        execution_id=execution_id,
                    )

            for task in tasks:
                batch_result = batch_results.get(task.key)
                if batch_result is None:
                    continue
                text = (
                    batch_result.text
                    if isinstance(batch_result, BatchAnalysisResult)
                    else str(batch_result)
                )
                if text.startswith("Error:"):
                    results[task.key] = RoundExecutionResult(
                        provider=round_spec.provider,
                        model=round_spec.model_identity,
                        execution_status="error",
                        execution_error=text,
                        cache_identity=round_spec.cache_identity,
                        execution_id=execution_id,
                    )
                    continue
                usage_metadata = getattr(batch_result, "usage_metadata", None)
                analysis = AnalysisResult(
                    text=text,
                    model=round_spec.model_identity,
                    provider=round_spec.provider,
                    metadata={
                        "batch": True,
                        "batch_stage": round_spec.stage_name,
                        "batch_execution": "vertex_batch",
                        "decision_valid": False,
                        **({"usage_metadata": usage_metadata} if usage_metadata else {}),
                    },
                )
                cache_key = cache_keys[task.key]
                self._analysis_service.persist_cached_result(cache_key, analysis)
                persisted[task.key] = analysis
                results[task.key] = self._round_result(
                    analysis,
                    round_spec,
                    cache_key,
                    execution_id,
                    from_cache=False,
                )
                self._record_usage(
                    analysis,
                    entries_by_id[task.key],
                    tenant,
                    round_spec,
                    cache_key,
                )

        completed = 0
        for entry in entries:
            result = results.get(entry.unique_id)
            if result is not None and progress is not None:
                completed += 1
                progress(entry.unique_id, result, completed, len(entries))
        with self._pending_lock:
            self._pending_runs.setdefault(id(round_spec), deque()).append(
                (round_spec, execution_id, persisted, cache_keys),
            )
        return results

    def record_validation(
        self,
        round_spec: RoundSpec,
        validated_results: Mapping[str, bool],
    ) -> None:
        with self._pending_lock:
            pending = self._pending_runs.get(id(round_spec))
            if not pending:
                return
            execution_id = (
                validated_results.execution_id
                if isinstance(validated_results, ValidationResults)
                else None
            )
            index = 0
            if execution_id is not None:
                index = next(
                    (i for i, run in enumerate(pending) if run[1] == execution_id),
                    -1,
                )
                if index < 0:
                    return
            _spec, _execution_id, saved, cache_keys = pending[index]
            del pending[index]
            if not pending:
                del self._pending_runs[id(round_spec)]
        for unique_id, decision_valid in validated_results.items():
            analysis = saved.get(unique_id)
            if analysis is None:
                continue
            analysis.metadata.update({
                "batch_stage": round_spec.stage_name,
                "batch_execution": "vertex_batch",
                "decision_valid": decision_valid,
            })
            self._analysis_service.persist_cached_result(cache_keys[unique_id], analysis)

    def _get_many(self, cache_keys: Any) -> dict[CacheKey, AnalysisResult]:
        cache = self._analysis_service._cache  # noqa: SLF001
        keys = list(cache_keys)
        get_many = getattr(cache, "get_many", None)
        if callable(get_many):
            return dict(get_many(keys))
        return {key: value for key in keys if (value := cache.get(key)) is not None}

    @staticmethod
    def _cache_key(
        entry: CallLogEntry,
        tenant: TenantConfig,
        round_spec: RoundSpec,
    ) -> CacheKey:
        return (
            tenant.tenant_id,
            entry.unique_id,
            round_spec.prompt_key,
            round_spec.prompt_version,
            round_spec.provider,
            round_spec.model_key,
            round_spec.custom_fragment.strip(),
        )

    @staticmethod
    def _round_result(
        analysis: AnalysisResult,
        round_spec: RoundSpec,
        cache_key: CacheKey,
        execution_id: int,
        *,
        from_cache: bool,
    ) -> RoundExecutionResult:
        return RoundExecutionResult(
            raw_text=analysis.text,
            provider=analysis.provider,
            model=analysis.model,
            from_cache=from_cache,
            usage_metadata=analysis.metadata.get("usage_metadata"),
            cache_key=cache_key,
            cache_identity=round_spec.cache_identity,
            execution_id=execution_id,
        )

    def _record_usage(
        self,
        analysis: AnalysisResult,
        entry: CallLogEntry,
        tenant: TenantConfig,
        round_spec: RoundSpec,
        cache_key: CacheKey,
    ) -> None:
        tracker = self._analysis_service._usage_tracker  # noqa: SLF001
        usage = extract_usage_metadata(analysis.metadata.get("usage_metadata"))
        if tracker is None or usage is None:
            return
        tracker.record(
            entry=entry,
            tenant=tenant,
            prompt_key=round_spec.prompt_key,
            custom_fragment=round_spec.custom_fragment.strip(),
            provider_name=round_spec.provider,
            model_key=round_spec.model_key,
            mode=round_spec.usage_mode,
            usage=usage,
            cache_key=cache_key,
        )
