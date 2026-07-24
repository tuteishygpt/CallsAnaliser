"""
Script to run daily batch analysis automatically.
Defaults to processing "yesterday's" calls.
"""
import argparse
import datetime
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_batch")

try:
    from calls_analyser.ui.dependencies import (
        build_dependencies,
        build_email_report_service_for_settings,
    )
    from calls_analyser.ui import utils
    from calls_analyser.services.gemini_batch import VertexBatchRunner, BatchTask, guess_mime_type
    from calls_analyser.services.batch_results import (
        BatchRunResult,
        FollowUpResult,
        build_error_row,
        build_success_row,
        parse_follow_up_result,
    )
    from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
    from calls_analyser.domain.models import AnalysisResult, Language
    from calls_analyser.services.analysis import CacheKey
    from calls_analyser.services.usage import extract_usage_metadata
    # from calls_analyser.domain.exceptions import AIModelError
except ImportError:
    # If run directly from outside the package context without installation
    logger.error("Could not import project dependencies. Make sure you are running from the project root.")
    # sys.exit(1) # Don't exit, let function call fail if used


def parse_args():
    parser = argparse.ArgumentParser(description="Run daily batch analysis.")
    parser.add_argument(
        "--date",
        type=str,
        help="Date to analyze (YYYY-MM-DD). Defaults to yesterday.",
        default=None
    )
    parser.add_argument(
        "--time-from",
        type=str,
        help="Start time (HH:MM). Default: 00:00",
        default=None
    )
    parser.add_argument(
        "--time-to",
        type=str,
        help="End time (HH:MM). Default: 23:59",
        default=None
    )
    parser.add_argument(
        "--call-type",
        type=str,
        help="Filter by call type (e.g. ANSWERED, MISSED, VOICEMAIL).",
        default=""
    )
    parser.add_argument(
        "--tenant-id",
        type=str,
        help="Override Tenant ID.",
        default=None
    )
    return parser.parse_args()


def get_target_date(date_str: str | None) -> datetime.date:
    if date_str:
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        # Default to yesterday
        return datetime.date.today() - datetime.timedelta(days=1)


def _get_cached_results(cache, cache_keys: list[CacheKey]) -> dict[CacheKey, AnalysisResult]:  # noqa: ANN001
    get_many = getattr(cache, "get_many", None)
    if callable(get_many):
        try:
            return dict(get_many(cache_keys))
        except Exception as e:
            logger.warning(f"Bulk cache lookup failed: {e}. Continuing with uncached items.")
            return {}

    cached_results: dict[CacheKey, AnalysisResult] = {}
    for cache_key in cache_keys:
        try:
            cached_result = cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache lookup failed for {cache_key[1]}: {e}")
            continue
        if cached_result:
            cached_results[cache_key] = cached_result
    return cached_results


def _has_runtime_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _resolve_runtime_settings(deps: Any, tenant: Any) -> Any | None:
    tenant_settings_service = getattr(deps, "tenant_settings_service", None)
    if tenant_settings_service is None:
        return None

    resolve = getattr(tenant_settings_service, "resolve", None)
    if not callable(resolve):
        return None
    return resolve(tenant.tenant_id)


def _runtime_str_or_default(value: object, default: str) -> str:
    if not _has_runtime_value(value):
        return default
    return str(getattr(value, "value", value)).strip()


def _runtime_int_or_default(value: object, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(str(getattr(value, "value", value)).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _runtime_filter_or_arg(runtime_settings: Any | None, name: str, explicit_value: object) -> object:
    if _has_runtime_value(explicit_value):
        return explicit_value

    scheduler_filters = getattr(runtime_settings, "scheduler_filters", None)
    if isinstance(scheduler_filters, dict):
        return scheduler_filters.get(name)
    return explicit_value


def _display_filter_value(value: object) -> str:
    return str(value) if _has_runtime_value(value) else "All"


def _runtime_language_or_default(runtime_settings: Any | None, default_language: object) -> object:
    language_code = getattr(runtime_settings, "batch_language_code", None)
    if not _has_runtime_value(language_code):
        return default_language

    normalized = str(getattr(language_code, "value", language_code)).strip()
    if normalized.lower() in {"auto", "default"}:
        return Language.AUTO
    try:
        return Language(normalized)
    except ValueError:
        logger.warning(
            "Invalid tenant batch language %r. Falling back to configured batch language.",
            normalized,
        )
        return default_language


def _resolve_batch_prompt_text_and_version(deps: Any, tenant: Any) -> tuple[str, int]:
    fallback_text = getattr(deps, "batch_prompt_text", None) or ""
    prompt_version = 1
    prompt_service = getattr(deps, "prompt_service", None)
    get_prompt = getattr(prompt_service, "get_prompt", None)
    if not callable(get_prompt):
        return fallback_text, prompt_version

    try:
        prompt_template = get_prompt(
            deps.batch_prompt_key,
            tenant_id=tenant.tenant_id,
        )
    except Exception as e:
        logger.warning(
            "Failed to resolve batch prompt for tenant %s: %s. Falling back to configured prompt text.",
            tenant.tenant_id,
            e,
        )
        return fallback_text, prompt_version

    prompt_version = getattr(prompt_template, "version", prompt_version)
    prompt_body = getattr(prompt_template, "body", None)
    if isinstance(prompt_body, str) and prompt_body.strip():
        return prompt_body, prompt_version
    if prompt_body is not None and not isinstance(prompt_body, str):
        prompt_body_text = str(prompt_body)
        if prompt_body_text.strip():
            return prompt_body_text, prompt_version
    return fallback_text, prompt_version


@dataclass(frozen=True)
class BatchExecutionContext:
    """One immutable identity and input snapshot for a complete batch run."""

    tenant: Any
    prompt_key: str
    batch_model_key: str
    provider_name: str
    batch_size: int
    batch_language: Any
    merged_prompt: str
    prompt_version: int
    time_from: Any
    time_to: Any
    call_type: Any
    email_to: str = ""
    email_from: str = ""
    email_from_name: str = ""


def resolve_batch_execution_context(
    deps: Any,
    *,
    tenant_id: str | None = None,
    time_from_str: object = None,
    time_to_str: object = None,
    call_type_str: object = "",
    tenant: Any | None = None,
    runtime_settings: Any | None = None,
) -> BatchExecutionContext:
    """Resolve scheduler/cache identity and all matching execution inputs once."""
    resolved_tenant = tenant or deps.tenant_service.resolve(tenant_id or None)
    resolved_settings = (
        runtime_settings
        if runtime_settings is not None
        else _resolve_runtime_settings(deps, resolved_tenant)
    )
    batch_model_key = _runtime_str_or_default(
        getattr(resolved_settings, "batch_model_key", None),
        deps.batch_model_key,
    )
    provider = deps.ai_registry.get(batch_model_key)
    if not provider:
        raise ValueError(f"Model {batch_model_key} not found in registry.")

    batch_language = _runtime_language_or_default(
        resolved_settings,
        deps.batch_language,
    )
    prompt_text, prompt_version = _resolve_batch_prompt_text_and_version(
        deps,
        resolved_tenant,
    )
    lang_instruction = GeminiAIAdapter._system_instruction(batch_language)
    merged_prompt = f"[SYSTEM INSTRUCTION: {lang_instruction}]\n\n{prompt_text}".strip()

    time_from_value = _runtime_filter_or_arg(
        resolved_settings,
        "time_from",
        time_from_str,
    )
    time_to_value = _runtime_filter_or_arg(
        resolved_settings,
        "time_to",
        time_to_str,
    )
    call_type_value = _runtime_filter_or_arg(
        resolved_settings,
        "call_type",
        call_type_str,
    )
    time_from = utils.parse_time_value(time_from_value)
    time_to = utils.parse_time_value(time_to_value)
    utils.validate_time_range(time_from, time_to)
    call_type = utils.resolve_call_type(call_type_value)

    return BatchExecutionContext(
        tenant=resolved_tenant,
        prompt_key=str(deps.batch_prompt_key),
        batch_model_key=batch_model_key,
        provider_name=str(getattr(provider, "provider_name", batch_model_key)),
        batch_size=_runtime_int_or_default(
            getattr(resolved_settings, "batch_size", None),
            deps.batch_params.batch_size,
        ),
        batch_language=batch_language,
        merged_prompt=merged_prompt,
        prompt_version=int(prompt_version),
        time_from=time_from,
        time_to=time_to,
        call_type=call_type,
        email_to=getattr(resolved_settings, "email_to", "") if resolved_settings else "",
        email_from=getattr(resolved_settings, "email_from", "") if resolved_settings else "",
        email_from_name=getattr(resolved_settings, "email_from_name", "") if resolved_settings else "",
    )


def run_batch_process(
    deps: Any,
    day: datetime.date,
    time_from_str: Optional[str],
    time_to_str: Optional[str],
    call_type_str: str,
    tenant_id_arg: Optional[str],
    *,
    execution_context: BatchExecutionContext | None = None,
) -> BatchRunResult:
    """Run one batch with strict validation and per-item failure isolation."""
    failed_empty = BatchRunResult("failed", 0, 0, 0, 0)
    if not deps.project_imports_available:
        logger.error("Project imports not available.")
        return failed_empty

    if execution_context is None:
        if (
            getattr(deps, "tenant_settings_service", None) is None
            and not deps.batch_params.enable_gemini_batch
        ):
            logger.warning("Gemini batch is disabled.")
            return failed_empty
        try:
            tenant = deps.tenant_service.resolve(tenant_id_arg or None)
            runtime_settings = _resolve_runtime_settings(deps, tenant)
        except Exception as exc:
            logger.error("Failed to resolve tenant batch settings: %s", exc)
            return failed_empty
        batch_enabled = (
            bool(getattr(runtime_settings, "batch_enabled", True))
            if runtime_settings is not None
            else deps.batch_params.enable_gemini_batch
        )
        if not batch_enabled:
            logger.warning("Gemini batch is disabled for tenant %s.", tenant.tenant_id)
            return failed_empty
        try:
            execution_context = resolve_batch_execution_context(
                deps,
                tenant=tenant,
                runtime_settings=runtime_settings,
                time_from_str=time_from_str,
                time_to_str=time_to_str,
                call_type_str=call_type_str,
            )
        except Exception as exc:
            logger.error("Invalid batch execution context: %s", exc)
            return failed_empty

    context = execution_context
    tenant = context.tenant
    try:
        entries = list(
            deps.call_log_service.list_calls(
                day,
                tenant,
                time_from=context.time_from,
                time_to=context.time_to,
                call_type=context.call_type,
            )
        )
    except Exception as exc:
        logger.error("Error fetching calls: %s", exc)
        return failed_empty
    if not entries:
        logger.info("No calls found for this filter.")
        return BatchRunResult.from_counts(
            total_count=0,
            success_count=0,
            failure_count=0,
        )

    custom_fragment = ""
    cache_entries: list[tuple[int, object, CacheKey]] = []
    for index, entry in enumerate(entries):
        cache_entries.append(
            (
                index,
                entry,
                (
                    tenant.tenant_id,
                    entry.unique_id,
                    context.prompt_key,
                    context.prompt_version,
                    context.provider_name,
                    context.batch_model_key,
                    custom_fragment,
                ),
            )
        )
    cached_results = _get_cached_results(
        deps.analysis_service._cache,
        [cache_key for _, _, cache_key in cache_entries],
    )
    cache_key_by_id = {
        entry.unique_id: cache_key for _, entry, cache_key in cache_entries
    }

    tasks: list[BatchTask] = []
    task_indices: list[int] = []
    parsed_result_by_id: dict[str, FollowUpResult] = {}
    error_by_id: dict[str, str] = {}
    cached_count = 0
    for index, entry, cache_key in cache_entries:
        cached_result = cached_results.get(cache_key)
        if cached_result:
            cached_count += 1
            try:
                parsed_result_by_id[entry.unique_id] = parse_follow_up_result(
                    cached_result.text
                )
            except Exception as exc:
                logger.error("Invalid cached result for %s: %s", entry.unique_id, exc)
                error_by_id[entry.unique_id] = f"Invalid cached result: {exc}"
            continue
        try:
            handle = deps.call_log_service.ensure_recording(entry.unique_id, tenant)
            tasks.append(
                BatchTask(
                    key=entry.unique_id,
                    path=handle.local_uri,
                    mime_type=guess_mime_type(handle.local_uri),
                )
            )
            task_indices.append(index)
        except Exception as exc:
            logger.error("Failed to prepare audio for %s: %s", entry.unique_id, exc)
            error_by_id[entry.unique_id] = str(exc)

    result_map: dict[str, str] = {}
    usage_by_id: dict[str, object] = {}
    if tasks:
        try:
            batch_runner = VertexBatchRunner(model=context.batch_model_key)
            run_batch_results = getattr(batch_runner, "run_batch_results", None)
            if callable(run_batch_results):
                batch_results = run_batch_results(
                    tasks,
                    context.merged_prompt,
                    chunk_size=context.batch_size,
                )
                result_map = {
                    key: getattr(value, "text", str(value))
                    for key, value in batch_results.items()
                }
                usage_by_id = {
                    key: getattr(value, "usage_metadata", None)
                    for key, value in batch_results.items()
                }
            else:
                result_map = batch_runner.run_batch(
                    tasks,
                    context.merged_prompt,
                    chunk_size=context.batch_size,
                )
        except Exception as exc:
            logger.error("Batch execution failed: %s", exc)
            for task in tasks:
                error_by_id[task.key] = f"Batch execution failed: {exc}"

    for task_index, task in enumerate(tasks):
        entry = entries[task_indices[task_index]]
        text_result = result_map.get(entry.unique_id)
        if not isinstance(text_result, str) or not text_result:
            error_by_id[entry.unique_id] = "No valid text result returned."
            continue
        if text_result.startswith("Error:"):
            error_by_id[entry.unique_id] = text_result or "No result returned."
            continue
        try:
            parsed_result = parse_follow_up_result(text_result)
        except Exception as exc:
            logger.error("Invalid model result for %s: %s", entry.unique_id, exc)
            error_by_id[entry.unique_id] = f"Invalid model result: {exc}"
            continue

        cache_key = cache_key_by_id[entry.unique_id]
        usage_metadata = usage_by_id.get(entry.unique_id)
        new_result = AnalysisResult(
            text=text_result,
            model=context.batch_model_key,
            provider=context.provider_name,
            metadata={
                "batch": True,
                **({"usage_metadata": usage_metadata} if usage_metadata else {}),
            },
        )
        try:
            deps.analysis_service._cache[cache_key] = new_result
        except Exception as exc:
            logger.error("Cache write failed for %s: %s", entry.unique_id, exc)
            error_by_id[entry.unique_id] = f"Cache write failed: {exc}"
            continue

        usage_tracker = getattr(deps, "usage_tracker", None)
        if usage_tracker is not None:
            try:
                usage = extract_usage_metadata(usage_metadata)
                if usage is not None:
                    usage_tracker.record(
                        entry=entry,
                        tenant=tenant,
                        prompt_key=context.prompt_key,
                        custom_fragment=custom_fragment,
                        provider_name=context.provider_name,
                        model_key=context.batch_model_key,
                        mode="scheduler_batch",
                        usage=usage,
                        cache_key=cache_key,
                    )
            except Exception as exc:
                logger.error("Usage write failed for %s: %s", entry.unique_id, exc)
        parsed_result_by_id[entry.unique_id] = parsed_result

    rows = [
        (
            build_success_row(entry, tenant, parsed_result_by_id[entry.unique_id])
            if entry.unique_id in parsed_result_by_id
            else build_error_row(
                entry,
                error_by_id.get(entry.unique_id, "No result returned."),
            )
        )
        for entry in entries
    ]
    results_df = pd.DataFrame(rows)
    email_report_service = build_email_report_service_for_settings(
        context,
        fallback_service=getattr(deps, "email_report_service", None),
    )
    if email_report_service is not None and parsed_result_by_id:
        try:
            email_report_service.send(
                results_df,
                filter_option="Needs follow-up",
                report_date=day.isoformat(),
                tenant_id=tenant.tenant_id,
            )
        except Exception as exc:
            logger.error("Email report failed: %s", exc)

    success_count = len(parsed_result_by_id)
    return BatchRunResult.from_counts(
        total_count=len(entries),
        success_count=success_count,
        failure_count=len(entries) - success_count,
        cached_count=cached_count,
    )


def main():
    args = parse_args()
    target_date = get_target_date(args.date)
    
    logger.info("Initializing dependencies...")
    deps = build_dependencies()
    
    run_batch_process(
        deps, 
        target_date, 
        args.time_from, 
        args.time_to, 
        args.call_type, 
        args.tenant_id
    )


if __name__ == "__main__":
    main()
