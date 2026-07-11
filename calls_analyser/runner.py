"""
Script to run daily batch analysis automatically.
Defaults to processing "yesterday's" calls.
"""
import argparse
import datetime
import json
import logging
import os
import sys
from typing import Any, List, Optional
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_batch")

try:
    from calls_analyser.ui.dependencies import build_dependencies
    from calls_analyser.ui import utils
    from calls_analyser.services.gemini_batch import VertexBatchRunner, BatchTask, guess_mime_type
    from calls_analyser.services.batch_results import build_error_row, build_success_row
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


def run_batch_process(
    deps, 
    day: datetime.date,
    time_from_str: Optional[str],
    time_to_str: Optional[str],
    call_type_str: str,
    tenant_id_arg: Optional[str]
):
    if not deps.project_imports_available:
        logger.error("Project imports not available.")
        return

    if (
        getattr(deps, "tenant_settings_service", None) is None
        and not deps.batch_params.enable_gemini_batch
    ):
        logger.warning("Gemini batch is disabled in batch_params.json (enable_gemini_batch=False).")
        return

    # Resolve tenant
    tenant = deps.tenant_service.resolve(tenant_id_arg or None)
    runtime_settings = _resolve_runtime_settings(deps, tenant)

    batch_enabled = (
        bool(getattr(runtime_settings, "batch_enabled", True))
        if runtime_settings is not None
        else deps.batch_params.enable_gemini_batch
    )
    if not batch_enabled:
        logger.warning("Gemini batch is disabled for tenant %s.", tenant.tenant_id)
        return

    batch_model_key = _runtime_str_or_default(
        getattr(runtime_settings, "batch_model_key", None),
        deps.batch_model_key,
    )
    batch_size = _runtime_int_or_default(
        getattr(runtime_settings, "batch_size", None),
        deps.batch_params.batch_size,
    )
    batch_language = _runtime_language_or_default(runtime_settings, deps.batch_language)
    time_from_value = _runtime_filter_or_arg(runtime_settings, "time_from", time_from_str)
    time_to_value = _runtime_filter_or_arg(runtime_settings, "time_to", time_to_str)
    call_type_value = _runtime_filter_or_arg(runtime_settings, "call_type", call_type_str)

    # Parse and validate filters
    try:
        time_from = utils.parse_time_value(time_from_value)
        time_to = utils.parse_time_value(time_to_value)
        utils.validate_time_range(time_from, time_to)
        call_type = utils.resolve_call_type(call_type_value)
    except ValueError as e:
        logger.error(f"Invalid filter parameters: {e}")
        return

    # Fetch calls 
    logger.info(
        "Fetching calls for %s (Time: %s - %s, Type: %s)...",
        day,
        _display_filter_value(time_from_value),
        _display_filter_value(time_to_value),
        _display_filter_value(call_type_value),
    )
    
    try:
        entries = deps.call_log_service.list_calls(
            day,
            tenant,
            time_from=time_from,
            time_to=time_to,
            call_type=call_type
        )
    except Exception as e:
        logger.error(f"Error fetching calls: {e}")
        return

    if not entries:
        logger.info("No calls found for this filter.")
        return

    logger.info(f"Found {len(entries)} calls.")

    # Build prompt
    lang_instruction = GeminiAIAdapter._system_instruction(batch_language)
    prompt_text, prompt_version = _resolve_batch_prompt_text_and_version(deps, tenant)
    merged_prompt = f"[SYSTEM INSTRUCTION: {lang_instruction}]\n\n{prompt_text}".strip()
    
    provider = deps.ai_registry.get(batch_model_key)
    if not provider:
        logger.error(f"Model {batch_model_key} not found in registry.")
        return

    provider_name = getattr(provider, "provider_name", batch_model_key)
    custom_fragment = "" 

    tasks: List[BatchTask] = []
    task_indices: List[int] = []
    result_text_by_id: dict[str, str] = {}
    error_by_id: dict[str, str] = {}

    cache_entries: list[tuple[int, object, CacheKey]] = []
    for idx, entry in enumerate(entries):
        cache_entries.append(
            (
                idx,
                entry,
                (
                    tenant.tenant_id,
                    entry.unique_id,
                    deps.batch_prompt_key,
                    prompt_version,
                    provider_name,
                    batch_model_key,
                    custom_fragment,
                ),
            )
        )
    cached_results = _get_cached_results(
        deps.analysis_service._cache,
        [cache_key for _, _, cache_key in cache_entries],
    )

    # Check cache and identify missing
    cached_count = 0
    for idx, entry, cache_key in cache_entries:
        cached_result = cached_results.get(cache_key)

        if cached_result:
            cached_count += 1
            result_text_by_id[entry.unique_id] = cached_result.text
        else:
            try:
                handle = deps.call_log_service.ensure_recording(entry.unique_id, tenant)
                mime_type = guess_mime_type(handle.local_uri)
                tasks.append(
                    BatchTask(
                        key=entry.unique_id,
                        path=handle.local_uri,
                        mime_type=mime_type,
                    )
                )
                task_indices.append(idx)
            except Exception as e:
                logger.error(f"Failed to prepare audio for {entry.unique_id}: {e}")
                error_by_id[entry.unique_id] = f"❌ {e}"

    logger.info(f"Summary: {len(entries)} total. {cached_count} already cached. {len(tasks)} to process.")

    result_map: dict[str, str] = {}
    usage_by_id: dict[str, object] = {}
    if tasks:
        # Run batch via Vertex AI Batch API
        logger.info(f"Starting Vertex AI Batch for {len(tasks)} items...")
        runner = VertexBatchRunner(model=batch_model_key)

        try:
            run_batch_results = getattr(runner, "run_batch_results", None)
            if callable(run_batch_results):
                batch_results = run_batch_results(
                    tasks,
                    merged_prompt,
                    chunk_size=batch_size,
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
                result_map = runner.run_batch(
                    tasks,
                    merged_prompt,
                    chunk_size=batch_size,
                )
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            for task in tasks:
                error_by_id[task.key] = f"❌ Batch execution failed: {e}"
    else:
        logger.info("Nothing new to process; preparing report from cached results.")

    # Process results and save to cache
    success_count = 0
    for i, task in enumerate(tasks):
        original_idx = task_indices[i]
        entry = entries[original_idx]
        text_result = result_map.get(entry.unique_id)

        if text_result and not text_result.startswith("Error:"):
            # Save to cache
            cache_key = (
                tenant.tenant_id,
                entry.unique_id,
                deps.batch_prompt_key,
                prompt_version,
                provider_name,
                batch_model_key,
                custom_fragment,
            )
            usage_metadata = usage_by_id.get(entry.unique_id)
            new_result = AnalysisResult(
                text=text_result,
                model=batch_model_key,
                provider=provider_name,
                metadata={
                    "batch": True,
                    **({"usage_metadata": usage_metadata} if usage_metadata else {}),
                },
            )
            deps.analysis_service._cache[cache_key] = new_result
            usage_tracker = getattr(deps, "usage_tracker", None)
            if usage_tracker is not None:
                usage = extract_usage_metadata(usage_metadata)
                if usage is not None:
                    usage_tracker.record(
                        entry=entry,
                        tenant=tenant,
                        prompt_key=deps.batch_prompt_key,
                        custom_fragment=custom_fragment,
                        provider_name=provider_name,
                        model_key=batch_model_key,
                        mode="scheduler_batch",
                        usage=usage,
                        cache_key=cache_key,
                    )
            result_text_by_id[entry.unique_id] = text_result
            success_count += 1
            logger.info(f"Processed {entry.unique_id} successfully.")
        else:
            logger.error(f"Failed or error for {entry.unique_id}: {text_result}")
            error_by_id[entry.unique_id] = text_result or "No result returned."

    logger.info(f"Batch completed. Successfully processed and cached: {success_count}/{len(tasks)}")

    rows = []
    for entry in entries:
        if entry.unique_id in result_text_by_id:
            rows.append(build_success_row(entry, tenant, result_text_by_id[entry.unique_id]))
        else:
            rows.append(
                build_error_row(
                    entry,
                    error_by_id.get(entry.unique_id, "No result returned."),
                )
            )
    results_df = pd.DataFrame(rows)

    email_report_service = getattr(deps, "email_report_service", None)
    if email_report_service is not None:
        if not result_text_by_id:
            logger.warning("Email report skipped: no successful batch results to send.")
        else:
            try:
                email_report_service.send(
                    results_df,
                    filter_option="Needs follow-up",
                    report_date=day.isoformat(),
                    tenant_id=tenant.tenant_id,
                )
                logger.info("Email report sent successfully.")
            except Exception as e:
                logger.error(f"Email report failed: {e}")
    else:
        logger.warning("Email report skipped: BREVO_API_KEY or GOOGLE_app is not configured.")

    return results_df


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
