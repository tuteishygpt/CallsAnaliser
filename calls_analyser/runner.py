"""
Script to run daily batch analysis automatically.
Defaults to processing "yesterday's" calls.
"""
import argparse
import datetime
import logging
import sys
from types import SimpleNamespace
from typing import Any, Optional
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_batch")

try:
    from calls_analyser.ui.dependencies import build_dependencies
    from calls_analyser.ui import utils
    from calls_analyser.services.gemini_batch import VertexBatchRunner
    from calls_analyser.services.batch_executors import VertexBatchExecutor
    from calls_analyser.services.batch_orchestrator import BatchAnalysisOrchestrator
    from calls_analyser.services.batch_results import build_batch_item_row
    from calls_analyser.services.tenant_settings import TenantRuntimeSettings
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


def _orchestrator_settings(
    deps: Any,
    runtime_settings: Any | None,
) -> TenantRuntimeSettings:
    """Return complete settings even for legacy dependency fixtures."""
    return TenantRuntimeSettings(
        batch_model_key=_runtime_str_or_default(
            getattr(runtime_settings, "batch_model_key", None), deps.batch_model_key,
        ),
        batch_language_code=_runtime_str_or_default(
            getattr(runtime_settings, "batch_language_code", None),
            getattr(deps.batch_language, "value", deps.batch_language),
        ),
        batch_enabled=bool(getattr(runtime_settings, "batch_enabled", True)),
        batch_size=_runtime_int_or_default(
            getattr(runtime_settings, "batch_size", None), deps.batch_params.batch_size,
        ),
        follow_up_verification_mode=getattr(
            runtime_settings, "follow_up_verification_mode", "off",
        ),
        follow_up_verification_model_key=getattr(
            runtime_settings, "follow_up_verification_model_key", "",
        ),
        follow_up_verification_prompt_key=getattr(
            runtime_settings, "follow_up_verification_prompt_key", "",
        ),
        scheduler_filters=dict(getattr(runtime_settings, "scheduler_filters", {}) or {}),
    )


class _SchedulerPromptService:
    def __init__(self, deps: Any) -> None:
        self._deps = deps

    def get_prompt(self, key: str, *, tenant_id: str | None = None) -> Any:
        template = self._deps.prompt_service.get_prompt(key, tenant_id=tenant_id)
        body = getattr(template, "body", None)
        if not isinstance(body, str) or not body.strip():
            body = self._deps.batch_prompt_text if key == self._deps.batch_prompt_key else ""
        return SimpleNamespace(
            key=getattr(template, "key", key),
            version=getattr(template, "version", 1),
            body=body,
        )


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

    settings = _orchestrator_settings(deps, runtime_settings)
    analysis_service = deps.analysis_service
    analysis_service._call_log_service = deps.call_log_service
    analysis_service._usage_tracker = getattr(deps, "usage_tracker", None)
    if not callable(getattr(analysis_service, "persist_cached_result", None)):
        def persist_cached_result(cache_key, result):  # noqa: ANN001
            set_item = getattr(analysis_service._cache, "__setitem__", None)
            if callable(set_item):
                set_item(cache_key, result)
        analysis_service.persist_cached_result = persist_cached_result
    executor = VertexBatchExecutor(
        analysis_service,
        runner_factory=lambda model: VertexBatchRunner(model=model),
        batch_size_resolver=lambda _tenant: settings.batch_size,
    )
    orchestrator = BatchAnalysisOrchestrator(
        executor,
        prompt_service=_SchedulerPromptService(deps),
        ai_registry=deps.ai_registry,
    )
    run_result = orchestrator.run_with_settings(
        entries,
        tenant,
        settings,
        primary_prompt_key=deps.batch_prompt_key,
        primary_usage_mode="scheduler_batch",
        verification_usage_mode="scheduler_batch_verify",
    )
    logger.info(
        "Batch counters: total=%d round_1_success=%d verification_requested=%d "
        "verification_success=%d verification_changed_to_false=%d "
        "verification_failed=%d final_follow_up=%d",
        run_result.total, run_result.round_1_success,
        run_result.verification_requested, run_result.verification_success,
        run_result.verification_changed_to_false, run_result.verification_failed,
        run_result.final_follow_up,
    )
    results_df = pd.DataFrame(
        [build_batch_item_row(item, tenant) for item in run_result.items],
    )
    email_report_service = getattr(deps, "email_report_service", None)
    if email_report_service is not None:
        if run_result.round_1_success == 0:
            logger.warning("Email report skipped: no valid primary decisions to send.")
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
