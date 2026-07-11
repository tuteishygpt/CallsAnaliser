"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import os
import json
import datetime

from calls_analyser.env import load_environment

load_environment()

print(f"DEBUG: CWD is {os.getcwd()}")
print(f"DEBUG: SUPABASE_URL present: {bool(os.environ.get('SUPABASE_URL'))}")

from calls_analyser.ui import config as ui_config
from calls_analyser.ui.dependencies import build_dependencies
from calls_analyser.ui.handlers import UIHandlers
from calls_analyser.ui.layout import build_demo


deps = build_dependencies()
handlers = UIHandlers(deps)

# Diagnostic: Check Supabase connection
print("DEBUG: Checking DB connection from app.py...")
try:
    if hasattr(deps.analysis_service._cache, "_table"):
        count = deps.analysis_service._cache._table.select("*", count="exact", head=True).execute().count
        print(f"DEBUG: Successfully connected to Supabase. Table 'analysis_results' has {count} rows.")
    else:
        print("DEBUG: Using local file cache (not Supabase). Check configuration.")
except Exception as e:
    print(f"DEBUG: Failed to query Supabase: {e}")


def _build_app():
    return build_demo(deps, handlers)


demo = _build_app()

# Expose configuration for tests
PROJECT_IMPORTS_AVAILABLE = deps.project_imports_available
tenant_service = deps.tenant_service
call_log_service = deps.call_log_service
ai_registry = deps.ai_registry
analysis_service = deps.analysis_service
BATCH_MODEL_KEY = deps.batch_model_key
BATCH_PROMPT_KEY = deps.batch_prompt_key
BATCH_PROMPT_TEXT = deps.batch_prompt_text
BATCH_LANGUAGE = deps.batch_language
Language = ui_config.Language


def _sync_test_overrides() -> None:
    """Update handler dependencies with any monkeypatched globals (used in tests)."""

    handlers.deps.project_imports_available = PROJECT_IMPORTS_AVAILABLE
    handlers.deps.tenant_service = tenant_service
    handlers.deps.call_log_service = call_log_service
    handlers.deps.ai_registry = ai_registry
    handlers.deps.analysis_service = analysis_service
    handlers.deps.batch_model_key = BATCH_MODEL_KEY
    handlers.deps.batch_prompt_key = BATCH_PROMPT_KEY
    handlers.deps.batch_prompt_text = BATCH_PROMPT_TEXT
    handlers.deps.batch_language = BATCH_LANGUAGE


def ui_mass_analyze(date_value, time_from_value, time_to_value, call_type_value, tenant_id, authed):
    """Thin wrapper used in tests to run the batch pipeline."""

    _sync_test_overrides()

    return handlers._run_mass_analyze(  # noqa: SLF001
        date_value,
        time_from_value,
        time_to_value,
        call_type_value,
        tenant_id,
        authed,
        custom_prompt_override=None,
    )


def _register_scheduler_job_if_available(scheduler, deps, job) -> bool:
    """Register the background scheduler job when the current config allows it."""

    bp = deps.batch_params
    tenant_settings_service = getattr(deps, "tenant_settings_service", None)

    if tenant_settings_service is not None:
        enabled_tenants = tenant_settings_service.list_scheduler_enabled_tenants() or []
        if not enabled_tenants:
            print("ℹ️  [Scheduler] No tenants opted in to scheduler. Background jobs disabled.")
            return False
    elif not bp.scheduler_enabled:
        print("ℹ️  [Scheduler] Scheduler is disabled in batch_params.")
        return False

    hour, minute = 1, 0
    try:
        parts = bp.scheduler_cron_time.split(":")
        hour = int(parts[0])
        minute = int(parts[1])
    except Exception:
        print("⚠️  [Scheduler] Invalid cron_time format. Using default 01:00.")

    if bp.scheduler_mode == "interval":
        interval_mins = max(1, bp.scheduler_interval_minutes)
        print(
            f"ℹ️  [Scheduler] Mode: INTERVAL (every {interval_mins} mins). "
            f"Filters: {bp.filter_time_from}-{bp.filter_time_to}, Type: {bp.filter_call_type}"
        )
        scheduler.add_job(
            job,
            "interval",
            minutes=interval_mins,
            next_run_time=datetime.datetime.now() + datetime.timedelta(seconds=10),
        )
    else:
        print(
            f"ℹ️  [Scheduler] Mode: CRON (at {hour:02d}:{minute:02d}). "
            f"Filters: {bp.filter_time_from}-{bp.filter_time_to}, Type: {bp.filter_call_type}"
        )
        scheduler.add_job(job, "cron", hour=hour, minute=minute)

    scheduler.start()
    print("ℹ️  [Scheduler] Background scheduler started.")
    return True


# ----------------------------------------------------------------------------
# Scheduler for automated daily batch (runs on Hugging Face Spaces / Servers)
# ----------------------------------------------------------------------------
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from calls_analyser import runner as daily_runner
    from calls_analyser.services.scheduler import run_scheduled_batches_for_enabled_tenants

    def run_scheduled_job():
        """Wrapper to run the batch job for 'yesterday'."""
        print("⏰ [Scheduler] Starting daily batch analysis...")
        target_date = datetime.date.today() - datetime.timedelta(days=1)

        tenant_settings_service = getattr(deps, "tenant_settings_service", None)
        if tenant_settings_service is not None:
            try:
                summary = run_scheduled_batches_for_enabled_tenants(
                    tenant_settings_service=tenant_settings_service,
                    runner=daily_runner.run_batch_process,
                    deps=deps,
                    day=target_date,
                )
            except Exception as e:
                print(f"WARNING [Scheduler] Multi-tenant scheduled batch failed: {e}")
                raise

            print(
                "[Scheduler] Multi-tenant daily batch finished. "
                f"Successes: {len(summary.successes)}, failures: {len(summary.failures)}."
            )
            for failure in summary.failures:
                print(
                    "WARNING [Scheduler] Tenant batch failed "
                    f"tenant_id={failure.tenant_id} "
                    f"error_type={failure.exception_type}: {failure.error}"
                )
            return summary

        bp = deps.batch_params
        daily_runner.run_batch_process(
            deps,
            day=target_date,
            time_from_str=bp.filter_time_from,
            time_to_str=bp.filter_time_to,
            call_type_str=bp.filter_call_type,
            tenant_id_arg=None,
        )
        print("✅ [Scheduler] Daily batch finished.")

    scheduler = BackgroundScheduler()
    _register_scheduler_job_if_available(scheduler, deps, run_scheduled_job)

except ImportError as e:
    print(f"⚠️  [Scheduler] Import Error details: {e}")
    print("⚠️  [Scheduler] APScheduler not installed or import failed. Background jobs disabled.")
except Exception as e:
    print(f"⚠️  [Scheduler] Failed to start scheduler: {e}")


if __name__ == "__main__":
    demo.launch(
    allowed_paths=[os.environ.get("VOCHI_ALLOWED_PATH", "D:\\tmp")],
    ssr_mode=False,  # адключаем SSR
)

