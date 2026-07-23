"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import datetime
import copy
import os
from typing import Any

from calls_analyser.env import load_environment

load_environment()

print(f"DEBUG: CWD is {os.getcwd()}")
print(f"DEBUG: SUPABASE_URL present: {bool(os.environ.get('SUPABASE_URL'))}")

from calls_analyser.services.scheduler import (
    cron_scheduled_for,
    interval_scheduled_for,
    run_scheduled_batch_for_tenant,
    scheduler_timezone,
)
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
        count = (
            deps.analysis_service._cache._table.select(
                "*",
                count="exact",
                head=True,
            )
            .execute()
            .count
        )
        print(
            "DEBUG: Successfully connected to Supabase. "
            f"Table 'analysis_results' has {count} rows."
        )
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


def ui_mass_analyze(
    date_value,
    time_from_value,
    time_to_value,
    call_type_value,
    tenant_id,
    authed,
):
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


def _scheduler_now(timezone: datetime.tzinfo) -> datetime.datetime:
    """Return an aware wall-clock value in the configured scheduler timezone."""
    return datetime.datetime.now(timezone)


def _register_scheduler_jobs_if_available(
    scheduler: Any,
    deps: Any,
    *,
    runner: Any,
) -> bool:
    """Register one guarded startup-snapshot job per enabled tenant."""

    run_repository = getattr(deps, "scheduler_run_repository", None)
    if run_repository is None:
        print(
            "WARNING [Scheduler] scheduler_runs repository is unavailable. "
            "Scheduled execution disabled (fail closed)."
        )
        return False

    tenant_settings_service = getattr(deps, "tenant_settings_service", None)
    if tenant_settings_service is None:
        print(
            "WARNING [Scheduler] Tenant settings service is unavailable. "
            "Scheduled execution disabled."
        )
        return False

    timezone = scheduler_timezone(os.environ.get("SCHEDULER_TIMEZONE"))
    enabled_tenants = tenant_settings_service.list_scheduler_enabled_tenants() or []
    if not enabled_tenants:
        print("[Scheduler] No tenants opted in. Background jobs disabled.")
        return False

    for tenant_id in enabled_tenants:
        runtime_settings = copy.deepcopy(
            tenant_settings_service.resolve(tenant_id)
        )
        mode = runtime_settings.scheduler_mode

        if mode == "interval":
            interval_minutes = runtime_settings.scheduler_interval_minutes

            def run_interval_job(
                *,
                _tenant_id=tenant_id,
                _runtime_settings=runtime_settings,
                _interval_minutes=interval_minutes,
            ):
                now = _scheduler_now(timezone)
                return run_scheduled_batch_for_tenant(
                    tenant_id=_tenant_id,
                    runtime_settings=_runtime_settings,
                    scheduled_for=interval_scheduled_for(now, _interval_minutes),
                    now=now,
                    run_repository=run_repository,
                    runner=runner,
                    deps=deps,
                )

            scheduler.add_job(
                run_interval_job,
                "interval",
                minutes=interval_minutes,
                timezone=timezone,
                id=f"scheduler:{tenant_id}",
                replace_existing=True,
                max_instances=1,
            )
            continue

        cron_time = datetime.time.fromisoformat(runtime_settings.scheduler_cron_time)
        cron_time = cron_time.replace(second=0, microsecond=0, tzinfo=None)

        def run_cron_job(
            *,
            _tenant_id=tenant_id,
            _runtime_settings=runtime_settings,
            _cron_time=cron_time,
        ):
            now = _scheduler_now(timezone)
            return run_scheduled_batch_for_tenant(
                tenant_id=_tenant_id,
                runtime_settings=_runtime_settings,
                scheduled_for=cron_scheduled_for(now, _cron_time),
                now=now,
                run_repository=run_repository,
                runner=runner,
                deps=deps,
            )

        scheduler.add_job(
            run_cron_job,
            "cron",
            hour=cron_time.hour,
            minute=cron_time.minute,
            timezone=timezone,
            id=f"scheduler:{tenant_id}",
            replace_existing=True,
            max_instances=1,
        )

    scheduler.start()
    print(f"[Scheduler] Started {len(enabled_tenants)} guarded tenant job(s).")
    return True


# Scheduler for automated daily batch (runs on Hugging Face Spaces / servers).
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from calls_analyser import runner as daily_runner

    scheduler = BackgroundScheduler()
    _register_scheduler_jobs_if_available(
        scheduler,
        deps,
        runner=daily_runner.run_batch_process,
    )
except ImportError as e:
    print(f"WARNING [Scheduler] Import error: {e}")
    print("WARNING [Scheduler] APScheduler unavailable. Background jobs disabled.")
except Exception as e:
    print(f"WARNING [Scheduler] Failed to start scheduler: {e}")


if __name__ == "__main__":
    demo.launch(
        allowed_paths=[os.environ.get("VOCHI_ALLOWED_PATH", r"D:\tmp")],
        ssr_mode=False,
    )
