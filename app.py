"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import os
import json
import tempfile
import datetime

from calls_analyser.env import load_environment

load_environment()

# ---------------------------------------------------------------------------
# Google Service Account credentials bootstrap.
#
# On Hugging Face Spaces (or any environment without a key file on disk),
# store the full JSON content of the service-account key in the secret
# ``GOOGLE_SERVICE_ACCOUNT_JSON``.  The block below writes it to a temp
# file and sets ``GOOGLE_APPLICATION_CREDENTIALS`` so that every Google
# client library picks it up automatically.
#
# Locally you can either:
#   - set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json in .env, or
#   - set GOOGLE_SERVICE_ACCOUNT_JSON with the raw JSON content.
# ---------------------------------------------------------------------------
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    import base64
    sa_b64 = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_B64", "").strip()
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_b64 and not sa_json:
        try:
            sa_json = base64.b64decode(sa_b64).decode("utf-8").strip()
        except Exception as exc:
            print(f"WARNING: GOOGLE_SERVICE_ACCOUNT_JSON_B64 decode failed: {exc}")
    if sa_json:
        try:
            json.loads(sa_json)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="gcp_sa_",
            )
            tmp.write(sa_json)
            tmp.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
            print(f"DEBUG: Wrote service-account credentials to {tmp.name}")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: GOOGLE_SERVICE_ACCOUNT_JSON is set but invalid: {exc}")

print(f"DEBUG: CWD is {os.getcwd()}")
print(f"DEBUG: SUPABASE_URL present: {bool(os.environ.get('SUPABASE_URL'))}")
print(f"DEBUG: GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '(not set)')}")

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

