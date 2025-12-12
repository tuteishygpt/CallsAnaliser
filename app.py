"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

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


# ----------------------------------------------------------------------------
# Scheduler for automated daily batch (runs on Hugging Face Spaces / Servers)
# ----------------------------------------------------------------------------
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from calls_analyser import runner as daily_runner
    import datetime

    def run_scheduled_job():
        """Wrapper to run the batch job for 'yesterday'."""
        print("⏰ [Scheduler] Starting daily batch analysis...")
        # Calculate yesterday
        target_date = datetime.date.today() - datetime.timedelta(days=1)
        
        # Run the batch process using the same dependencies
        # Note: We create new dependencies inside the job to ensure clean state if needed,
        # but here reusing 'deps' is also fine if 'deps' is thread-safe.
        # For safety/updates, we might want to re-build deps or just use the global 'deps'.
        # Using global 'deps' for now as it holds the loaded secrets/config.
        bp = deps.batch_params
        daily_runner.run_batch_process(
            deps, 
            day=target_date, 
            time_from_str=bp.filter_time_from, 
            time_to_str=bp.filter_time_to, 
            call_type_str=bp.filter_call_type, 
            tenant_id_arg=None
        )
        print("✅ [Scheduler] Daily batch finished.")

    # Create and configure scheduler
    scheduler = BackgroundScheduler()
    
    # Read settings from batch_params
    bp = deps.batch_params
    
    # Define update schedule job based on params
    # We always start the scheduler, but condition valid jobs.
    if bp.scheduler_enabled:
        hour, minute = 1, 0
        try:
             # expect "HH:MM"
             parts = bp.scheduler_cron_time.split(":")
             hour = int(parts[0])
             minute = int(parts[1])
        except Exception:
             print("⚠️  [Scheduler] Invalid cron_time format. Using default 01:00.")
        
        if bp.scheduler_mode == "interval":
            interval_mins = max(1, bp.scheduler_interval_minutes)
            print(f"ℹ️  [Scheduler] Mode: INTERVAL (every {interval_mins} mins). Filters: {bp.filter_time_from}-{bp.filter_time_to}, Type: {bp.filter_call_type}")
            scheduler.add_job(run_scheduled_job, "interval", minutes=interval_mins, next_run_time=datetime.datetime.now() + datetime.timedelta(seconds=10)) 
        else:
            print(f"ℹ️  [Scheduler] Mode: CRON (at {hour:02d}:{minute:02d}). Filters: {bp.filter_time_from}-{bp.filter_time_to}, Type: {bp.filter_call_type}")
            scheduler.add_job(run_scheduled_job, "cron", hour=hour, minute=minute)
            
        scheduler.start()
        print("ℹ️  [Scheduler] Background scheduler started.")
    else:
         print("ℹ️  [Scheduler] Scheduler is disabled in batch_params.")

except ImportError as e:
    print(f"⚠️  [Scheduler] Import Error details: {e}")
    print("⚠️  [Scheduler] APScheduler not installed or import failed. Background jobs disabled.")
except Exception as e:
    print(f"⚠️  [Scheduler] Failed to start scheduler: {e}")


if __name__ == "__main__":
    demo.launch(allowed_paths=[os.environ.get("VOCHI_ALLOWED_PATH", "D:\\tmp")])
