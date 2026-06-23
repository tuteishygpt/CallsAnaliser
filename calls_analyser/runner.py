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
import tempfile
from typing import List, Optional
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Bootstrap service-account credentials from env secret (HF Spaces / CI).
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_json:
        try:
            json.loads(sa_json)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="gcp_sa_",
            )
            tmp.write(sa_json)
            tmp.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
        except (json.JSONDecodeError, OSError):
            pass

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_batch")

try:
    from calls_analyser.ui.dependencies import build_dependencies
    from calls_analyser.ui import utils
    from calls_analyser.services.gemini_batch import VertexBatchRunner, BatchTask, guess_mime_type
    from calls_analyser.services.batch_results import build_error_row, build_success_row
    from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
    from calls_analyser.domain.models import AnalysisResult
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

    if not deps.batch_params.enable_gemini_batch:
        logger.warning("Gemini batch is disabled in batch_params.json (enable_gemini_batch=False).")
        return

    # Resolve tenant
    tenant = deps.tenant_service.resolve(tenant_id_arg or None) 

    # Parse and validate filters
    try:
        time_from = utils.parse_time_value(time_from_str)
        time_to = utils.parse_time_value(time_to_str)
        utils.validate_time_range(time_from, time_to)
        call_type = utils.resolve_call_type(call_type_str)
    except ValueError as e:
        logger.error(f"Invalid filter parameters: {e}")
        return

    # Fetch calls 
    logger.info(f"Fetching calls for {day} (Time: {time_from or 'All'} - {time_to or 'All'}, Type: {call_type_str or 'All'})...")
    
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
    lang_instruction = GeminiAIAdapter._system_instruction(deps.batch_language)
    prompt_text = deps.batch_prompt_text or ""
    merged_prompt = f"[SYSTEM INSTRUCTION: {lang_instruction}]\n\n{prompt_text}".strip()
    
    provider = deps.ai_registry.get(deps.batch_model_key)
    if not provider:
        logger.error(f"Model {deps.batch_model_key} not found in registry.")
        return

    provider_name = getattr(provider, "provider_name", deps.batch_model_key)
    custom_fragment = "" 

    tasks: List[BatchTask] = []
    task_indices: List[int] = []
    result_text_by_id: dict[str, str] = {}
    error_by_id: dict[str, str] = {}

    # Check cache and identify missing
    cached_count = 0
    for idx, entry in enumerate(entries):
        cache_key = (
            tenant.tenant_id,
            entry.unique_id,
            deps.batch_prompt_key,
            provider_name,
            deps.batch_model_key,
            custom_fragment,
        )
        
        cached_result = deps.analysis_service._cache.get(cache_key)
        
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
    if tasks:
        # Run batch via Vertex AI Batch API
        logger.info(f"Starting Vertex AI Batch for {len(tasks)} items...")
        runner = VertexBatchRunner(model=deps.batch_model_key)

        try:
            result_map = runner.run_batch(
                tasks,
                merged_prompt,
                chunk_size=deps.batch_params.batch_size,
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
                provider_name,
                deps.batch_model_key,
                custom_fragment,
            )
            new_result = AnalysisResult(
                text=text_result,
                model=deps.batch_model_key,
                provider=provider_name,
                metadata={"batch": True}
            )
            deps.analysis_service._cache[cache_key] = new_result
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
        logger.warning("Email report skipped: GOOGLE_app is not configured.")

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
