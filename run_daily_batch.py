"""
Script to run daily batch analysis automatically.
Defaults to processing "yesterday's" calls.

Usage:
    python run_daily_batch.py
    python run_daily_batch.py --date 2023-10-27 --time-from 09:00 --time-to 17:00 --call-type MISSED
"""
import argparse
import datetime
import logging
import sys
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_batch")

try:
    from calls_analyser.ui.dependencies import build_dependencies
    from calls_analyser.ui import utils
    from calls_analyser.services.gemini_batch import GeminiBatchRunner, BatchTask, guess_mime_type
    from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
    from calls_analyser.domain.models import AnalysisResult
    from calls_analyser.domain.exceptions import AIModelError
except ImportError:
    logger.error("Could not import project dependencies. Make sure you are running from the project root.")
    sys.exit(1)


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

    # Prepare for batch
    api_key = deps.secrets_adapter.get_optional_secret("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found.")
        return

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

    logger.info(f"Summary: {len(entries)} total. {cached_count} already cached. {len(tasks)} to process.")

    if not tasks:
        logger.info("Nothing to process. All done.")
        return

    # Run batch
    logger.info(f"Starting Gemini Batch for {len(tasks)} items...")
    runner = GeminiBatchRunner(api_key=api_key, model=deps.batch_model_key)
    
    try:
        result_map = runner.run_batch(
            tasks,
            merged_prompt,
            chunk_size=deps.batch_params.batch_size,
        )
    except Exception as e:
        logger.error(f"Batch execution failed: {e}")
        return

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
            success_count += 1
            logger.info(f"Processed {entry.unique_id} successfully.")
        else:
            logger.error(f"Failed or error for {entry.unique_id}: {text_result}")

    logger.info(f"Batch completed. Successfully processed and cached: {success_count}/{len(tasks)}")


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
