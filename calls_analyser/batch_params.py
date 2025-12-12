"""Batch configuration loader for Gemini BATCH integration."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any


DEFAULT_BATCH_PARAMS_PATH = os.environ.get("BATCH_PARAMS_FILE", "batch_params.json")


@dataclass
class BatchParams:
    """Settings that toggle Gemini BATCH integration and control chunk size."""

    enable_gemini_batch: bool = True
    batch_size: int = 20
    
    # Scheduler settings
    scheduler_enabled: bool = False
    scheduler_mode: str = "cron"  # "cron" or "interval"
    scheduler_cron_time: str = "01:00"  # HH:MM
    scheduler_interval_minutes: int = 120
    
    # Auto-run filters
    filter_time_from: str | None = None
    filter_time_to: str | None = None
    filter_call_type: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchParams":
        enable = bool(payload.get("enable_gemini_batch", True))
        
        # Batch size
        try:
            size_raw = int(payload.get("batch_size", cls.batch_size))  # type: ignore[arg-type]
        except Exception:
            size_raw = cls.batch_size
        size = max(1, size_raw)
        
        # Scheduler
        sch_data = payload.get("scheduler", {})
        if not isinstance(sch_data, dict):
            sch_data = {}
            
        scheduler_enabled = bool(sch_data.get("enabled", False))
        scheduler_mode = str(sch_data.get("mode", "cron"))
        scheduler_cron_time = str(sch_data.get("cron_time", "01:00"))
        try:
            scheduler_interval_minutes = int(sch_data.get("interval_minutes", 120))
        except:
            scheduler_interval_minutes = 120
            
        filters = sch_data.get("filters", {})
        filter_time_from = filters.get("time_from")
        filter_time_to = filters.get("time_to")
        filter_call_type = filters.get("call_type")

        return cls(
            enable_gemini_batch=enable, 
            batch_size=size,
            
            scheduler_enabled=scheduler_enabled,
            scheduler_mode=scheduler_mode,
            scheduler_cron_time=scheduler_cron_time,
            scheduler_interval_minutes=scheduler_interval_minutes,
            
            filter_time_from=filter_time_from,
            filter_time_to=filter_time_to,
            filter_call_type=filter_call_type
        )


def load_batch_params(path: str = DEFAULT_BATCH_PARAMS_PATH) -> BatchParams:
    """Load batch parameters from JSON file if present, otherwise defaults."""

    if not path:
        return BatchParams()

    # Try CWD
    if os.path.exists(path):
        final_path = path
    else:
        # Try project root (assuming this file is in calls_analyser/)
        # calls_analyser/batch_params.py -> .. -> project_root
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        possible_path = os.path.join(project_root, os.path.basename(path))
        if os.path.exists(possible_path):
            final_path = possible_path
        else:
             print(f"DEBUG: batch_params.json not found at {path} or {possible_path}, using defaults.")
             return BatchParams()

    try:
        with open(final_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return BatchParams()
        return BatchParams.from_dict(data)
    except Exception as e:
        print(f"DEBUG: Failed to load batch_params: {e}")
        return BatchParams()
