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

    enable_gemini_batch: bool = False
    batch_size: int = 20

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchParams":
        enable = bool(payload.get("enable_gemini_batch", False))
        try:
            size_raw = int(payload.get("batch_size", cls.batch_size))  # type: ignore[arg-type]
        except Exception:
            size_raw = cls.batch_size
        size = max(1, size_raw)
        return cls(enable_gemini_batch=enable, batch_size=size)


def load_batch_params(path: str = DEFAULT_BATCH_PARAMS_PATH) -> BatchParams:
    """Load batch parameters from JSON file if present, otherwise defaults."""

    if not path:
        return BatchParams()

    if not os.path.exists(path):
        return BatchParams()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return BatchParams()
        return BatchParams.from_dict(data)
    except Exception:
        return BatchParams()
