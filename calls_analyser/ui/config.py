"""Configuration and constants for the Gradio UI."""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

try:  # pragma: no cover - optional dependency wiring
    from calls_analyser.domain.models import Language
    from calls_analyser.config import (
        PROMPTS as CFG_PROMPTS,
        MODEL_CANDIDATES as CFG_MODEL_CANDIDATES,
        BATCH_MODEL_KEY as CFG_BATCH_MODEL_KEY,
        BATCH_PROMPT_KEY as CFG_BATCH_PROMPT_KEY,
        BATCH_PROMPT_TEXT as CFG_BATCH_PROMPT_TEXT,
        BATCH_LANGUAGE_CODE as CFG_BATCH_LANGUAGE_CODE,
        BATCH_custom as CFG_BATCH_CUSTOM,
        BATCH_CUSTOM_CONDITIONS_DEFAULT as CFG_BATCH_CUSTOM_CONDITIONS_DEFAULT,
        BATCH_CUSTOM_PROMPT_TEMPLATE as CFG_BATCH_CUSTOM_PROMPT_TEMPLATE,
    )
    PROJECT_IMPORTS_AVAILABLE = True
except ImportError:  # pragma: no cover - executed when project deps unavailable
    PROJECT_IMPORTS_AVAILABLE = False

    class Language:  # type: ignore[override]
        RUSSIAN = "ru"
        BELARUSIAN = "be"
        ENGLISH = "en"
        AUTO = "auto"

        def __call__(self, value: str):  # minimal compatibility for tests
            return value

    CFG_PROMPTS: Dict[str, object] = {}
    CFG_MODEL_CANDIDATES: List[Tuple[str, str]] = []
    CFG_BATCH_MODEL_KEY = ""
    CFG_BATCH_PROMPT_KEY = ""
    CFG_BATCH_PROMPT_TEXT = ""
    CFG_BATCH_LANGUAGE_CODE = "auto"
    CFG_BATCH_CUSTOM = ""
    CFG_BATCH_CUSTOM_CONDITIONS_DEFAULT = ""
    CFG_BATCH_CUSTOM_PROMPT_TEMPLATE = ""

PROMPTS = CFG_PROMPTS if PROJECT_IMPORTS_AVAILABLE else {}
TPL_OPTIONS = [(tpl.title, tpl.key) for tpl in PROMPTS.values()] + [("Custom", "custom")]
LANG_OPTIONS = [
    ("Russian", Language.RUSSIAN),
    ("Auto", Language.AUTO),
    ("Belarusian", Language.BELARUSIAN),
    ("English", Language.ENGLISH),
]
CALL_TYPE_OPTIONS = [
    ("All types", ""),
    ("Inbound", "0"),
    ("Outbound", "1"),
    ("Internal", "2"),
]
MODEL_CANDIDATES = CFG_MODEL_CANDIDATES if PROJECT_IMPORTS_AVAILABLE else []

DEFAULT_TENANT_ID = os.environ.get("DEFAULT_TENANT_ID", "Amedis")
DEFAULT_BASE_URL = os.environ.get("VOCHI_BASE_URL", "https://bot.vochi.by/api/v1")

BATCH_PROMPT_KEY = CFG_BATCH_PROMPT_KEY
BATCH_PROMPT_TEXT = (CFG_BATCH_PROMPT_TEXT or "").strip()
BATCH_MODEL_KEY = CFG_BATCH_MODEL_KEY
BATCH_LANGUAGE_CODE = CFG_BATCH_LANGUAGE_CODE
BATCH_CUSTOM = CFG_BATCH_CUSTOM
BATCH_CUSTOM_CONDITIONS_DEFAULT = CFG_BATCH_CUSTOM_CONDITIONS_DEFAULT
BATCH_CUSTOM_PROMPT_TEMPLATE = CFG_BATCH_CUSTOM_PROMPT_TEMPLATE
