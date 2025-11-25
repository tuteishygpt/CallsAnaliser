"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import os

from calls_analyser.ui import config as ui_config
from calls_analyser.ui.dependencies import build_dependencies
from calls_analyser.ui.handlers import UIHandlers
from calls_analyser.ui.layout import build_demo


deps = build_dependencies()
handlers = UIHandlers(deps)


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

if __name__ == "__main__":
    demo.launch(allowed_paths=[os.environ.get("VOCHI_ALLOWED_PATH", "D:\\tmp")])
