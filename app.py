"""Gradio UI wired to hexagonal architecture services."""
from __future__ import annotations

import os

from calls_analyser.ui.dependencies import build_dependencies
from calls_analyser.ui.handlers import UIHandlers
from calls_analyser.ui.layout import build_demo


def _build_app():
    deps = build_dependencies()
    handlers = UIHandlers(deps)
    return build_demo(deps, handlers)


demo = _build_app()

if __name__ == "__main__":
    demo.launch(allowed_paths=[os.environ.get("VOCHI_ALLOWED_PATH", "D:\\tmp")])
