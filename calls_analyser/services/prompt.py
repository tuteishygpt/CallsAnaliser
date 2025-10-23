"""Prompt management service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PromptTemplate:
    """Represents a prompt template."""

    key: str
    title: str
    body: str


class PromptService:
    """Provides prompt templates and rendering logic."""

    def __init__(self, templates: Dict[str, PromptTemplate]) -> None:
        self._templates = templates

    def get_prompt(self, key: str, fallback_key: str = "simple") -> PromptTemplate:
        """Return a template by key, falling back to ``fallback_key``."""

        return self._templates.get(key, self._templates[fallback_key])

    def list_templates(self) -> Dict[str, PromptTemplate]:
        """Return all templates."""

        return dict(self._templates)
