"""Prompt management service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Protocol


@dataclass(frozen=True)
class PromptTemplate:
    """Represents a prompt template."""

    key: str
    title: str
    body: str
    version: int = 1


class PromptTemplateRepository(Protocol):
    """Repository contract for tenant prompt templates."""

    def get_template(self, tenant_id: str, key: str) -> PromptTemplate | None:
        """Return the active template for ``tenant_id`` and ``key`` if present."""

    def list_templates(self, tenant_id: str) -> Mapping[str, PromptTemplate]:
        """Return active templates for ``tenant_id`` keyed by prompt key."""


class PromptService:
    """Provides prompt templates and rendering logic."""

    def __init__(
        self,
        templates: Dict[str, PromptTemplate],
        tenant_templates: Mapping[str, Mapping[str, PromptTemplate]] | None = None,
        *,
        prompt_repository: PromptTemplateRepository | None = None,
    ) -> None:
        self._templates = templates
        self._tenant_templates = {
            tenant_id: dict(items)
            for tenant_id, items in (tenant_templates or {}).items()
        }
        self._prompt_repository = prompt_repository

    def get_prompt(
        self,
        key: str,
        fallback_key: str = "simple",
        tenant_id: str | None = None,
    ) -> PromptTemplate:
        """Return a template by key, falling back to ``fallback_key``."""

        if tenant_id:
            repository_template = self._repository_template(tenant_id, key)
            if repository_template is not None:
                return repository_template
            repository_fallback = self._repository_template(tenant_id, fallback_key)
            if repository_fallback is not None:
                return repository_fallback

            tenant_templates = self._tenant_templates.get(tenant_id, {})
            if key in tenant_templates:
                return tenant_templates[key]
            if fallback_key in tenant_templates:
                return tenant_templates[fallback_key]
        return self._templates.get(key, self._templates[fallback_key])

    def list_templates(self, tenant_id: str | None = None) -> Dict[str, PromptTemplate]:
        """Return all templates."""

        templates = dict(self._templates)
        if tenant_id:
            templates.update(self._tenant_templates.get(tenant_id, {}))
            if self._prompt_repository is not None:
                templates.update(self._prompt_repository.list_templates(tenant_id))
        return templates

    def _repository_template(self, tenant_id: str, key: str) -> PromptTemplate | None:
        if self._prompt_repository is None:
            return None
        return self._prompt_repository.get_template(tenant_id, key)
