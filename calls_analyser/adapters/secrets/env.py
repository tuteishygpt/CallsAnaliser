"""Environment-based secrets adapter."""
from __future__ import annotations

import os
from typing import Optional

from calls_analyser.domain.exceptions import SecretsError
from calls_analyser.ports.secrets import SecretsPort


class EnvSecretsAdapter(SecretsPort):
    """Reads secrets from environment variables with optional tenant prefix."""

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix

    def _build_key(self, key: str, tenant_id: Optional[str]) -> str:
        parts = [self._prefix] if self._prefix else []
        if tenant_id:
            parts.append(tenant_id.upper())
        parts.append(key)
        return "_".join(part for part in parts if part)

    def get_secret(self, key: str, tenant_id: Optional[str] = None) -> str:
        value = self.get_optional_secret(key, tenant_id)
        if value is None:
            raise SecretsError(f"Missing secret for key '{key}' (tenant={tenant_id or 'default'})")
        return value

    def get_optional_secret(self, key: str, tenant_id: Optional[str] = None) -> Optional[str]:
        env_key = self._build_key(key, tenant_id)
        value = os.environ.get(env_key)
        if value:
            return value
        if tenant_id:
            # fallback to global without tenant prefix
            env_key = self._build_key(key, None)
            value = os.environ.get(env_key)
        return value
