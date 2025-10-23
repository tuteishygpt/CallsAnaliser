"""Tenant configuration service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from calls_analyser.domain.exceptions import SecretsError
from calls_analyser.ports.secrets import SecretsPort


@dataclass
class TenantConfig:
    """Configuration for a tenant."""

    tenant_id: str
    vochi_base_url: str
    vochi_client_id: str
    bearer_token: Optional[str] = None


class TenantService:
    """Resolves tenant configuration using the secrets port."""

    def __init__(self, secrets: SecretsPort, default_tenant: str, default_base_url: str = "https://crm.vochi.by/api") -> None:
        self._secrets = secrets
        self._default_tenant = default_tenant
        self._default_base_url = default_base_url

    def resolve(self, tenant_id: Optional[str] = None) -> TenantConfig:
        """Return configuration for ``tenant_id`` or the default tenant."""

        tid = tenant_id or self._default_tenant
        if not tid:
            raise SecretsError("Tenant id is required")
        base = self._secrets.get_optional_secret("VOCHI_BASE_URL", tenant_id=tid)
        if not base:
            base = self._secrets.get_optional_secret("VOCHI_BASE_URL", tenant_id=None) or self._default_base_url
        client_id = self._secrets.get_secret("VOCHI_CLIENT_ID", tenant_id=tid)
        bearer = self._secrets.get_optional_secret("VOCHI_BEARER", tenant_id=tid)
        return TenantConfig(tenant_id=tid, vochi_base_url=base, vochi_client_id=client_id, bearer_token=bearer)
