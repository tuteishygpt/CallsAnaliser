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
    provider: str = "vochi"
    bearer_token: Optional[str] = None
    mts_domain: Optional[str] = None
    mts_api_key: Optional[str] = None

    def recording_url(self, unique_id: str) -> str:
        """Build a provider-specific recording URL for a unique call id."""

        if self.provider == "mts_vats":
            base = self.vochi_base_url.rstrip("/")
            return f"{base}/history/record/{unique_id}"
        return f"{self.vochi_base_url.rstrip('/')}/calllogs/{self.vochi_client_id}/{unique_id}"


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

        provider = (self._secrets.get_optional_secret("TELEPHONY_PROVIDER", tenant_id=tid) or "vochi").strip().lower()

        if provider == "mts_vats":
            domain = self._secrets.get_secret("MTS_DOMAIN", tenant_id=tid)
            api_key = self._secrets.get_secret("MTS_API_KEY", tenant_id=tid)
            base = f"https://{domain.strip().split('/')[0]}/crmapi/v1"
            return TenantConfig(
                provider=provider,
                tenant_id=tid,
                vochi_base_url=base,
                vochi_client_id="",
                bearer_token=None,
                mts_domain=domain,
                mts_api_key=api_key,
            )

        base = self._secrets.get_optional_secret("VOCHI_BASE_URL", tenant_id=tid)
        if not base:
            base = self._secrets.get_optional_secret("VOCHI_BASE_URL", tenant_id=None) or self._default_base_url
        # Normalize base to include '/api' path segment
        base = base.rstrip('/')
        if not base.endswith('/api'):
            # Only append if '/api' is not already present as the last segment
            base = f"{base}/api"
        client_id = self._secrets.get_secret("VOCHI_CLIENT_ID", tenant_id=tid)
        bearer = self._secrets.get_optional_secret("VOCHI_BEARER", tenant_id=tid)
        return TenantConfig(
            provider=provider,
            tenant_id=tid,
            vochi_base_url=base,
            vochi_client_id=client_id,
            bearer_token=bearer,
        )
