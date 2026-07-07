"""Tenant configuration service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from calls_analyser.domain.exceptions import SecretsError
from calls_analyser.ports.secrets import SecretsPort
from calls_analyser.services.tenant_settings import TenantSettingsRepository


@dataclass
class TenantConfig:
    """Configuration for a tenant."""

    tenant_id: str
    vochi_base_url: str
    provider: str = "vochi"
    mts_domain: Optional[str] = None
    mts_api_key: Optional[str] = None
    vochi_api_key: Optional[str] = None

    def recording_url(self, unique_id: str) -> str:
        """Build a provider-specific recording URL for a unique call id."""

        if self.provider == "mts_vats":
            base = self.vochi_base_url.rstrip("/")
            return f"{base}/history/record/{unique_id}"
        return f"{self.vochi_base_url.rstrip('/')}/recording/{unique_id}"


class TenantService:
    """Resolves tenant configuration using the secrets port."""

    def __init__(
        self,
        secrets: SecretsPort,
        default_tenant: str,
        default_base_url: str = "https://bot.vochi.by/api/v1",
        tenant_settings_source: TenantSettingsRepository | None = None,
    ) -> None:
        self._secrets = secrets
        self._default_tenant = default_tenant
        self._default_base_url = default_base_url
        self._tenant_settings_source = tenant_settings_source

    def resolve(self, tenant_id: Optional[str] = None) -> TenantConfig:
        """Return configuration for ``tenant_id`` or the default tenant."""

        tid = tenant_id or self._default_tenant
        if not tid:
            raise SecretsError("Tenant id is required")

        provider = self._resolve_tenant_value(
            tid,
            setting_keys=("telephony_provider", "TELEPHONY_PROVIDER"),
            secret_keys=("TELEPHONY_PROVIDER",),
        )
        if not provider:
            provider = self._secrets.get_optional_secret("TELEPHONY_PROVIDER", tenant_id=tid) or "vochi"
        provider = provider.strip().lower()

        if provider == "mts_vats":
            domain = self._resolve_tenant_value(
                tid,
                setting_keys=("mts_domain", "MTS_DOMAIN"),
                secret_keys=("MTS_DOMAIN",),
            )
            if not domain:
                domain = self._secrets.get_secret("MTS_DOMAIN", tenant_id=tid)
            api_key = self._resolve_tenant_value(
                tid,
                setting_keys=("mts_api_key", "MTS_API_KEY"),
                secret_keys=("MTS_API_KEY",),
            )
            if not api_key:
                api_key = self._secrets.get_secret("MTS_API_KEY", tenant_id=tid)
            base = f"https://{domain.strip().split('/')[0]}/crmapi/v1"
            return TenantConfig(
                provider=provider,
                tenant_id=tid,
                vochi_base_url=base,
                mts_domain=domain,
                mts_api_key=api_key,
            )

        base = self._resolve_tenant_value(
            tid,
            setting_keys=("vochi_base_url", "VOCHI_BASE_URL"),
            secret_keys=("VOCHI_BASE_URL",),
        )
        if not base:
            base = self._secrets.get_optional_secret("VOCHI_BASE_URL", tenant_id=tid)
        if not base:
            base = self._secrets.get_optional_secret("VOCHI_BASE_URL", tenant_id=None) or self._default_base_url
        base = base.rstrip('/')
        api_key = self._resolve_tenant_value(
            tid,
            setting_keys=("vochi_api_key", "VOCHI_API_KEY"),
            secret_keys=("VOCHI_API_KEY",),
        )
        if not api_key:
            api_key = self._secrets.get_secret("VOCHI_API_KEY", tenant_id=tid)
        return TenantConfig(
            provider=provider,
            tenant_id=tid,
            vochi_base_url=base,
            vochi_api_key=api_key,
        )

    def _resolve_tenant_value(
        self,
        tenant_id: str,
        *,
        setting_keys: tuple[str, ...],
        secret_keys: tuple[str, ...],
    ) -> Optional[str]:
        source = self._tenant_settings_source
        if source is None:
            return None

        for key in setting_keys:
            value = source.get_setting(tenant_id, key)
            if self._has_value(value):
                return str(value).strip()

        for key in secret_keys:
            value = source.get_secret(tenant_id, key)
            if self._has_value(value):
                return str(value).strip()

        return None

    @staticmethod
    def _has_value(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True
