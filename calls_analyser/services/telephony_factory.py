"""Tenant-aware telephony provider factory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from calls_analyser.adapters.telephony.mts_vats import MtsVatsTelephonyAdapter
from calls_analyser.adapters.telephony.vochi import VochiTelephonyAdapter
from calls_analyser.domain.exceptions import TelephonyError
from calls_analyser.ports.telephony import TelephonyPort
from calls_analyser.services.tenant import TenantConfig


@dataclass(frozen=True)
class TelephonyProviderDefinition:
    """Definition needed to build one tenant-specific telephony adapter."""

    key: str
    title: str
    required_secrets: tuple[str, ...]
    optional_settings: tuple[str, ...]
    factory: Callable[[TenantConfig], TelephonyPort]


class TelephonyProviderFactory:
    """Builds telephony adapters from tenant runtime configuration."""

    def __init__(
        self,
        definitions: dict[str, TelephonyProviderDefinition],
    ) -> None:
        self._definitions = {
            key.strip().lower(): definition for key, definition in definitions.items()
        }

    def create(self, tenant: TenantConfig) -> TelephonyPort:
        """Create a telephony adapter for ``tenant``."""

        provider_key = tenant.provider.strip().lower()
        try:
            definition = self._definitions[provider_key]
        except KeyError as exc:
            available = ", ".join(sorted(self._definitions)) or "<empty>"
            raise TelephonyError(
                f"Telephony provider '{tenant.provider}' is not registered. "
                f"Available: {available}"
            ) from exc
        return definition.factory(tenant)


def default_telephony_provider_factory() -> TelephonyProviderFactory:
    """Return the built-in provider factory."""

    return TelephonyProviderFactory(
        {
            "vochi": TelephonyProviderDefinition(
                key="vochi",
                title="VoChi",
                required_secrets=("VOCHI_API_KEY",),
                optional_settings=("VOCHI_BASE_URL",),
                factory=lambda tenant: VochiTelephonyAdapter(
                    base_url=tenant.vochi_base_url,
                    api_key=tenant.vochi_api_key or "",
                ),
            ),
            "mts_vats": TelephonyProviderDefinition(
                key="mts_vats",
                title="MTS VATS",
                required_secrets=("MTS_DOMAIN", "MTS_API_KEY"),
                optional_settings=(),
                factory=lambda tenant: MtsVatsTelephonyAdapter(
                    domain=tenant.mts_domain or tenant.vochi_base_url,
                    api_key=tenant.mts_api_key or "",
                ),
            ),
        }
    )
