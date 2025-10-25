from __future__ import annotations

import pytest

from calls_analyser.domain.exceptions import SecretsError
from calls_analyser.ports.secrets import SecretsPort
from calls_analyser.services.tenant import TenantService, TenantConfig


class FakeSecrets(SecretsPort):
    def __init__(self, values: dict[tuple[str | None, str], str | None]) -> None:
        self._values = values
        self.calls: list[tuple[str, str, str | None]] = []

    def get_secret(self, key: str, tenant_id: str | None = None) -> str:
        self.calls.append(("get", key, tenant_id))
        try:
            value = self._values[(tenant_id, key)]
        except KeyError as exc:
            raise SecretsError(f"Missing secret {key!r} for tenant {tenant_id!r}") from exc
        if value is None:
            raise SecretsError(f"Secret {key!r} for tenant {tenant_id!r} is not set")
        return value

    def get_optional_secret(self, key: str, tenant_id: str | None = None) -> str | None:
        self.calls.append(("get_optional", key, tenant_id))
        return self._values.get((tenant_id, key))


def test_resolve_appends_api_once_for_tenant_specific_base_url() -> None:
    secrets = FakeSecrets(
        {
            ("tenant-a", "VOCHI_BASE_URL"): "https://crm.example.com",  # missing trailing /api
            ("tenant-a", "VOCHI_CLIENT_ID"): "client-123",
            ("tenant-a", "VOCHI_BEARER"): "token-abc",
        }
    )
    service = TenantService(secrets, default_tenant="tenant-a")

    config = service.resolve("tenant-a")

    assert isinstance(config, TenantConfig)
    assert config.vochi_base_url == "https://crm.example.com/api"
    assert config.vochi_client_id == "client-123"
    assert config.bearer_token == "token-abc"


def test_resolve_falls_back_to_global_base_url_and_trims_trailing_slash() -> None:
    secrets = FakeSecrets(
        {
            (None, "VOCHI_BASE_URL"): "https://global.example.com/",
            ("tenant-b", "VOCHI_CLIENT_ID"): "client-xyz",
            ("tenant-b", "VOCHI_BEARER"): None,
        }
    )
    service = TenantService(secrets, default_tenant="tenant-a", default_base_url="https://default.example/api")

    config = service.resolve("tenant-b")

    assert config.vochi_base_url == "https://global.example.com/api"
    assert config.bearer_token is None


def test_resolve_uses_default_base_url_when_no_secret_present() -> None:
    secrets = FakeSecrets({("tenant-c", "VOCHI_CLIENT_ID"): "client-id"})
    service = TenantService(secrets, default_tenant="tenant-c", default_base_url="https://fallback.example/api")

    config = service.resolve()

    assert config.vochi_base_url == "https://fallback.example/api"
    assert config.tenant_id == "tenant-c"


def test_resolve_requires_tenant_id_when_default_missing() -> None:
    secrets = FakeSecrets({})
    service = TenantService(secrets, default_tenant="")

    with pytest.raises(SecretsError, match="Tenant id is required"):
        service.resolve()
