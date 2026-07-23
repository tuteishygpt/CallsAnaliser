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


class FakeTenantSettingsRepository:
    def __init__(
        self,
        settings: dict[tuple[str, str], str | None],
        secrets: dict[tuple[str, str], str | None],
    ) -> None:
        self._settings = settings
        self._secrets = secrets
        self.calls: list[tuple[str, str, str]] = []

    def get_setting(self, tenant_id: str, key: str) -> str | None:
        self.calls.append(("get_setting", tenant_id, key))
        return self._settings.get((tenant_id, key))

    def get_secret(self, tenant_id: str, key: str) -> str | None:
        self.calls.append(("get_secret", tenant_id, key))
        return self._secrets.get((tenant_id, key))

    def list_tenant_ids(self) -> list[str]:
        return []


def test_resolve_vochi_api_v1_configuration() -> None:
    secrets = FakeSecrets(
        {
            ("tenant-a", "VOCHI_BASE_URL"): "https://bot.example.com/api/v1/",
            ("tenant-a", "VOCHI_API_KEY"): "secret-api-key",
        }
    )
    service = TenantService(secrets, default_tenant="tenant-a")

    config = service.resolve("tenant-a")

    assert isinstance(config, TenantConfig)
    assert config.vochi_base_url == "https://bot.example.com/api/v1"
    assert config.vochi_api_key == "secret-api-key"
    assert config.recording_url("uid-1") == "https://bot.example.com/api/v1/recording/uid-1"


def test_resolve_falls_back_to_global_base_url_and_trims_trailing_slash() -> None:
    secrets = FakeSecrets(
        {
            (None, "VOCHI_BASE_URL"): "https://global.example.com/api/v1/",
            ("tenant-b", "VOCHI_API_KEY"): "tenant-key",
        }
    )
    service = TenantService(secrets, default_tenant="tenant-a", default_base_url="https://default.example/api/v1")

    config = service.resolve("tenant-b")

    assert config.vochi_base_url == "https://global.example.com/api/v1"
    assert config.vochi_api_key == "tenant-key"


def test_resolve_uses_default_base_url_when_no_secret_present() -> None:
    secrets = FakeSecrets({("tenant-c", "VOCHI_API_KEY"): "tenant-key"})
    service = TenantService(secrets, default_tenant="tenant-c", default_base_url="https://fallback.example/api/v1/")

    config = service.resolve()

    assert config.vochi_base_url == "https://fallback.example/api/v1"
    assert config.tenant_id == "tenant-c"


def test_resolve_requires_vochi_api_key() -> None:
    service = TenantService(FakeSecrets({}), default_tenant="tenant-a")

    with pytest.raises(SecretsError, match="VOCHI_API_KEY"):
        service.resolve()


def test_resolve_requires_tenant_id_when_default_missing() -> None:
    secrets = FakeSecrets({})
    service = TenantService(secrets, default_tenant="")

    with pytest.raises(SecretsError, match="Tenant id is required"):
        service.resolve()


def test_resolve_mts_vats_tenant_configuration() -> None:
    secrets = FakeSecrets(
        {
            ("tenant-mts", "TELEPHONY_PROVIDER"): "mts_vats",
            ("tenant-mts", "MTS_DOMAIN"): "193130978.vats.mts.by",
            ("tenant-mts", "MTS_API_KEY"): "secret-key",
        }
    )
    service = TenantService(secrets, default_tenant="tenant-mts")

    config = service.resolve()

    assert config.provider == "mts_vats"
    assert config.vochi_base_url == "https://193130978.vats.mts.by/crmapi/v1"
    assert config.mts_api_key == "secret-key"
    assert config.recording_url("uid-1") == "https://193130978.vats.mts.by/crmapi/v1/history/record/uid-1"


def test_resolve_prefers_tenant_repository_over_env_fallback() -> None:
    tenant_source = FakeTenantSettingsRepository(
        settings={
            ("tenant-a", "telephony_provider"): "vochi",
            ("tenant-a", "vochi_base_url"): "https://tenant.example.com/api/v1/",
        },
        secrets={
            ("tenant-a", "VOCHI_API_KEY"): "tenant-vochi-key",
            ("tenant-a", "MTS_DOMAIN"): "tenant.mts.example",
            ("tenant-a", "MTS_API_KEY"): "tenant-mts-key",
        },
    )
    env_secrets = FakeSecrets(
        {
            ("tenant-a", "TELEPHONY_PROVIDER"): "vochi",
            ("tenant-a", "VOCHI_BASE_URL"): "https://env.example.com/api/v1/",
            ("tenant-a", "VOCHI_API_KEY"): "env-vochi-key",
            ("tenant-a", "MTS_DOMAIN"): "env.mts.example",
            ("tenant-a", "MTS_API_KEY"): "env-mts-key",
        }
    )
    service = TenantService(
        env_secrets,
        default_tenant="tenant-a",
        tenant_settings_source=tenant_source,
        default_base_url="https://global.example.com/api/v1/",
    )

    config = service.resolve("tenant-a")

    assert config.provider == "vochi"
    assert config.vochi_base_url == "https://tenant.example.com/api/v1"
    assert config.vochi_api_key == "tenant-vochi-key"
    assert config.recording_url("uid-1") == "https://tenant.example.com/api/v1/recording/uid-1"


def test_resolve_prefers_tenant_repository_for_mts_vats_configuration() -> None:
    tenant_source = FakeTenantSettingsRepository(
        settings={
            ("tenant-mts", "telephony_provider"): "mts_vats",
        },
        secrets={
            ("tenant-mts", "MTS_DOMAIN"): "tenant.mts.example",
            ("tenant-mts", "MTS_API_KEY"): "tenant-mts-key",
        },
    )
    env_secrets = FakeSecrets(
        {
            ("tenant-mts", "MTS_DOMAIN"): "env.mts.example",
            ("tenant-mts", "MTS_API_KEY"): "env-mts-key",
        }
    )
    service = TenantService(
        env_secrets,
        default_tenant="tenant-mts",
        tenant_settings_source=tenant_source,
    )

    config = service.resolve()

    assert config.provider == "mts_vats"
    assert config.mts_domain == "tenant.mts.example"
    assert config.mts_api_key == "tenant-mts-key"
    assert config.vochi_base_url == "https://tenant.mts.example/crmapi/v1"
    assert config.recording_url("uid-1") == "https://tenant.mts.example/crmapi/v1/history/record/uid-1"
