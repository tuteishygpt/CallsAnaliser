from __future__ import annotations

import base64

import pytest

from calls_analyser.services.tenant_admin_settings import (
    InMemoryTenantAdminRepository,
    TenantAdminSettingsService,
    TenantAdminValidationError,
)
from calls_analyser.services.tenant_secret_codec import TenantSecretCodec, TenantSecretCodecError


def _codec() -> TenantSecretCodec:
    return TenantSecretCodec(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))


def _repository(*, codec: TenantSecretCodec | None = None) -> InMemoryTenantAdminRepository:
    return InMemoryTenantAdminRepository(
        tenants={"t1": {"display_name": "Tenant One", "status": "active"}},
        settings={
            "t1": {
                "telephony_provider": "vochi",
                "vochi_base_url": "https://vochi.test",
                "batch_size": 20,
                "scheduler_interval_minutes": 15,
                "scheduler_filters": {"time_from": "08:00"},
                "custom_json": {"preserve": True},
                "TELEPHONY_PROVIDER": "legacy-provider",
                "VOCHI_BASE_URL": "https://legacy.test",
                "MTS_DOMAIN": "legacy-mts.example",
            }
        },
        secrets={
            "t1": {
                "VOCHI_API_KEY": "legacy-plain",
                "EXTRA": "preserve-secret",
            }
        },
        prompts={
            "t1": [
                {"key": "simple", "title": "Simple", "body": "Body", "version": 1, "is_active": True},
                {"key": "simple", "title": "Old", "body": "Old body", "version": 0, "is_active": False},
            ]
        },
        codec=codec or _codec(),
    )


def test_load_exposes_raw_missing_values_and_only_active_read_only_prompts() -> None:
    service = TenantAdminSettingsService(_repository())

    document = service.load("t1")

    assert document["tenant_id"] == "t1"
    assert document["default_language"] == ""
    assert document["batch_enabled"] == ""
    assert document["scheduler_mode"] == ""
    assert document["active_prompts"] == [
        {"key": "simple", "title": "Simple", "body": "Body", "version": 1}
    ]
    assert "additional_settings" not in document
    assert "prompt_body" not in document


def test_save_uses_focused_changes_and_preserves_arbitrary_rows() -> None:
    repository = _repository()
    service = TenantAdminSettingsService(repository)
    document = service.load("t1")
    document.update(
        display_name="Updated",
        batch_size=25,
        scheduler_interval_minutes=30,
        scheduler_time_from="",
        scheduler_time_to="",
        scheduler_call_type="",
    )

    persisted = service.save("t1", document, "user-1")

    raw = repository.debug_raw("t1")
    assert persisted["display_name"] == "Updated"
    assert raw["settings"]["batch_size"] == 25
    assert "scheduler_filters" not in raw["settings"]
    assert raw["settings"]["custom_json"] == {"preserve": True}
    assert raw["settings"]["MTS_DOMAIN"] == "legacy-mts.example"
    assert raw["secrets"]["EXTRA"] == "preserve-secret"


def test_save_cleans_aliases_for_canonical_typed_fields_without_bulk_rows() -> None:
    repository = _repository()
    service = TenantAdminSettingsService(repository)
    document = service.load("t1")
    document.update(
        telephony_provider="mts_vats",
        mts_domain="mts.example",
        mts_api_key="mts-key",
    )

    service.save("t1", document, "user-1")

    settings = repository.debug_raw("t1")["settings"]
    assert "TELEPHONY_PROVIDER" not in settings
    assert "VOCHI_BASE_URL" not in settings
    assert "MTS_DOMAIN" not in settings
    assert settings["custom_json"] == {"preserve": True}


def test_save_removes_setting_alias_when_equal_canonical_value_needs_no_upsert() -> None:
    repository = InMemoryTenantAdminRepository(
        tenants={"t1": {"display_name": "Tenant", "status": "active"}},
        settings={
            "t1": {
                "telephony_provider": "vochi",
                "TELEPHONY_PROVIDER": "legacy-conflict",
                "vochi_base_url": "https://vochi.test",
                "MTS_DOMAIN": "unrelated-legacy-alias",
            }
        },
        secrets={"t1": {"VOCHI_API_KEY": "vochi-key"}},
        codec=_codec(),
    )
    service = TenantAdminSettingsService(repository)

    service.save("t1", service.load("t1"), "user-1")

    settings = repository.debug_raw("t1")["settings"]
    assert "TELEPHONY_PROVIDER" not in settings
    assert settings["MTS_DOMAIN"] == "unrelated-legacy-alias"


def test_save_removes_secret_setting_alias_when_canonical_secret_is_unchanged() -> None:
    repository = InMemoryTenantAdminRepository(
        tenants={"t1": {"display_name": "Tenant", "status": "active"}},
        settings={
            "t1": {
                "telephony_provider": "vochi",
                "vochi_base_url": "https://vochi.test",
                "VOCHI_API_KEY": "legacy-conflict",
                "MTS_DOMAIN": "unrelated-legacy-alias",
            }
        },
        secrets={"t1": {"VOCHI_API_KEY": "canonical-key"}},
        codec=_codec(),
    )
    service = TenantAdminSettingsService(repository)

    service.save("t1", service.load("t1"), "user-1")

    raw = repository.debug_raw("t1")
    assert "VOCHI_API_KEY" not in raw["settings"]
    assert raw["settings"]["MTS_DOMAIN"] == "unrelated-legacy-alias"
    assert raw["secrets"]["VOCHI_API_KEY"] == "canonical-key"


def test_save_migrates_unchanged_alias_only_secret_to_encrypted_canonical_row() -> None:
    repository = InMemoryTenantAdminRepository(
        tenants={"t1": {"display_name": "Tenant", "status": "active"}},
        settings={
            "t1": {
                "telephony_provider": "mts_vats",
                "MTS_DOMAIN": "mts.example",
                "VOCHI_API_KEY": "unrelated-legacy-alias",
            }
        },
        secrets={"t1": {"MTS_API_KEY": "canonical-mts-key"}},
        codec=_codec(),
    )
    service = TenantAdminSettingsService(repository)

    service.save("t1", service.load("t1"), "user-1")

    raw = repository.debug_raw("t1")
    assert raw["secrets"]["MTS_DOMAIN"].startswith("enc:v1:")
    assert repository.get_secret("t1", "MTS_DOMAIN") == "mts.example"
    assert "MTS_DOMAIN" not in raw["settings"]
    assert raw["settings"]["VOCHI_API_KEY"] == "unrelated-legacy-alias"
    assert raw["secrets"]["MTS_API_KEY"] == "canonical-mts-key"


def test_unchanged_secret_is_not_reencrypted_but_changed_secret_is() -> None:
    repository = _repository()
    service = TenantAdminSettingsService(repository)
    original = repository.debug_raw("t1")["secrets"]["VOCHI_API_KEY"]
    document = service.load("t1")

    service.save("t1", document, "user-1")
    assert repository.debug_raw("t1")["secrets"]["VOCHI_API_KEY"] == original

    document["vochi_api_key"] = "changed-secret"
    service.save("t1", document, "user-1")
    stored = repository.debug_raw("t1")["secrets"]["VOCHI_API_KEY"]
    assert stored.startswith("enc:v1:")
    assert "changed-secret" not in stored


def test_non_secret_save_works_without_master_key() -> None:
    repository = _repository(codec=TenantSecretCodec(None))
    service = TenantAdminSettingsService(repository)
    document = service.load("t1")
    document["display_name"] = "No key required"

    service.save("t1", document, "user-1")

    assert repository.debug_raw("t1")["tenant"]["display_name"] == "No key required"
    assert repository.debug_raw("t1")["secrets"]["VOCHI_API_KEY"] == "legacy-plain"


def test_non_secret_save_preserves_unreadable_encrypted_secret_without_master_key() -> None:
    encrypted = _codec().encrypt("t1", "VOCHI_API_KEY", "hidden-value")
    repository = InMemoryTenantAdminRepository(
        tenants={"t1": {"display_name": "Before", "status": "active"}},
        settings={
            "t1": {
                "telephony_provider": "vochi",
                "vochi_base_url": "https://vochi.test",
            }
        },
        secrets={"t1": {"VOCHI_API_KEY": encrypted}},
        codec=TenantSecretCodec(None),
    )
    service = TenantAdminSettingsService(repository)

    document = service.load("t1")
    assert document["vochi_api_key"] == ""
    assert "hidden-value" not in repr(document)
    assert encrypted not in repr(document)

    document["display_name"] = "Updated"
    persisted = service.save("t1", document, "user-1")

    assert persisted["display_name"] == "Updated"
    assert persisted["vochi_api_key"] == ""
    assert repository.debug_raw("t1")["secrets"]["VOCHI_API_KEY"] == encrypted


def test_admin_aggregate_read_fails_closed_for_malformed_encrypted_secret() -> None:
    repository = InMemoryTenantAdminRepository(
        tenants={"t1": {"display_name": "Tenant", "status": "active"}},
        secrets={"t1": {"VOCHI_API_KEY": "enc:v1:not-valid:also-invalid"}},
        codec=_codec(),
    )

    with pytest.raises(TenantSecretCodecError, match="Tenant secret configuration error"):
        TenantAdminSettingsService(repository).load("t1")


@pytest.mark.parametrize("stored", ["enc:v2:abc:def", "enc:v1:bad", "enc:v1:!!!!:!!!!"])
def test_admin_aggregate_read_rejects_malformed_envelope_without_master_key(
    stored: str,
) -> None:
    repository = InMemoryTenantAdminRepository(
        tenants={"t1": {"display_name": "Tenant", "status": "active"}},
        secrets={"t1": {"VOCHI_API_KEY": stored}},
        codec=TenantSecretCodec(None),
    )

    with pytest.raises(TenantSecretCodecError) as exc:
        TenantAdminSettingsService(repository).load("t1")

    assert type(exc.value) is TenantSecretCodecError


def test_admin_aggregate_read_fails_closed_for_wrong_configured_key() -> None:
    encrypted = _codec().encrypt("t1", "VOCHI_API_KEY", "hidden-value")
    wrong_codec = TenantSecretCodec(
        base64.urlsafe_b64encode(b"w" * 32).decode().rstrip("=")
    )
    repository = InMemoryTenantAdminRepository(
        tenants={"t1": {"display_name": "Tenant", "status": "active"}},
        secrets={"t1": {"VOCHI_API_KEY": encrypted}},
        codec=wrong_codec,
    )

    with pytest.raises(TenantSecretCodecError, match="Tenant secret configuration error"):
        TenantAdminSettingsService(repository).load("t1")


def test_validation_finishes_before_first_repository_write() -> None:
    repository = _repository()
    service = TenantAdminSettingsService(repository)
    before = repository.debug_raw("t1")
    document = service.load("t1")
    document.update(display_name="Would otherwise change", batch_size=0)

    with pytest.raises(TenantAdminValidationError, match="Batch processing / batch size"):
        service.save("t1", document, "user-1")

    assert repository.debug_raw("t1") == before


@pytest.mark.parametrize(
    ("provider", "updates", "field"),
    [
        ("vochi", {"vochi_base_url": ""}, "VoChi base URL"),
        ("vochi", {"vochi_api_key": ""}, "VoChi API key"),
        ("mts_vats", {"mts_domain": ""}, "MTS domain"),
        ("mts_vats", {"mts_api_key": ""}, "MTS API key"),
    ],
)
def test_provider_specific_effective_values_are_required(provider, updates, field) -> None:
    repository = _repository()
    service = TenantAdminSettingsService(repository)
    document = service.load("t1")
    document.update(
        {
            "telephony_provider": provider,
            "mts_domain": "mts.example",
            "mts_api_key": "mts-key",
            **updates,
        }
    )

    with pytest.raises(TenantAdminValidationError, match=field):
        service.save("t1", document, "user-1")


def test_legacy_mts_provider_name_is_rejected() -> None:
    service = TenantAdminSettingsService(_repository())
    document = service.load("t1")
    document["telephony_provider"] = "mts"

    with pytest.raises(TenantAdminValidationError, match="Telephony / provider"):
        service.save("t1", document, "user-1")


@pytest.mark.parametrize("field", ["vochi_api_key", "mts_domain", "mts_api_key"])
def test_secret_inputs_are_trimmed_and_whitespace_is_blank(field: str) -> None:
    repository = _repository()
    service = TenantAdminSettingsService(repository)
    document = service.load("t1")
    document.update(
        telephony_provider="mts_vats" if field.startswith("mts_") else "vochi",
        mts_domain="mts.example",
        mts_api_key="mts-key",
    )
    document[field] = "   "

    expected = {
        "vochi_api_key": "VoChi API key",
        "mts_domain": "MTS domain",
        "mts_api_key": "MTS API key",
    }[field]
    with pytest.raises(TenantAdminValidationError, match=expected):
        service.save("t1", document, "user-1")


def test_changed_secret_is_trimmed_before_encryption() -> None:
    repository = _repository()
    service = TenantAdminSettingsService(repository)
    document = service.load("t1")
    document["vochi_api_key"] = "  trimmed-key  "

    persisted = service.save("t1", document, "user-1")

    assert persisted["vochi_api_key"] == "trimmed-key"
    assert repository.get_secret("t1", "VOCHI_API_KEY") == "trimmed-key"


def test_sequential_failure_can_leave_earlier_changes_persisted() -> None:
    class FailingRepository(InMemoryTenantAdminRepository):
        def upsert_setting(self, tenant_id: str, key: str, value: object) -> None:
            if key == "batch_size":
                raise RuntimeError("network failure")
            super().upsert_setting(tenant_id, key, value)

    repository = FailingRepository(
        tenants={"t1": {"display_name": "Before", "status": "active"}},
        settings={"t1": {"batch_size": 20, "scheduler_interval_minutes": 15}},
    )
    service = TenantAdminSettingsService(repository)
    document = service.load("t1")
    document.update(display_name="Persisted first", batch_size=25)

    with pytest.raises(RuntimeError, match="network failure"):
        service.save("t1", document, "user-1")

    assert service.load("t1")["display_name"] == "Persisted first"
    assert service.load("t1")["batch_size"] == 20
