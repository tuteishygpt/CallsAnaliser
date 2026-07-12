from __future__ import annotations

from types import SimpleNamespace

from calls_analyser.services.tenant_settings import (
    InMemoryTenantSettingsRepository,
    TenantSettingsService,
)


def _batch_params(**overrides):
    values = {
        "enable_gemini_batch": True,
        "batch_size": 25,
        "scheduler_enabled": False,
        "scheduler_mode": "cron",
        "scheduler_cron_time": "02:30",
        "scheduler_interval_minutes": 90,
        "filter_time_from": "09:00",
        "filter_time_to": "18:00",
        "filter_call_type": "answered",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _defaults(**overrides):
    values = {
        "DEFAULT_LANGUAGE": "ru",
        "DEFAULT_MODEL_KEY": "models/default",
        "BATCH_MODEL_KEY": "models/fallback-batch",
        "BATCH_LANGUAGE_CODE": "ru",
        "BATCH_CUSTOM": "off",
        "FOLLOW_UP_VERIFICATION_MODEL_KEY": "models/default-verifier",
        "FOLLOW_UP_VERIFICATION_PROMPT_KEY": "FOLLOW_UP_VERIFICATION_PROMPT",
        "EMAIL_TO": "global-to@example.com",
        "EMAIL_FROM": "global-from@example.com",
        "EMAIL_FROM_NAME": "Global reports",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_follow_up_verification_defaults_to_off_with_configured_model_and_prompt() -> None:
    service = TenantSettingsService(
        InMemoryTenantSettingsRepository(),
        batch_params=_batch_params(),
        defaults=_defaults(FOLLOW_UP_VERIFICATION_MODE="enforce"),
    )

    settings = service.resolve("existing-tenant")

    assert settings.follow_up_verification_mode == "off"
    assert settings.follow_up_verification_model_key == "models/default-verifier"
    assert settings.follow_up_verification_prompt_key == "FOLLOW_UP_VERIFICATION_PROMPT"


def test_tenant_can_override_follow_up_verification_settings() -> None:
    repository = InMemoryTenantSettingsRepository(
        settings={
            "tenant-a": {
                "follow_up_verification_mode": "SHADOW",
                "follow_up_verification_model_key": "models/tenant-verifier",
                "follow_up_verification_prompt_key": "tenant-verification-prompt",
            }
        }
    )
    service = TenantSettingsService(
        repository,
        batch_params=_batch_params(),
        defaults=_defaults(),
    )

    settings = service.resolve("tenant-a")

    assert settings.follow_up_verification_mode == "shadow"
    assert settings.follow_up_verification_model_key == "models/tenant-verifier"
    assert settings.follow_up_verification_prompt_key == "tenant-verification-prompt"


def test_invalid_follow_up_verification_mode_is_safely_disabled() -> None:
    repository = InMemoryTenantSettingsRepository(
        settings={"tenant-a": {"follow_up_verification_mode": "audit"}}
    )
    service = TenantSettingsService(
        repository,
        batch_params=_batch_params(),
        defaults=_defaults(FOLLOW_UP_VERIFICATION_MODE="enforce"),
    )

    assert service.resolve("tenant-a").follow_up_verification_mode == "off"


def test_tenant_setting_overrides_batch_fallbacks() -> None:
    repository = InMemoryTenantSettingsRepository(
        settings={
            "tenant-a": {
                "batch_model_key": "models/tenant-batch",
                "batch_size": "7",
                "scheduler_filters": {
                    "call_type": "missed",
                    "time_from": "10:00",
                },
            },
        }
    )
    service = TenantSettingsService(
        repository,
        batch_params=_batch_params(),
        defaults=_defaults(),
    )

    settings = service.resolve("tenant-a")

    assert settings.batch_model_key == "models/tenant-batch"
    assert settings.batch_size == 7
    assert settings.scheduler_filters == {"call_type": "missed", "time_from": "10:00"}


def test_invalid_numeric_values_fall_back() -> None:
    repository = InMemoryTenantSettingsRepository(
        settings={
            "tenant-a": {
                "batch_size": "not-a-number",
                "scheduler_interval_minutes": -5,
            },
        }
    )
    service = TenantSettingsService(
        repository,
        batch_params=_batch_params(batch_size=33, scheduler_interval_minutes=120),
        defaults=_defaults(),
    )

    settings = service.resolve("tenant-a")

    assert settings.batch_size == 33
    assert settings.scheduler_interval_minutes == 120


def test_email_settings_resolve_from_tenant_settings_and_secrets_before_global_fallback() -> None:
    repository = InMemoryTenantSettingsRepository(
        settings={"tenant-a": {"email_to": "tenant-to@example.com"}},
        secrets={
            "tenant-a": {
                "EMAIL_FROM": "tenant-from@example.com",
                "EMAIL_FROM_NAME": "Tenant reports",
            },
        },
    )
    service = TenantSettingsService(
        repository,
        batch_params=_batch_params(),
        defaults=_defaults(),
    )

    settings = service.resolve("tenant-a")

    assert settings.email_to == "tenant-to@example.com"
    assert settings.email_from == "tenant-from@example.com"
    assert settings.email_from_name == "Tenant reports"


def test_scheduler_enabled_tenant_list_includes_only_resolved_true_tenants() -> None:
    repository = InMemoryTenantSettingsRepository(
        settings={
            "tenant-enabled": {"scheduler_enabled": True},
            "tenant-string-enabled": {"scheduler_enabled": "true"},
            "tenant-disabled": {"scheduler_enabled": False},
            "tenant-string-disabled": {"scheduler_enabled": "no"},
            "tenant-missing": {},
        }
    )
    service = TenantSettingsService(
        repository,
        batch_params=_batch_params(scheduler_enabled=False),
        defaults=_defaults(),
    )

    assert service.list_scheduler_enabled_tenants() == [
        "tenant-enabled",
        "tenant-string-enabled",
    ]


def test_scheduler_enabled_tenant_list_excludes_missing_setting_when_global_enabled() -> None:
    repository = InMemoryTenantSettingsRepository(
        settings={
            "tenant-enabled": {"scheduler_enabled": "yes"},
            "tenant-missing": {},
        }
    )
    service = TenantSettingsService(
        repository,
        batch_params=_batch_params(scheduler_enabled=True),
        defaults=_defaults(),
    )

    assert service.resolve("tenant-missing").scheduler_enabled is True
    assert service.list_scheduler_enabled_tenants() == ["tenant-enabled"]
