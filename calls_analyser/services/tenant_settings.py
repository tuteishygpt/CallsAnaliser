"""Tenant runtime settings resolution."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


_MISSING = object()


@dataclass
class TenantRuntimeSettings:
    """Resolved runtime settings for one tenant."""

    default_language: str = "auto"
    default_model_key: str = ""
    batch_model_key: str = ""
    batch_language_code: str = "auto"
    batch_enabled: bool = True
    batch_size: int = 20
    follow_up_verification_mode: str = "off"
    follow_up_verification_model_key: str = ""
    follow_up_verification_prompt_key: str = ""
    scheduler_enabled: bool = False
    scheduler_mode: str = "cron"
    scheduler_cron_time: str = "01:00"
    scheduler_interval_minutes: int = 120
    scheduler_filters: dict[str, Any] = field(default_factory=dict)
    custom_batch_enabled: bool = False
    email_to: str = ""
    email_from: str = ""
    email_from_name: str = ""


class TenantSettingsRepository(Protocol):
    """Repository contract for tenant settings and secrets lookup."""

    def get_setting(self, tenant_id: str, key: str) -> Any | None:
        """Return a tenant setting value, or ``None`` when it is not present."""

    def get_secret(self, tenant_id: str, key: str) -> Any | None:
        """Return a tenant secret value, or ``None`` when it is not present."""

    def list_tenant_ids(self) -> Iterable[str]:
        """Return tenant ids known to the repository."""


class InMemoryTenantSettingsRepository:
    """In-memory tenant settings repository for tests and local composition."""

    def __init__(
        self,
        *,
        settings: Mapping[str, Mapping[str, Any]] | None = None,
        secrets: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._settings = {
            tenant_id: dict(values)
            for tenant_id, values in (settings or {}).items()
        }
        self._secrets = {
            tenant_id: dict(values)
            for tenant_id, values in (secrets or {}).items()
        }

    def get_setting(self, tenant_id: str, key: str) -> Any | None:
        return self._settings.get(tenant_id, {}).get(key)

    def get_secret(self, tenant_id: str, key: str) -> Any | None:
        return self._secrets.get(tenant_id, {}).get(key)

    def list_tenant_ids(self) -> Iterable[str]:
        return dict.fromkeys([*self._settings.keys(), *self._secrets.keys()]).keys()


class TenantSettingsService:
    """Resolve tenant settings with tenant values first, then global fallbacks."""

    def __init__(
        self,
        repository: TenantSettingsRepository,
        *,
        batch_params: Any | None = None,
        defaults: Any | None = None,
    ) -> None:
        self._repository = repository
        self._batch_params = batch_params
        self._defaults = defaults

    def resolve(self, tenant_id: str) -> TenantRuntimeSettings:
        """Return runtime settings for ``tenant_id``."""

        if not tenant_id:
            raise ValueError("Tenant id is required")

        return TenantRuntimeSettings(
            default_language=self._resolve_str(
                tenant_id,
                "default_language",
                fallback_sources=[self._defaults],
                fallback_names=("default_language", "DEFAULT_LANGUAGE"),
                hard_default="auto",
            ),
            default_model_key=self._resolve_str(
                tenant_id,
                "default_model_key",
                fallback_sources=[self._defaults],
                fallback_names=("default_model_key", "DEFAULT_MODEL_KEY", "model_default"),
                hard_default="",
            ),
            batch_model_key=self._resolve_str(
                tenant_id,
                "batch_model_key",
                fallback_sources=[self._defaults],
                fallback_names=("batch_model_key", "BATCH_MODEL_KEY", "model_default"),
                hard_default="",
            ),
            batch_language_code=self._resolve_str(
                tenant_id,
                "batch_language_code",
                fallback_sources=[self._defaults],
                fallback_names=("batch_language_code", "BATCH_LANGUAGE_CODE", "batch_language"),
                hard_default="auto",
            ),
            batch_enabled=self._resolve_bool(
                tenant_id,
                "batch_enabled",
                fallback_sources=[self._batch_params, self._defaults],
                fallback_names=("enable_gemini_batch", "batch_enabled", "BATCH_ENABLED"),
                hard_default=True,
            ),
            batch_size=self._resolve_int(
                tenant_id,
                "batch_size",
                fallback_sources=[self._batch_params, self._defaults],
                fallback_names=("batch_size", "BATCH_SIZE"),
                hard_default=20,
                min_value=1,
            ),
            follow_up_verification_mode=self._resolve_choice(
                tenant_id,
                "follow_up_verification_mode",
                fallback_sources=[],
                fallback_names=(),
                hard_default="off",
                allowed={"off", "shadow", "enforce"},
            ),
            follow_up_verification_model_key=self._resolve_str(
                tenant_id,
                "follow_up_verification_model_key",
                fallback_sources=[self._defaults],
                fallback_names=(
                    "follow_up_verification_model_key",
                    "FOLLOW_UP_VERIFICATION_MODEL_KEY",
                ),
                hard_default="",
            ),
            follow_up_verification_prompt_key=self._resolve_str(
                tenant_id,
                "follow_up_verification_prompt_key",
                fallback_sources=[self._defaults],
                fallback_names=(
                    "follow_up_verification_prompt_key",
                    "FOLLOW_UP_VERIFICATION_PROMPT_KEY",
                ),
                hard_default="",
            ),
            scheduler_enabled=self._resolve_bool(
                tenant_id,
                "scheduler_enabled",
                fallback_sources=[self._batch_params, self._defaults],
                fallback_names=("scheduler_enabled", "SCHEDULER_ENABLED"),
                hard_default=False,
            ),
            scheduler_mode=self._resolve_choice(
                tenant_id,
                "scheduler_mode",
                fallback_sources=[self._batch_params, self._defaults],
                fallback_names=("scheduler_mode", "SCHEDULER_MODE"),
                hard_default="cron",
                allowed={"cron", "interval"},
            ),
            scheduler_cron_time=self._resolve_str(
                tenant_id,
                "scheduler_cron_time",
                fallback_sources=[self._batch_params, self._defaults],
                fallback_names=("scheduler_cron_time", "SCHEDULER_CRON_TIME"),
                hard_default="01:00",
            ),
            scheduler_interval_minutes=self._resolve_int(
                tenant_id,
                "scheduler_interval_minutes",
                fallback_sources=[self._batch_params, self._defaults],
                fallback_names=("scheduler_interval_minutes", "SCHEDULER_INTERVAL_MINUTES"),
                hard_default=120,
                min_value=1,
            ),
            scheduler_filters=self._resolve_scheduler_filters(tenant_id),
            custom_batch_enabled=self._resolve_bool(
                tenant_id,
                "custom_batch_enabled",
                fallback_sources=[self._defaults],
                fallback_names=("custom_batch_enabled", "BATCH_CUSTOM_ENABLED", "BATCH_custom", "BATCH_CUSTOM"),
                hard_default=False,
            ),
            email_to=self._resolve_email_str(tenant_id, "email_to", "EMAIL_TO"),
            email_from=self._resolve_email_str(tenant_id, "email_from", "EMAIL_FROM"),
            email_from_name=self._resolve_email_str(tenant_id, "email_from_name", "EMAIL_FROM_NAME"),
        )

    def list_scheduler_enabled_tenants(self) -> list[str]:
        """Return tenant ids with an explicit scheduler opt-in setting."""

        return [
            tenant_id
            for tenant_id in self._repository.list_tenant_ids()
            if self._coerce_bool(
                self._setting_value(tenant_id, "scheduler_enabled"),
                False,
            )
        ]

    def _resolve_str(
        self,
        tenant_id: str,
        setting_key: str,
        *,
        fallback_sources: Iterable[Any],
        fallback_names: tuple[str, ...],
        hard_default: str,
    ) -> str:
        value = self._setting_value(tenant_id, setting_key)
        if value is _MISSING:
            value = self._first_fallback(fallback_sources, fallback_names)
        return self._coerce_str(value, hard_default)

    def _resolve_email_str(self, tenant_id: str, setting_key: str, secret_key: str) -> str:
        value = self._setting_value(tenant_id, setting_key)
        if value is _MISSING:
            value = self._secret_value(tenant_id, secret_key)
        if value is _MISSING:
            value = self._first_fallback(
                [self._defaults],
                (setting_key, secret_key),
            )
        return self._coerce_str(value, "")

    def _resolve_bool(
        self,
        tenant_id: str,
        setting_key: str,
        *,
        fallback_sources: Iterable[Any],
        fallback_names: tuple[str, ...],
        hard_default: bool,
    ) -> bool:
        value = self._setting_value(tenant_id, setting_key)
        fallback = self._coerce_bool(
            self._first_fallback(fallback_sources, fallback_names),
            hard_default,
        )
        return self._coerce_bool(value, fallback)

    def _resolve_int(
        self,
        tenant_id: str,
        setting_key: str,
        *,
        fallback_sources: Iterable[Any],
        fallback_names: tuple[str, ...],
        hard_default: int,
        min_value: int,
    ) -> int:
        value = self._setting_value(tenant_id, setting_key)
        fallback = self._coerce_int(
            self._first_fallback(fallback_sources, fallback_names),
            hard_default,
            min_value=min_value,
        )
        return self._coerce_int(value, fallback, min_value=min_value)

    def _resolve_choice(
        self,
        tenant_id: str,
        setting_key: str,
        *,
        fallback_sources: Iterable[Any],
        fallback_names: tuple[str, ...],
        hard_default: str,
        allowed: set[str],
    ) -> str:
        value = self._setting_value(tenant_id, setting_key)
        fallback = self._coerce_choice(
            self._first_fallback(fallback_sources, fallback_names),
            hard_default,
            allowed=allowed,
        )
        return self._coerce_choice(value, fallback, allowed=allowed)

    def _resolve_scheduler_filters(self, tenant_id: str) -> dict[str, Any]:
        value = self._setting_value(tenant_id, "scheduler_filters")
        if value is not _MISSING:
            if isinstance(value, Mapping):
                return dict(value)
            return self._fallback_scheduler_filters()
        return self._fallback_scheduler_filters()

    def _fallback_scheduler_filters(self) -> dict[str, Any]:
        value = self._first_fallback([self._defaults], ("scheduler_filters", "SCHEDULER_FILTERS"))
        if isinstance(value, Mapping):
            return dict(value)

        filters: dict[str, Any] = {}
        for source_name, filter_name in (
            ("filter_time_from", "time_from"),
            ("filter_time_to", "time_to"),
            ("filter_call_type", "call_type"),
        ):
            filter_value = self._lookup(self._batch_params, (source_name,))
            if filter_value is not _MISSING and self._has_value(filter_value):
                filters[filter_name] = filter_value
        return filters

    def _setting_value(self, tenant_id: str, key: str) -> Any:
        value = self._repository.get_setting(tenant_id, key)
        if self._has_value(value):
            return value
        return _MISSING

    def _secret_value(self, tenant_id: str, key: str) -> Any:
        value = self._repository.get_secret(tenant_id, key)
        if self._has_value(value):
            return value
        return _MISSING

    def _first_fallback(self, sources: Iterable[Any], names: tuple[str, ...]) -> Any:
        for source in sources:
            value = self._lookup(source, names)
            if value is not _MISSING and self._has_value(value):
                return value
        return _MISSING

    @classmethod
    def _lookup(cls, source: Any, names: tuple[str, ...]) -> Any:
        if source is None:
            return _MISSING
        for name in names:
            if isinstance(source, Mapping) and name in source:
                return cls._unwrap_value(source[name])
            if hasattr(source, name):
                return cls._unwrap_value(getattr(source, name))
        return _MISSING

    @staticmethod
    def _unwrap_value(value: Any) -> Any:
        if isinstance(value, str):
            return value
        enum_value = getattr(value, "value", _MISSING)
        if enum_value is not _MISSING:
            return enum_value
        return value

    @staticmethod
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    @classmethod
    def _coerce_str(cls, value: Any, fallback: str) -> str:
        if value is _MISSING or value is None:
            return fallback
        value = cls._unwrap_value(value)
        text = str(value).strip()
        return text if text else fallback

    @staticmethod
    def _coerce_bool(value: Any, fallback: bool) -> bool:
        if value is _MISSING or value is None:
            return fallback
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value == 1:
                return True
            if value == 0:
                return False
            return fallback
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
        return fallback

    @staticmethod
    def _coerce_int(value: Any, fallback: int, *, min_value: int) -> int:
        if value is _MISSING or value is None or isinstance(value, bool):
            return fallback
        try:
            if isinstance(value, float):
                if not value.is_integer():
                    return fallback
                parsed = int(value)
            else:
                parsed = int(str(value).strip())
        except (TypeError, ValueError):
            return fallback
        return parsed if parsed >= min_value else fallback

    @classmethod
    def _coerce_choice(cls, value: Any, fallback: str, *, allowed: set[str]) -> str:
        text = cls._coerce_str(value, fallback).lower()
        return text if text in allowed else fallback
