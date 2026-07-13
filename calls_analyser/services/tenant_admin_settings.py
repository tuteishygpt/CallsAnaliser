"""Typed tenant administration over focused repository operations."""
from __future__ import annotations

import copy
import re
import threading
from collections.abc import Mapping
from typing import Any, Protocol

from .prompt import PromptTemplate
from .tenant_secret_codec import (
    UNREADABLE_SECRET,
    TenantSecretCodec,
    TenantSecretKeyUnavailableError,
)


SETTING_FIELDS = {
    "telephony_provider": "telephony_provider",
    "vochi_base_url": "vochi_base_url",
    "default_language": "default_language",
    "default_model_key": "default_model_key",
    "batch_language_code": "batch_language_code",
    "batch_model_key": "batch_model_key",
    "batch_enabled": "batch_enabled",
    "batch_size": "batch_size",
    "custom_batch_enabled": "custom_batch_enabled",
    "scheduler_enabled": "scheduler_enabled",
    "scheduler_mode": "scheduler_mode",
    "scheduler_cron_time": "scheduler_cron_time",
    "scheduler_interval_minutes": "scheduler_interval_minutes",
    "email_to": "email_to",
    "email_from": "email_from",
    "email_from_name": "email_from_name",
}
SECRET_FIELDS = {
    "vochi_api_key": "VOCHI_API_KEY",
    "mts_domain": "MTS_DOMAIN",
    "mts_api_key": "MTS_API_KEY",
}
SETTING_ALIASES = {
    "telephony_provider": ("TELEPHONY_PROVIDER",),
    "vochi_base_url": ("VOCHI_BASE_URL",),
}
SECRET_SETTING_ALIASES = {
    "vochi_api_key": ("VOCHI_API_KEY",),
    "mts_domain": ("MTS_DOMAIN",),
    "mts_api_key": ("MTS_API_KEY",),
}
BOOLEAN_FIELDS = {"batch_enabled", "custom_batch_enabled", "scheduler_enabled"}
INTEGER_FIELDS = {"batch_size", "scheduler_interval_minutes"}
TIME_FIELDS = {"scheduler_cron_time", "scheduler_time_from", "scheduler_time_to"}
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


class TenantAdminValidationError(ValueError):
    """Editable tenant document validation failed before persistence."""


class TenantAdminRepository(Protocol):
    def read_raw_document(self, tenant_id: str) -> Mapping[str, Any]: ...

    def update_tenant_profile(self, tenant_id: str, display_name: str, status: str) -> None: ...

    def upsert_setting(self, tenant_id: str, key: str, value: Any) -> None: ...

    def delete_setting(self, tenant_id: str, key: str) -> None: ...

    def upsert_secret(self, tenant_id: str, key: str, plaintext: str) -> None: ...

    def delete_secret(self, tenant_id: str, key: str) -> None: ...


class InMemoryTenantAdminRepository:
    """Lock-protected tenant configuration store shared by local services."""

    def __init__(
        self,
        *,
        tenants: Mapping[str, Mapping[str, Any]] | None = None,
        settings: Mapping[str, Mapping[str, Any]] | None = None,
        secrets: Mapping[str, Mapping[str, str]] | None = None,
        prompts: Mapping[str, list[Mapping[str, Any]]] | None = None,
        codec: TenantSecretCodec | None = None,
    ) -> None:
        self._tenants = copy.deepcopy(dict(tenants or {}))
        self._settings = copy.deepcopy(dict(settings or {}))
        self._secrets = copy.deepcopy(dict(secrets or {}))
        self._prompts = copy.deepcopy(dict(prompts or {}))
        self._codec = codec or TenantSecretCodec(None)
        self._lock = threading.RLock()

    def get_setting(self, tenant_id: str, key: str) -> Any | None:
        with self._lock:
            return copy.deepcopy(self._settings.get(tenant_id, {}).get(key))

    def get_secret(self, tenant_id: str, key: str) -> str | None:
        with self._lock:
            value = self._secrets.get(tenant_id, {}).get(key)
            return None if value is None else self._codec.decrypt(tenant_id, key, value)

    def list_tenant_ids(self):
        with self._lock:
            return tuple(dict.fromkeys([*self._tenants, *self._settings, *self._secrets]))

    def get_template(self, tenant_id: str, key: str) -> PromptTemplate | None:
        with self._lock:
            rows = [
                row for row in self._prompts.get(tenant_id, [])
                if row.get("key") == key and row.get("is_active")
            ]
            if not rows:
                return None
            row = max(rows, key=lambda item: int(item.get("version", 1)))
            return PromptTemplate(key, str(row["title"]), str(row["body"]), int(row["version"]))

    def list_templates(self, tenant_id: str):
        with self._lock:
            keys = {
                str(row.get("key", ""))
                for row in self._prompts.get(tenant_id, [])
                if row.get("is_active")
            }
            return {
                key: template
                for key in sorted(keys)
                if (template := self.get_template(tenant_id, key)) is not None
            }

    def read_tenant_profile(self, tenant_id: str) -> Mapping[str, Any]:
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError("Tenant not found")
            return copy.deepcopy(self._tenants[tenant_id])

    def read_settings(self, tenant_id: str) -> Mapping[str, Any]:
        with self._lock:
            return copy.deepcopy(self._settings.get(tenant_id, {}))

    def read_secrets(self, tenant_id: str) -> Mapping[str, Any]:
        with self._lock:
            result: dict[str, Any] = {}
            for key, value in self._secrets.get(tenant_id, {}).items():
                try:
                    result[key] = self._codec.decrypt(tenant_id, key, value)
                except TenantSecretKeyUnavailableError:
                    result[key] = UNREADABLE_SECRET
            return result

    def read_active_prompts(self, tenant_id: str) -> list[Mapping[str, Any]]:
        with self._lock:
            return copy.deepcopy(
                [row for row in self._prompts.get(tenant_id, []) if row.get("is_active")]
            )

    def read_raw_document(self, tenant_id: str) -> Mapping[str, Any]:
        return {
            "tenant": self.read_tenant_profile(tenant_id),
            "settings": self.read_settings(tenant_id),
            "secrets": self.read_secrets(tenant_id),
            "prompts": self.read_active_prompts(tenant_id),
        }

    def update_tenant_profile(self, tenant_id: str, display_name: str, status: str) -> None:
        with self._lock:
            if tenant_id not in self._tenants:
                raise KeyError("Tenant not found")
            self._tenants[tenant_id].update(display_name=display_name, status=status)

    def upsert_setting(self, tenant_id: str, key: str, value: Any) -> None:
        with self._lock:
            self._settings.setdefault(tenant_id, {})[key] = copy.deepcopy(value)

    def delete_setting(self, tenant_id: str, key: str) -> None:
        with self._lock:
            self._settings.get(tenant_id, {}).pop(key, None)

    def upsert_secret(self, tenant_id: str, key: str, plaintext: str) -> None:
        encrypted = self._codec.encrypt(tenant_id, key, plaintext)
        with self._lock:
            self._secrets.setdefault(tenant_id, {})[key] = encrypted

    def delete_secret(self, tenant_id: str, key: str) -> None:
        with self._lock:
            self._secrets.get(tenant_id, {}).pop(key, None)

    def debug_raw(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(
                {
                    "tenant": self._tenants[tenant_id],
                    "settings": self._settings.get(tenant_id, {}),
                    "secrets": self._secrets.get(tenant_id, {}),
                    "prompts": self._prompts.get(tenant_id, []),
                }
            )


class TenantAdminSettingsService:
    """Validate the complete typed form and orchestrate focused writes."""

    def __init__(self, repository: TenantAdminRepository) -> None:
        self._repository = repository

    def load(self, tenant_id: str) -> dict[str, Any]:
        return self._editable(tenant_id, self._repository.read_raw_document(tenant_id))

    def save(
        self, tenant_id: str, document: Mapping[str, Any], actor_user_id: str
    ) -> dict[str, Any]:
        del actor_user_id  # Authorization/audit is owned by the caller in the focused MVP.
        current = self._repository.read_raw_document(tenant_id)
        desired = self._normalize(document)
        self._validate_provider(desired, current)

        current_tenant = dict(current.get("tenant") or {})
        if desired["tenant"] != {
            "display_name": str(current_tenant.get("display_name") or tenant_id),
            "status": str(current_tenant.get("status") or "active"),
        }:
            self._repository.update_tenant_profile(
                tenant_id, desired["tenant"]["display_name"], desired["tenant"]["status"]
            )

        current_settings = dict(current.get("settings") or {})
        for field, key in SETTING_FIELDS.items():
            self._persist_setting_change(
                tenant_id, field, key, desired["settings"].get(key), current_settings
            )
        self._persist_setting_change(
            tenant_id,
            "scheduler_filters",
            "scheduler_filters",
            desired["settings"].get("scheduler_filters"),
            current_settings,
        )

        current_secrets = dict(current.get("secrets") or {})
        for field, key in SECRET_FIELDS.items():
            value = desired["secrets"].get(key)
            current_value = current_secrets.get(key)
            alias_present = False
            if current_value is None:
                for alias in SECRET_SETTING_ALIASES.get(field, ()):
                    if alias in current_settings:
                        current_value = current_settings[alias]
                        alias_present = True
                        break
            needs_canonicalization = (
                key not in current_secrets
                and alias_present
                and value is not None
                and self._secret_field_is_effective(field, desired["settings"])
            )
            if current_value is UNREADABLE_SECRET and value is None:
                # Blank cannot safely mean delete when the existing value was not readable.
                changed = False
            else:
                changed = value != current_value or needs_canonicalization
            if changed:
                if value is None:
                    self._repository.delete_secret(tenant_id, key)
                else:
                    try:
                        self._repository.upsert_secret(tenant_id, key, value)
                    except TenantSecretKeyUnavailableError:
                        if needs_canonicalization:
                            # An unchanged alias can remain readable until encryption is configured.
                            continue
                        raise
            if changed or key in current_secrets:
                for alias in SECRET_SETTING_ALIASES.get(field, ()):
                    if alias in current_settings:
                        self._repository.delete_setting(tenant_id, alias)

        return self.load(tenant_id)

    @staticmethod
    def _secret_field_is_effective(field: str, settings: Mapping[str, Any]) -> bool:
        provider = settings.get("telephony_provider")
        return (field == "vochi_api_key" and provider == "vochi") or (
            field in {"mts_domain", "mts_api_key"} and provider == "mts_vats"
        )

    def _persist_setting_change(
        self,
        tenant_id: str,
        field: str,
        key: str,
        value: Any,
        current_settings: Mapping[str, Any],
    ) -> None:
        changed = value != current_settings.get(key)
        if changed:
            if value is None:
                self._repository.delete_setting(tenant_id, key)
            else:
                self._repository.upsert_setting(tenant_id, key, value)
        if changed or key in current_settings:
            for alias in SETTING_ALIASES.get(field, ()):
                if alias in current_settings:
                    self._repository.delete_setting(tenant_id, alias)

    def _editable(self, tenant_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        tenant = dict(raw.get("tenant") or {})
        settings = dict(raw.get("settings") or {})
        secrets = dict(raw.get("secrets") or {})
        result: dict[str, Any] = {
            "tenant_id": tenant_id,
            "display_name": str(tenant.get("display_name") or tenant_id),
            "status": str(tenant.get("status") or "active"),
            "scheduler_time_from": "",
            "scheduler_time_to": "",
            "scheduler_call_type": "",
        }
        for field, key in SETTING_FIELDS.items():
            value = settings.get(key)
            if value is None:
                value = next(
                    (settings[alias] for alias in SETTING_ALIASES.get(field, ()) if alias in settings),
                    "",
                )
            result[field] = value
        filters = settings.get("scheduler_filters")
        if isinstance(filters, Mapping):
            result["scheduler_time_from"] = str(filters.get("time_from") or "")
            result["scheduler_time_to"] = str(filters.get("time_to") or "")
            result["scheduler_call_type"] = str(filters.get("call_type") or "")
        for field, key in SECRET_FIELDS.items():
            value = secrets.get(key)
            if value is UNREADABLE_SECRET:
                value = ""
            if value is None:
                value = next(
                    (
                        settings[alias]
                        for alias in SECRET_SETTING_ALIASES.get(field, ())
                        if alias in settings
                    ),
                    "",
                )
            result[field] = str(value or "")
        result["active_prompts"] = sorted(
            [
                {
                    "key": str(row.get("key") or ""),
                    "title": str(row.get("title") or ""),
                    "body": str(row.get("body") or ""),
                    "version": int(row.get("version") or 1),
                }
                for row in raw.get("prompts") or []
                if row.get("is_active", True)
            ],
            key=lambda row: (row["key"], -row["version"]),
        )
        return result

    def _normalize(self, document: Mapping[str, Any]) -> dict[str, Any]:
        display_name = str(document.get("display_name") or "").strip()
        if not display_name:
            self._invalid("General / display name")
        status = str(document.get("status") or "").strip().casefold()
        if status not in {"active", "inactive"}:
            self._invalid("General / status")

        provider = str(document.get("telephony_provider") or "").strip().casefold()
        if provider not in {"", "vochi", "mts_vats"}:
            self._invalid("Telephony / provider")
        scheduler_mode = str(document.get("scheduler_mode") or "").strip().casefold()
        if scheduler_mode not in {"", "cron", "interval"}:
            self._invalid("Scheduler / mode")

        settings: dict[str, Any] = {}
        for field, key in SETTING_FIELDS.items():
            value = document.get(field)
            if field in BOOLEAN_FIELDS:
                if value not in (None, ""):
                    settings[key] = bool(value)
            elif field in INTEGER_FIELDS:
                if value not in (None, ""):
                    try:
                        converted = int(value)
                    except (TypeError, ValueError):
                        self._invalid(self._field_label(field))
                    if converted <= 0:
                        self._invalid(self._field_label(field))
                    settings[key] = converted
            else:
                text = str(value or "").strip()
                if text:
                    settings[key] = text

        for field in TIME_FIELDS:
            value = str(document.get(field) or "").strip()
            if value and not _TIME_RE.fullmatch(value):
                self._invalid(self._field_label(field))
        filters = {
            key: str(document.get(field) or "").strip()
            for field, key in (
                ("scheduler_time_from", "time_from"),
                ("scheduler_time_to", "time_to"),
                ("scheduler_call_type", "call_type"),
            )
            if str(document.get(field) or "").strip()
        }
        if filters:
            settings["scheduler_filters"] = filters

        secrets = {
            key: text
            for field, key in SECRET_FIELDS.items()
            if (text := str(document.get(field) or "").strip())
        }
        return {
            "tenant": {"display_name": display_name, "status": status},
            "settings": settings,
            "secrets": secrets,
        }

    def _validate_provider(
        self, desired: Mapping[str, Any], current: Mapping[str, Any]
    ) -> None:
        settings = desired["settings"]
        secrets = desired["secrets"]
        current_secrets = dict(current.get("secrets") or {})

        def secret_present(key: str) -> bool:
            return bool(secrets.get(key)) or current_secrets.get(key) is UNREADABLE_SECRET

        provider = settings.get("telephony_provider", "")
        if provider == "vochi":
            if not settings.get("vochi_base_url"):
                self._invalid("Telephony / VoChi base URL")
            if not secret_present("VOCHI_API_KEY"):
                self._invalid("Telephony / VoChi API key")
        elif provider == "mts_vats":
            if not secret_present("MTS_DOMAIN"):
                self._invalid("Telephony / MTS domain")
            if not secret_present("MTS_API_KEY"):
                self._invalid("Telephony / MTS API key")

    @staticmethod
    def _field_label(field: str) -> str:
        labels = {
            "batch_size": "Batch processing / batch size",
            "scheduler_interval_minutes": "Scheduler / interval",
            "scheduler_cron_time": "Scheduler / cron time",
            "scheduler_time_from": "Scheduler / time from",
            "scheduler_time_to": "Scheduler / time to",
        }
        return labels.get(field, field)

    @staticmethod
    def _invalid(field: str) -> None:
        raise TenantAdminValidationError(f"Invalid value: {field}")
