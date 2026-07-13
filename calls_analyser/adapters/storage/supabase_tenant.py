"""Supabase-backed tenant/auth/settings/prompt repositories."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

try:  # pragma: no cover - optional dependency is exercised through fakes in tests
    from supabase import create_client
except ImportError:  # pragma: no cover
    create_client = None  # type: ignore[assignment]

from calls_analyser.services.auth import AuthUserRecord, TenantAccessRecord, TenantRecord
from calls_analyser.services.prompt import PromptTemplate
from calls_analyser.services.tenant_secret_codec import (
    UNREADABLE_SECRET,
    TenantSecretCodec,
    TenantSecretKeyUnavailableError,
)


class SupabaseAuthRepository:
    """Auth repository backed by Supabase tenant tables."""

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self._client = client or _create_supabase_client(supabase_url, supabase_key)

    def get_user_by_login(self, login: str) -> AuthUserRecord | None:
        rows = _execute_rows(
            self._client.table("tenant_users")
            .select("id, login, password_hash, display_name, is_active")
            .eq("login", login)
            .eq("is_active", True)
            .limit(1)
        )
        return _auth_user_record(rows[0]) if rows else None

    def get_user_by_id(self, user_id: str) -> AuthUserRecord | None:
        rows = _execute_rows(
            self._client.table("tenant_users")
            .select("id, login, password_hash, display_name, is_active")
            .eq("id", user_id)
            .eq("is_active", True)
            .limit(1)
        )
        return _auth_user_record(rows[0]) if rows else None

    def get_tenant(self, tenant_id: str) -> TenantRecord | None:
        rows = _execute_rows(
            self._client.table("tenants")
            .select("id, display_name, status")
            .eq("id", tenant_id)
            .eq("status", "active")
            .limit(1)
        )
        return _tenant_record(rows[0]) if rows else None

    def get_tenant_including_inactive(self, tenant_id: str) -> TenantRecord | None:
        rows = _execute_rows(
            self._client.table("tenants")
            .select("id, display_name, status")
            .eq("id", tenant_id)
            .limit(1)
        )
        return _tenant_record(rows[0]) if rows else None

    def list_access_for_user(self, user_id: str) -> list[TenantAccessRecord]:
        rows = _execute_rows(
            self._client.table("tenant_user_access")
            .select("user_id, tenant_id, role")
            .eq("user_id", user_id)
        )
        return [
            TenantAccessRecord(
                user_id=str(row["user_id"]),
                tenant_id=str(row["tenant_id"]),
                role=str(row.get("role") or "operator"),
            )
            for row in rows
        ]


class SupabaseTenantSettingsRepository:
    """Tenant settings and secrets repository backed by Supabase."""

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        *,
        client: Any | None = None,
        codec: TenantSecretCodec | None = None,
    ) -> None:
        self._client = client or _create_supabase_client(supabase_url, supabase_key)
        self._codec = codec or TenantSecretCodec(None)

    def get_setting(self, tenant_id: str, key: str) -> Any | None:
        rows = _execute_rows(
            self._client.table("tenant_settings")
            .select("value")
            .eq("tenant_id", tenant_id)
            .eq("key", key)
            .limit(1)
        )
        if not rows:
            return None
        return rows[0].get("value")

    def get_secret(self, tenant_id: str, key: str) -> Any | None:
        rows = _execute_rows(
            self._client.table("tenant_secrets")
            .select("encrypted_value")
            .eq("tenant_id", tenant_id)
            .eq("key", key)
            .limit(1)
        )
        if not rows:
            return None
        value = rows[0].get("encrypted_value")
        return None if value is None else self._codec.decrypt(tenant_id, key, str(value))

    def list_tenant_ids(self) -> Iterable[str]:
        tenant_ids: dict[str, None] = {}
        for table_name, column_name in (
            ("tenants", "id"),
            ("tenant_settings", "tenant_id"),
            ("tenant_secrets", "tenant_id"),
        ):
            rows = _execute_rows(self._client.table(table_name).select(column_name))
            for row in rows:
                value = row.get(column_name)
                if value:
                    tenant_ids[str(value)] = None
        return tenant_ids.keys()

    def read_tenant_profile(self, tenant_id: str) -> Mapping[str, Any]:
        tenant_rows = _execute_rows(
            self._client.table("tenants")
            .select("id, display_name, status")
            .eq("id", tenant_id)
            .limit(1)
        )
        if not tenant_rows:
            raise KeyError("Tenant not found")
        row = tenant_rows[0]
        return {
            "display_name": str(row.get("display_name") or tenant_id),
            "status": str(row.get("status") or "active"),
        }

    def read_settings(self, tenant_id: str) -> Mapping[str, Any]:
        rows = _execute_rows(
            self._client.table("tenant_settings")
            .select("key, value")
            .eq("tenant_id", tenant_id)
        )
        return {str(row["key"]): row.get("value") for row in rows}

    def read_secrets(self, tenant_id: str) -> Mapping[str, Any]:
        rows = _execute_rows(
            self._client.table("tenant_secrets")
            .select("key, encrypted_value")
            .eq("tenant_id", tenant_id)
        )
        result: dict[str, Any] = {}
        for row in rows:
            key = str(row["key"])
            try:
                result[key] = self._codec.decrypt(
                    tenant_id, key, str(row.get("encrypted_value") or "")
                )
            except TenantSecretKeyUnavailableError:
                result[key] = UNREADABLE_SECRET
        return result

    def read_active_prompts(self, tenant_id: str) -> list[Mapping[str, Any]]:
        rows = _execute_rows(
            self._client.table("tenant_prompt_templates")
            .select("id, key, title, body, is_active, version, created_by, created_at, updated_at")
            .eq("tenant_id", tenant_id)
            .eq("is_active", True)
            .order("key")
            .order("version", desc=True)
        )
        return [dict(row) for row in rows]

    def read_raw_document(self, tenant_id: str) -> Mapping[str, Any]:
        return {
            "tenant": self.read_tenant_profile(tenant_id),
            "settings": self.read_settings(tenant_id),
            "secrets": self.read_secrets(tenant_id),
            "prompts": self.read_active_prompts(tenant_id),
        }

    def update_tenant_profile(self, tenant_id: str, display_name: str, status: str) -> None:
        self._client.table("tenants").update(
            {"display_name": display_name, "status": status}
        ).eq("id", tenant_id).execute()

    def upsert_setting(self, tenant_id: str, key: str, value: Any) -> None:
        self._client.table("tenant_settings").upsert(
            {"tenant_id": tenant_id, "key": key, "value": value},
            on_conflict="tenant_id,key",
        ).execute()

    def delete_setting(self, tenant_id: str, key: str) -> None:
        self._client.table("tenant_settings").delete().eq("tenant_id", tenant_id).eq(
            "key", key
        ).execute()

    def upsert_secret(self, tenant_id: str, key: str, plaintext: str) -> None:
        encrypted = self._codec.encrypt(tenant_id, key, plaintext)
        self._client.table("tenant_secrets").upsert(
            {"tenant_id": tenant_id, "key": key, "encrypted_value": encrypted},
            on_conflict="tenant_id,key",
        ).execute()

    def delete_secret(self, tenant_id: str, key: str) -> None:
        self._client.table("tenant_secrets").delete().eq("tenant_id", tenant_id).eq(
            "key", key
        ).execute()


class SupabasePromptTemplateRepository:
    """Prompt template repository backed by Supabase tenant prompts."""

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self._client = client or _create_supabase_client(supabase_url, supabase_key)

    def get_template(self, tenant_id: str, key: str) -> PromptTemplate | None:
        rows = _execute_rows(
            self._client.table("tenant_prompt_templates")
            .select("key, title, body, version")
            .eq("tenant_id", tenant_id)
            .eq("key", key)
            .eq("is_active", True)
            .order("version", desc=True)
            .limit(1)
        )
        return _prompt_template(rows[0]) if rows else None

    def list_templates(self, tenant_id: str) -> Mapping[str, PromptTemplate]:
        rows = _execute_rows(
            self._client.table("tenant_prompt_templates")
            .select("key, title, body, version")
            .eq("tenant_id", tenant_id)
            .eq("is_active", True)
            .order("version", desc=True)
        )
        templates: dict[str, PromptTemplate] = {}
        for row in rows:
            key = str(row["key"])
            if key not in templates:
                templates[key] = _prompt_template(row)
        return templates


def _create_supabase_client(supabase_url: str | None, supabase_key: str | None) -> Any:
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and key are required")
    if create_client is None:
        raise ImportError("supabase package is required for Supabase repositories")
    return create_client(supabase_url, supabase_key)


def _execute_rows(query: Any) -> list[Mapping[str, Any]]:
    response = query.execute()
    return list(getattr(response, "data", None) or [])


def _auth_user_record(row: Mapping[str, Any]) -> AuthUserRecord:
    login = str(row["login"])
    return AuthUserRecord(
        user_id=str(row["id"]),
        login=login,
        password_hash=str(row["password_hash"]),
        display_name=str(row.get("display_name") or login),
        is_active=bool(row.get("is_active", True)),
    )


def _tenant_record(row: Mapping[str, Any]) -> TenantRecord:
    status = str(row.get("status") or "active").lower()
    return TenantRecord(
        tenant_id=str(row["id"]),
        display_name=str(row.get("display_name") or row["id"]),
        is_active=status == "active",
    )


def _prompt_template(row: Mapping[str, Any]) -> PromptTemplate:
    return PromptTemplate(
        key=str(row["key"]),
        title=str(row["title"]),
        body=str(row["body"]),
        version=int(row.get("version") or 1),
    )
