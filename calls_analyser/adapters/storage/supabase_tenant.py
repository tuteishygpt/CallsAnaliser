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
    ) -> None:
        self._client = client or _create_supabase_client(supabase_url, supabase_key)

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
        return rows[0].get("encrypted_value")

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
