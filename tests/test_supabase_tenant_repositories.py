from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from calls_analyser.adapters.storage.supabase_tenant import (
    SupabaseAuthRepository,
    SupabasePromptTemplateRepository,
    SupabaseTenantSettingsRepository,
)
from calls_analyser.services.auth import AuthService
from calls_analyser.services.prompt import PromptService, PromptTemplate


@dataclass
class _Response:
    data: list[dict[str, Any]]


class _FakeQuery:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self._filters: list[tuple[str, Any]] = []
        self._order_by: str | None = None
        self._order_desc = False
        self._limit: int | None = None

    def select(self, *_args: Any, **_kwargs: Any) -> "_FakeQuery":
        return self

    def eq(self, column: str, value: Any) -> "_FakeQuery":
        self._filters.append((column, value))
        return self

    def order(self, column: str, *, desc: bool = False) -> "_FakeQuery":
        self._order_by = column
        self._order_desc = desc
        return self

    def limit(self, value: int) -> "_FakeQuery":
        self._limit = value
        return self

    def execute(self) -> _Response:
        rows = [
            row
            for row in self._rows
            if all(row.get(column) == value for column, value in self._filters)
        ]
        if self._order_by is not None:
            rows = sorted(
                rows,
                key=lambda row: row.get(self._order_by) or 0,
                reverse=self._order_desc,
            )
        if self._limit is not None:
            rows = rows[: self._limit]
        return _Response(list(rows))


class _FakeTable:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def select(self, *args: Any, **kwargs: Any) -> _FakeQuery:
        return _FakeQuery(self._rows).select(*args, **kwargs)


class _FakeClient:
    def __init__(self, tables: dict[str, list[dict[str, Any]]]) -> None:
        self._tables = tables

    def table(self, name: str) -> _FakeTable:
        return _FakeTable(self._tables[name])


def test_supabase_auth_repository_reads_active_users_tenants_and_access() -> None:
    client = _FakeClient(
        {
            "tenant_users": [
                {
                    "id": "user-1",
                    "login": "alice",
                    "password_hash": "hash",
                    "display_name": "Alice",
                    "is_active": True,
                },
                {
                    "id": "user-2",
                    "login": "bob",
                    "password_hash": "hash",
                    "display_name": "Bob",
                    "is_active": False,
                },
            ],
            "tenants": [
                {"id": "tenant-a", "display_name": "Tenant A", "status": "active"},
                {"id": "tenant-b", "display_name": "Tenant B", "status": "inactive"},
            ],
            "tenant_user_access": [
                {"user_id": "user-1", "tenant_id": "tenant-a", "role": "admin"},
                {"user_id": "user-1", "tenant_id": "tenant-b", "role": "operator"},
            ],
        }
    )
    repository = SupabaseAuthRepository(client=client)

    user = repository.get_user_by_login("alice")

    assert user is not None
    assert user.user_id == "user-1"
    assert user.login == "alice"
    assert repository.get_user_by_login("bob") is None
    assert repository.get_tenant("tenant-b") is None
    assert AuthService(repository).list_allowed_tenants("user-1")[0].tenant_id == "tenant-a"


def test_supabase_tenant_settings_repository_reads_json_secret_and_known_tenants() -> None:
    repository = SupabaseTenantSettingsRepository(
        client=_FakeClient(
            {
                "tenants": [{"id": "tenant-a"}, {"id": "tenant-b"}],
                "tenant_settings": [
                    {
                        "tenant_id": "tenant-a",
                        "key": "scheduler_filters",
                        "value": {"call_type": "missed"},
                    },
                    {"tenant_id": "tenant-c", "key": "batch_size", "value": 9},
                ],
                "tenant_secrets": [
                    {
                        "tenant_id": "tenant-b",
                        "key": "EMAIL_FROM",
                        "encrypted_value": "stored-ciphertext",
                    },
                    {
                        "tenant_id": "tenant-c",
                        "key": "EMAIL_TO",
                        "encrypted_value": "ciphertext-2",
                    },
                ],
            }
        )
    )

    assert repository.get_setting("tenant-a", "scheduler_filters") == {"call_type": "missed"}
    assert repository.get_setting("tenant-c", "batch_size") == 9
    assert repository.get_secret("tenant-b", "EMAIL_FROM") == "stored-ciphertext"
    assert list(repository.list_tenant_ids()) == ["tenant-a", "tenant-b", "tenant-c"]


def test_prompt_service_resolves_latest_active_supabase_tenant_prompt_before_global() -> None:
    repository = SupabasePromptTemplateRepository(
        client=_FakeClient(
            {
                "tenant_prompt_templates": [
                    {
                        "tenant_id": "tenant-a",
                        "key": "simple",
                        "title": "Inactive",
                        "body": "old",
                        "version": 10,
                        "is_active": False,
                    },
                    {
                        "tenant_id": "tenant-a",
                        "key": "simple",
                        "title": "Tenant Simple v2",
                        "body": "tenant prompt",
                        "version": 2,
                        "is_active": True,
                    },
                    {
                        "tenant_id": "tenant-a",
                        "key": "detailed",
                        "title": "Tenant Detailed",
                        "body": "tenant detailed",
                        "version": 3,
                        "is_active": True,
                    },
                ],
            }
        )
    )
    service = PromptService(
        {"simple": PromptTemplate(key="simple", title="Global Simple", body="global")},
        prompt_repository=repository,
    )

    template = service.get_prompt("simple", tenant_id="tenant-a")

    assert template == PromptTemplate(
        key="simple",
        title="Tenant Simple v2",
        body="tenant prompt",
        version=2,
    )
    assert service.get_prompt("missing", tenant_id="tenant-a").title == "Tenant Simple v2"
    assert service.list_templates("tenant-a")["detailed"].version == 3
