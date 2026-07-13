from __future__ import annotations

from dataclasses import dataclass
import base64
from typing import Any

from calls_analyser.adapters.storage.supabase_tenant import (
    SupabaseAuthRepository,
    SupabasePromptTemplateRepository,
    SupabaseTenantSettingsRepository,
)
from calls_analyser.services.auth import AuthService
from calls_analyser.services.prompt import PromptService, PromptTemplate
from calls_analyser.services.tenant_admin_settings import TenantAdminSettingsService
from calls_analyser.services.tenant_secret_codec import TenantSecretCodec


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

    def update(self, values: dict[str, Any]) -> "_FakeMutation":
        return _FakeMutation(self._rows, "update", values)

    def upsert(self, values: dict[str, Any], **_kwargs: Any) -> "_FakeMutation":
        return _FakeMutation(self._rows, "upsert", values)

    def delete(self) -> "_FakeMutation":
        return _FakeMutation(self._rows, "delete", {})


class _FakeMutation:
    def __init__(self, rows, operation, values) -> None:
        self._rows = rows
        self._operation = operation
        self._values = values
        self._filters: list[tuple[str, Any]] = []

    def eq(self, column: str, value: Any) -> "_FakeMutation":
        self._filters.append((column, value))
        return self

    def execute(self) -> _Response:
        matching = [
            row for row in self._rows
            if all(row.get(column) == value for column, value in self._filters)
        ]
        if self._operation == "delete":
            self._rows[:] = [row for row in self._rows if row not in matching]
        elif self._operation == "update":
            for row in matching:
                row.update(self._values)
        elif self._operation == "upsert":
            key = (self._values.get("tenant_id"), self._values.get("key"))
            row = next(
                (
                    item for item in self._rows
                    if (item.get("tenant_id"), item.get("key")) == key
                ),
                None,
            )
            if row is None:
                self._rows.append(dict(self._values))
            else:
                row.update(self._values)
        return _Response([])


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


def test_supabase_shared_repository_uses_focused_encrypted_writes() -> None:
    key = base64.urlsafe_b64encode(b"z" * 32).decode().rstrip("=")
    codec = TenantSecretCodec(key)
    encrypted = codec.encrypt("tenant-a", "VOCHI_API_KEY", "plain-secret")
    tables = {
        "tenants": [{"id": "tenant-a", "display_name": "A", "status": "active"}],
        "tenant_settings": [
            {"tenant_id": "tenant-a", "key": "batch_size", "value": 20},
            {"tenant_id": "tenant-a", "key": "arbitrary", "value": {"keep": True}},
        ],
        "tenant_secrets": [
            {"tenant_id": "tenant-a", "key": "VOCHI_API_KEY", "encrypted_value": encrypted},
            {"tenant_id": "tenant-a", "key": "EXTRA", "encrypted_value": "legacy-extra"},
        ],
        "tenant_prompt_templates": [],
    }
    client = _FakeClient(
        tables
    )
    repository = SupabaseTenantSettingsRepository(client=client, codec=codec)

    raw = repository.read_raw_document("tenant-a")
    assert raw["secrets"] == {
        "VOCHI_API_KEY": "plain-secret",
        "EXTRA": "legacy-extra",
    }

    repository.update_tenant_profile("tenant-a", "Updated", "inactive")
    repository.upsert_setting("tenant-a", "batch_size", 25)
    repository.delete_setting("tenant-a", "missing")
    repository.upsert_secret("tenant-a", "VOCHI_API_KEY", "new-secret")
    repository.delete_secret("tenant-a", "missing")

    stored = next(
        row["encrypted_value"]
        for row in tables["tenant_secrets"]
        if row["key"] == "VOCHI_API_KEY"
    )
    assert stored.startswith("enc:v1:")
    assert "new-secret" not in stored
    assert tables["tenants"][0]["display_name"] == "Updated"
    assert tables["tenants"][0]["status"] == "inactive"
    assert next(row["value"] for row in tables["tenant_settings"] if row["key"] == "batch_size") == 25
    assert any(row["key"] == "arbitrary" for row in tables["tenant_settings"])
    assert any(row["key"] == "EXTRA" for row in tables["tenant_secrets"])


def test_supabase_admin_load_hides_encrypted_secret_when_master_key_is_missing() -> None:
    codec = TenantSecretCodec(base64.urlsafe_b64encode(b"z" * 32).decode().rstrip("="))
    encrypted = codec.encrypt("tenant-a", "VOCHI_API_KEY", "hidden-value")
    repository = SupabaseTenantSettingsRepository(
        client=_FakeClient(
            {
                "tenants": [{"id": "tenant-a", "display_name": "A", "status": "active"}],
                "tenant_settings": [
                    {"tenant_id": "tenant-a", "key": "telephony_provider", "value": "vochi"},
                    {"tenant_id": "tenant-a", "key": "vochi_base_url", "value": "https://vochi.test"},
                ],
                "tenant_secrets": [
                    {
                        "tenant_id": "tenant-a",
                        "key": "VOCHI_API_KEY",
                        "encrypted_value": encrypted,
                    }
                ],
                "tenant_prompt_templates": [],
            }
        ),
        codec=TenantSecretCodec(None),
    )

    document = TenantAdminSettingsService(repository).load("tenant-a")

    assert document["vochi_api_key"] == ""
    assert "hidden-value" not in repr(document)
    assert encrypted not in repr(document)
