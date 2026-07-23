from __future__ import annotations

from calls_analyser.services.auth import (
    AuthService,
    InMemoryAuthRepository,
    TenantSummary,
    hash_password,
    verify_password,
)


def test_hash_password_verifies_password_and_rejects_wrong_password() -> None:
    password_hash = hash_password("correct horse battery staple", salt=b"fixed-salt", iterations=1000)

    assert password_hash.startswith("pbkdf2_sha256$1000$")
    assert verify_password("correct horse battery staple", password_hash) is True
    assert verify_password("wrong password", password_hash) is False


def test_authenticate_success_returns_allowed_tenant_summaries_and_roles() -> None:
    service = AuthService(
        InMemoryAuthRepository(
            users=[
                {
                    "id": "user-1",
                    "login": "alice",
                    "password_hash": hash_password("secret", salt=b"alice-salt", iterations=1000),
                    "display_name": "Alice Example",
                    "is_active": True,
                }
            ],
            tenants=[
                {"id": "tenant-a", "display_name": "Tenant A", "status": "active"},
                {"id": "tenant-b", "display_name": "Tenant B", "status": "active"},
            ],
            access=[
                {"user_id": "user-1", "tenant_id": "tenant-a", "role": "admin"},
                {"user_id": "user-1", "tenant_id": "tenant-b", "role": "operator"},
            ],
        )
    )

    user = service.authenticate("alice", "secret")

    assert user is not None
    assert user.user_id == "user-1"
    assert user.login == "alice"
    assert user.display_name == "Alice Example"
    assert user.allowed_tenants == [
        TenantSummary(tenant_id="tenant-a", display_name="Tenant A", role="admin"),
        TenantSummary(tenant_id="tenant-b", display_name="Tenant B", role="operator"),
    ]
    assert user.roles_by_tenant == {"tenant-a": "admin", "tenant-b": "operator"}


def test_authenticate_rejects_unknown_login_bad_password_and_inactive_user() -> None:
    service = AuthService(
        InMemoryAuthRepository(
            users=[
                {
                    "id": "active-user",
                    "login": "active",
                    "password_hash": hash_password("secret", salt=b"active-salt", iterations=1000),
                    "is_active": True,
                },
                {
                    "id": "inactive-user",
                    "login": "inactive",
                    "password_hash": hash_password("secret", salt=b"inactive-salt", iterations=1000),
                    "is_active": False,
                },
            ],
            tenants=[],
            access=[],
        )
    )

    assert service.authenticate("missing", "secret") is None
    assert service.authenticate("active", "wrong") is None
    assert service.authenticate("inactive", "secret") is None


def test_list_allowed_tenants_only_returns_active_tenants_assigned_to_user() -> None:
    service = AuthService(
        InMemoryAuthRepository(
            users=[
                {
                    "id": "user-1",
                    "login": "alice",
                    "password_hash": hash_password("secret", salt=b"alice-salt", iterations=1000),
                    "is_active": True,
                },
                {
                    "id": "inactive-user",
                    "login": "bob",
                    "password_hash": hash_password("secret", salt=b"bob-salt", iterations=1000),
                    "is_active": False,
                },
            ],
            tenants=[
                {"id": "tenant-a", "display_name": "Tenant A", "status": "active"},
                {"id": "tenant-b", "display_name": "Tenant B", "status": "inactive"},
                {"id": "tenant-c", "display_name": "Tenant C", "status": "active"},
            ],
            access=[
                {"user_id": "user-1", "tenant_id": "tenant-a", "role": "manager"},
                {"user_id": "user-1", "tenant_id": "tenant-b", "role": "operator"},
                {"user_id": "inactive-user", "tenant_id": "tenant-c", "role": "admin"},
            ],
        )
    )

    assert service.list_allowed_tenants("user-1") == [
        TenantSummary(tenant_id="tenant-a", display_name="Tenant A", role="manager")
    ]
    assert service.list_allowed_tenants("inactive-user") == []
    assert service.list_allowed_tenants("missing") == []


def test_can_access_tenant_enforces_user_status_and_assignments() -> None:
    service = AuthService(
        InMemoryAuthRepository(
            users=[
                {
                    "id": "user-1",
                    "login": "alice",
                    "password_hash": hash_password("secret", salt=b"alice-salt", iterations=1000),
                    "is_active": True,
                },
                {
                    "id": "inactive-user",
                    "login": "bob",
                    "password_hash": hash_password("secret", salt=b"bob-salt", iterations=1000),
                    "is_active": False,
                },
            ],
            tenants=[
                {"id": "tenant-a", "display_name": "Tenant A", "status": "active"},
                {"id": "tenant-b", "display_name": "Tenant B", "status": "inactive"},
                {"id": "tenant-c", "display_name": "Tenant C", "status": "active"},
            ],
            access=[
                {"user_id": "user-1", "tenant_id": "tenant-a", "role": "operator"},
                {"user_id": "user-1", "tenant_id": "tenant-b", "role": "operator"},
                {"user_id": "inactive-user", "tenant_id": "tenant-c", "role": "operator"},
            ],
        )
    )

    assert service.can_access_tenant("user-1", "tenant-a") is True
    assert service.can_access_tenant("user-1", "tenant-c") is False
    assert service.can_access_tenant("user-1", "tenant-b") is False
    assert service.can_access_tenant("inactive-user", "tenant-c") is False
    assert service.can_access_tenant("missing", "tenant-a") is False


def test_live_admin_tenants_include_inactive_and_require_current_admin_role() -> None:
    repository = InMemoryAuthRepository(
        users=[{"id": "user-1", "login": "alice", "password_hash": "x", "is_active": True}],
        tenants=[
            {"id": "active", "display_name": "Active", "status": "active"},
            {"id": "inactive", "display_name": "Inactive", "status": "inactive"},
            {"id": "operator", "display_name": "Operator", "status": "active"},
        ],
        access=[
            {"user_id": "user-1", "tenant_id": "active", "role": "ADMIN"},
            {"user_id": "user-1", "tenant_id": "inactive", "role": "admin"},
            {"user_id": "user-1", "tenant_id": "operator", "role": "operator"},
        ],
    )
    service = AuthService(repository)

    assert service.list_admin_tenants("user-1") == [
        TenantSummary("active", "Active", "ADMIN"),
        TenantSummary("inactive", "Inactive", "admin"),
    ]
    assert service.list_admin_tenants("user-1", include_inactive=False) == [
        TenantSummary("active", "Active", "ADMIN")
    ]
    assert service.can_administer_tenant("user-1", "inactive") is True

    repository._access_by_user["user-1"][1] = repository._access_by_user["user-1"][1].__class__(
        "user-1", "inactive", "operator"
    )
    assert service.can_administer_tenant("user-1", "inactive") is False
    assert service.can_administer_tenant("user-1", "forged") is False


def test_live_admin_authorization_rejects_inactive_user_and_removed_access() -> None:
    repository = InMemoryAuthRepository(
        users=[{"id": "user-1", "login": "alice", "password_hash": "x", "is_active": False}],
        tenants=[{"id": "tenant-a", "display_name": "A", "status": "active"}],
        access=[{"user_id": "user-1", "tenant_id": "tenant-a", "role": "admin"}],
    )
    service = AuthService(repository)
    assert service.list_admin_tenants("user-1") == []
    assert service.can_administer_tenant("user-1", "tenant-a") is False

    repository._users_by_id["user-1"] = repository._users_by_id["user-1"].__class__(
        "user-1", "alice", "x", "alice", True
    )
    repository._access_by_user["user-1"].clear()
    assert service.can_administer_tenant("user-1", "tenant-a") is False
