"""Authentication and tenant authorization service."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol


DEFAULT_PBKDF2_ITERATIONS = 600_000
PASSWORD_HASH_ALGORITHM = "pbkdf2_sha256"


@dataclass(frozen=True)
class TenantSummary:
    tenant_id: str
    display_name: str
    role: str


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    login: str
    display_name: str
    allowed_tenants: list[TenantSummary]
    roles_by_tenant: dict[str, str]


@dataclass(frozen=True)
class AuthUserRecord:
    user_id: str
    login: str
    password_hash: str
    display_name: str
    is_active: bool


@dataclass(frozen=True)
class TenantRecord:
    tenant_id: str
    display_name: str
    is_active: bool


@dataclass(frozen=True)
class TenantAccessRecord:
    user_id: str
    tenant_id: str
    role: str


class AuthRepository(Protocol):
    """Repository contract for auth stores."""

    def get_user_by_login(self, login: str) -> AuthUserRecord | None:
        """Return the user record for ``login`` if it exists."""

    def get_user_by_id(self, user_id: str) -> AuthUserRecord | None:
        """Return the user record for ``user_id`` if it exists."""

    def get_tenant(self, tenant_id: str) -> TenantRecord | None:
        """Return tenant metadata for ``tenant_id`` if it exists."""

    def list_access_for_user(self, user_id: str) -> list[TenantAccessRecord]:
        """Return tenant access rows assigned to ``user_id``."""


def hash_password(
    password: str,
    *,
    salt: bytes | str | None = None,
    iterations: int = DEFAULT_PBKDF2_ITERATIONS,
) -> str:
    """Hash ``password`` with PBKDF2-SHA256."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")

    salt_bytes = _salt_bytes(salt)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations)
    return "$".join(
        [
            PASSWORD_HASH_ALGORITHM,
            str(iterations),
            _b64encode(salt_bytes),
            _b64encode(digest),
        ]
    )


def verify_password(password: str, password_hash: str) -> bool:
    """Return whether ``password`` matches ``password_hash``."""

    try:
        algorithm, iterations_raw, salt_b64, digest_b64 = password_hash.split("$", 3)
        if algorithm != PASSWORD_HASH_ALGORITHM:
            return False
        iterations = int(iterations_raw)
        salt = _b64decode(salt_b64)
        expected_digest = _b64decode(digest_b64)
    except (ValueError, TypeError):
        return False

    if iterations <= 0:
        return False

    actual_digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual_digest, expected_digest)


class AuthService:
    """Authenticates users and checks tenant access."""

    def __init__(self, repository: AuthRepository) -> None:
        self._repository = repository

    def authenticate(self, login: str, password: str) -> AuthenticatedUser | None:
        user = self._repository.get_user_by_login(login)
        if user is None or not user.is_active:
            return None
        if not verify_password(password, user.password_hash):
            return None

        allowed_tenants = self.list_allowed_tenants(user.user_id)
        return AuthenticatedUser(
            user_id=user.user_id,
            login=user.login,
            display_name=user.display_name,
            allowed_tenants=allowed_tenants,
            roles_by_tenant={tenant.tenant_id: tenant.role for tenant in allowed_tenants},
        )

    def list_allowed_tenants(self, user_id: str) -> list[TenantSummary]:
        user = self._repository.get_user_by_id(user_id)
        if user is None or not user.is_active:
            return []

        summaries: list[TenantSummary] = []
        for access in self._repository.list_access_for_user(user_id):
            tenant = self._repository.get_tenant(access.tenant_id)
            if tenant is None or not tenant.is_active:
                continue
            summaries.append(
                TenantSummary(
                    tenant_id=tenant.tenant_id,
                    display_name=tenant.display_name,
                    role=access.role,
                )
            )
        return summaries

    def can_access_tenant(self, user_id: str, tenant_id: str) -> bool:
        return any(tenant.tenant_id == tenant_id for tenant in self.list_allowed_tenants(user_id))


class InMemoryAuthRepository:
    """In-memory auth repository for tests and local wiring experiments."""

    def __init__(
        self,
        *,
        users: Iterable[Mapping[str, object]],
        tenants: Iterable[Mapping[str, object]],
        access: Iterable[Mapping[str, object]],
    ) -> None:
        self._users_by_id: dict[str, AuthUserRecord] = {}
        self._users_by_login: dict[str, AuthUserRecord] = {}
        for row in users:
            record = _user_record(row)
            self._users_by_id[record.user_id] = record
            self._users_by_login[record.login] = record

        self._tenants_by_id: dict[str, TenantRecord] = {}
        for row in tenants:
            record = _tenant_record(row)
            self._tenants_by_id[record.tenant_id] = record

        self._access_by_user: dict[str, list[TenantAccessRecord]] = {}
        for row in access:
            record = TenantAccessRecord(
                user_id=str(row["user_id"]),
                tenant_id=str(row["tenant_id"]),
                role=str(row.get("role") or "operator"),
            )
            self._access_by_user.setdefault(record.user_id, []).append(record)

    def get_user_by_login(self, login: str) -> AuthUserRecord | None:
        return self._users_by_login.get(login)

    def get_user_by_id(self, user_id: str) -> AuthUserRecord | None:
        return self._users_by_id.get(user_id)

    def get_tenant(self, tenant_id: str) -> TenantRecord | None:
        return self._tenants_by_id.get(tenant_id)

    def list_access_for_user(self, user_id: str) -> list[TenantAccessRecord]:
        return list(self._access_by_user.get(user_id, []))


def _salt_bytes(salt: bytes | str | None) -> bytes:
    if salt is None:
        return secrets.token_bytes(16)
    if isinstance(salt, bytes):
        return salt
    return salt.encode("utf-8")


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def _user_record(row: Mapping[str, object]) -> AuthUserRecord:
    user_id = row.get("id", row.get("user_id"))
    if user_id is None:
        raise ValueError("user row requires id or user_id")

    login = row.get("login")
    if login is None:
        raise ValueError("user row requires login")

    password_hash = row.get("password_hash")
    if password_hash is None:
        raise ValueError("user row requires password_hash")

    return AuthUserRecord(
        user_id=str(user_id),
        login=str(login),
        password_hash=str(password_hash),
        display_name=str(row.get("display_name") or login),
        is_active=bool(row.get("is_active", True)),
    )


def _tenant_record(row: Mapping[str, object]) -> TenantRecord:
    tenant_id = row.get("id", row.get("tenant_id"))
    if tenant_id is None:
        raise ValueError("tenant row requires id or tenant_id")

    display_name = row.get("display_name") or tenant_id
    status = str(row.get("status") or "active").lower()
    is_active = bool(row.get("is_active", status == "active")) and status == "active"
    return TenantRecord(
        tenant_id=str(tenant_id),
        display_name=str(display_name),
        is_active=is_active,
    )
