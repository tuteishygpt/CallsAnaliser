from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from calls_analyser.api.http import create_api_app
from calls_analyser.domain.models import AnalysisResult, CallLogEntry, Language
from calls_analyser.services.analysis import AnalysisOptions
from calls_analyser.services.tenant import TenantConfig


class FakeTenantService:
    def __init__(self) -> None:
        self.resolved: list[str | None] = []

    def resolve(self, tenant_id: str | None = None) -> TenantConfig:
        self.resolved.append(tenant_id)
        return TenantConfig(tenant_id=tenant_id or "default", vochi_base_url="https://api")


class FakeCallLogService:
    def __init__(self) -> None:
        self.calls: list[tuple[date, TenantConfig]] = []

    def list_calls(self, day: date, tenant: TenantConfig) -> list[CallLogEntry]:
        self.calls.append((day, tenant))
        return [
            CallLogEntry(
                unique_id=f"{tenant.tenant_id}-call",
                raw={"tenant": tenant.tenant_id},
            )
        ]


class FakeAnalysisService:
    def analyze_call(
        self,
        unique_id: str,
        tenant: TenantConfig,
        lang: Language,
        options: AnalysisOptions,
    ) -> AnalysisResult:
        return AnalysisResult(
            text=f"{unique_id}:{tenant.tenant_id}:{lang.value}:{options.model_key}",
            model=options.model_key,
            provider="fake",
        )


class FakePromptService:
    def list_templates(self) -> dict[str, Any]:
        return {}


class FakeAuthService:
    def __init__(
        self,
        allowed: set[tuple[str, str]],
        credentials: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.allowed = allowed
        self.calls: list[tuple[str, str]] = []
        self.credentials = credentials or {}
        self.authenticate_calls: list[tuple[str, str]] = []

    def authenticate(self, login: str, password: str) -> AuthenticatedUser | None:
        self.authenticate_calls.append((login, password))
        user_id = self.credentials.get((login, password))
        if user_id is None:
            return None
        return AuthenticatedUser(user_id=user_id)

    def can_access_tenant(self, user_id: str, tenant_id: str) -> bool:
        self.calls.append((user_id, tenant_id))
        return (user_id, tenant_id) in self.allowed


class FakeTokenAuthService(FakeAuthService):
    def __init__(
        self,
        allowed: set[tuple[str, str]],
        tokens: dict[str, str],
        credentials: dict[tuple[str, str], str] | None = None,
    ) -> None:
        super().__init__(allowed=allowed, credentials=credentials)
        self.tokens = tokens
        self.authenticate_token_calls: list[str] = []

    def authenticate_token(self, token: str) -> AuthenticatedUser | None:
        self.authenticate_token_calls.append(token)
        user_id = self.tokens.get(token)
        if user_id is None:
            return None
        return AuthenticatedUser(user_id=user_id)


@dataclass
class AuthenticatedUser:
    user_id: str


@dataclass
class ApiHarness:
    client: TestClient
    tenant_service: FakeTenantService
    call_log_service: FakeCallLogService
    auth_service: FakeAuthService | None


_UNSET = object()


def build_harness(auth_service: FakeAuthService | None | object = _UNSET) -> ApiHarness:
    tenant_service = FakeTenantService()
    call_log_service = FakeCallLogService()
    kwargs: dict[str, Any] = {}
    if auth_service is not _UNSET:
        kwargs["auth_service"] = auth_service

    try:
        app = create_api_app(
            tenant_service=tenant_service,
            call_log_service=call_log_service,
            analysis_service=FakeAnalysisService(),
            prompt_service=FakePromptService(),
            ai_registry={},
            **kwargs,
        )
    except TypeError as exc:
        pytest.fail(f"create_api_app should accept auth_service: {exc}")

    return ApiHarness(
        client=TestClient(app),
        tenant_service=tenant_service,
        call_log_service=call_log_service,
        auth_service=auth_service if isinstance(auth_service, FakeAuthService) else None,
    )


def test_no_auth_service_preserves_tenant_query_behavior() -> None:
    harness = build_harness()

    response = harness.client.get("/calls/2024-06-01", params={"tenant_id": "tenant-a"})

    assert response.status_code == 200
    assert response.json() == {"data": [{"tenant": "tenant-a"}]}
    assert harness.call_log_service.calls[0][1].tenant_id == "tenant-a"


def test_missing_auth_returns_401_when_auth_service_configured() -> None:
    auth_service = FakeAuthService({("user-1", "tenant-a")})
    harness = build_harness(auth_service)

    response = harness.client.get("/calls/2024-06-01", params={"tenant_id": "tenant-a"})

    assert response.status_code == 401
    assert auth_service.calls == []
    assert harness.call_log_service.calls == []


def test_x_user_id_alone_returns_401_when_auth_service_configured() -> None:
    auth_service = FakeAuthService({("user-1", "tenant-a")})
    harness = build_harness(auth_service)

    response = harness.client.get(
        "/calls/2024-06-01",
        params={"tenant_id": "tenant-a"},
        headers={"X-User-Id": "user-1"},
    )

    assert response.status_code == 401
    assert auth_service.calls == []
    assert harness.call_log_service.calls == []


def test_basic_wrong_password_returns_401() -> None:
    auth_service = FakeAuthService(
        {("api-user", "tenant-a")},
        credentials={("api-user", "correct"): "api-user"},
    )
    harness = build_harness(auth_service)
    credentials = base64.b64encode(b"api-user:wrong").decode("ascii")

    response = harness.client.get(
        "/calls/2024-06-01",
        params={"tenant_id": "tenant-a"},
        headers={"Authorization": f"Basic {credentials}"},
    )

    assert response.status_code == 401
    assert auth_service.authenticate_calls == [("api-user", "wrong")]
    assert auth_service.calls == []
    assert harness.call_log_service.calls == []


def test_basic_correct_credentials_allowed_tenant_succeeds() -> None:
    auth_service = FakeAuthService(
        {("api-user-id", "tenant-a")},
        credentials={("api-login", "secret"): "api-user-id"},
    )
    harness = build_harness(auth_service)
    credentials = base64.b64encode(b"api-login:secret").decode("ascii")

    response = harness.client.post(
        "/analysis/call-1",
        params={"tenant_id": "tenant-a"},
        headers={"Authorization": f"Basic {credentials}"},
        json={
            "prompt_key": "simple",
            "model_key": "fake-model",
            "language": "en",
        },
    )

    assert response.status_code == 200
    assert response.json()["result"]["text"] == "call-1:tenant-a:en:fake-model"
    assert auth_service.authenticate_calls == [("api-login", "secret")]
    assert auth_service.calls == [("api-user-id", "tenant-a")]


def test_basic_correct_credentials_denied_tenant_returns_403() -> None:
    auth_service = FakeAuthService(
        {("api-user-id", "tenant-a")},
        credentials={("api-login", "secret"): "api-user-id"},
    )
    harness = build_harness(auth_service)
    credentials = base64.b64encode(b"api-login:secret").decode("ascii")

    response = harness.client.get(
        "/calls/2024-06-01",
        params={"tenant_id": "tenant-b"},
        headers={"Authorization": f"Basic {credentials}"},
    )

    assert response.status_code == 403
    assert auth_service.authenticate_calls == [("api-login", "secret")]
    assert auth_service.calls == [("api-user-id", "tenant-b")]
    assert harness.call_log_service.calls == []


def test_bearer_token_uses_authenticate_token_when_available() -> None:
    auth_service = FakeTokenAuthService(
        {("token-user-id", "tenant-a")},
        tokens={"opaque-token": "token-user-id"},
    )
    harness = build_harness(auth_service)

    response = harness.client.get(
        "/calls/2024-06-01",
        params={"tenant_id": "tenant-a"},
        headers={"Authorization": "Bearer opaque-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"data": [{"tenant": "tenant-a"}]}
    assert auth_service.authenticate_token_calls == ["opaque-token"]
    assert auth_service.calls == [("token-user-id", "tenant-a")]


def test_bearer_token_without_authenticate_token_is_not_treated_as_user_id() -> None:
    auth_service = FakeAuthService({("opaque-token", "tenant-a")})
    harness = build_harness(auth_service)

    response = harness.client.get(
        "/calls/2024-06-01",
        params={"tenant_id": "tenant-a"},
        headers={"Authorization": "Bearer opaque-token"},
    )

    assert response.status_code == 401
    assert auth_service.calls == []
    assert harness.call_log_service.calls == []
