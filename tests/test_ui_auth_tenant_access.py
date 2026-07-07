from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pandas as pd

from calls_analyser.services.auth import AuthService, InMemoryAuthRepository, hash_password
from calls_analyser.ui.handlers import UIHandlers


class _TenantService:
    def __init__(self) -> None:
        self.resolved = []

    def resolve(self, tenant_id=None):  # noqa: ANN001
        self.resolved.append(tenant_id)
        return SimpleNamespace(
            tenant_id=tenant_id,
            provider="vochi",
            recording_url=lambda unique_id: f"https://example.test/{tenant_id}/{unique_id}",
        )


class _CallLogService:
    def __init__(self) -> None:
        self.list_calls_requests = []
        self.ensure_recording_requests = []

    def list_calls(self, day, tenant, **kwargs):  # noqa: ANN001
        self.list_calls_requests.append((day, tenant, kwargs))
        return [
            SimpleNamespace(
                started_at=dt.datetime(2026, 7, 1, 9, 0),
                caller_id="caller",
                destination="support",
                duration_seconds=30,
                unique_id="call-1",
                raw={"UniqueId": "call-1"},
            )
        ]

    def ensure_recording(self, unique_id, tenant):  # noqa: ANN001
        self.ensure_recording_requests.append((unique_id, tenant))
        return SimpleNamespace(
            local_uri=f"C:/tmp/{unique_id}.mp3",
            source_uri=f"https://example.test/{tenant.tenant_id}/{unique_id}.mp3",
        )


class _AnalysisService:
    def __init__(self) -> None:
        self.calls = []

    def analyze_call(self, unique_id, tenant, lang, options):  # noqa: ANN001
        self.calls.append((unique_id, tenant, lang, options))
        return SimpleNamespace(text=f"analysis for {unique_id}")


def _auth_service() -> AuthService:
    return AuthService(
        InMemoryAuthRepository(
            users=[
                {
                    "id": "user-1",
                    "login": "agent",
                    "password_hash": hash_password("secret", salt="tests"),
                    "display_name": "Agent",
                }
            ],
            tenants=[
                {"id": "tenant-a", "display_name": "Tenant A"},
                {"id": "tenant-b", "display_name": "Tenant B"},
            ],
            access=[
                {"user_id": "user-1", "tenant_id": "tenant-a", "role": "operator"},
                {"user_id": "user-1", "tenant_id": "tenant-b", "role": "admin"},
            ],
        )
    )


def _handlers(  # noqa: ANN001
    *,
    auth_service=None,
    call_log_service=None,
    analysis_service=None,
) -> UIHandlers:
    return UIHandlers(
        SimpleNamespace(
            auth_service=auth_service,
            project_imports_available=True,
            tenant_service=_TenantService(),
            call_log_service=call_log_service or _CallLogService(),
            ai_registry={"fake-model": object()},
            analysis_service=analysis_service or _AnalysisService(),
            usage_report_repository=None,
            email_report_service=None,
            batch_model_key="",
        )
    )


def test_successful_login_returns_allowed_tenant_dropdown_choices() -> None:
    handlers = _handlers(auth_service=_auth_service())

    authed, session, message, group_update, tenant_update, report_tenant_update = (
        handlers.check_password("agent", "secret")
    )

    assert authed is True
    assert session["user_id"] == "user-1"
    assert session["login"] == "agent"
    assert [tenant["tenant_id"] for tenant in session["allowed_tenants"]] == [
        "tenant-a",
        "tenant-b",
    ]
    assert tenant_update["choices"] == [
        ("Tenant A (operator)", "tenant-a"),
        ("Tenant B (admin)", "tenant-b"),
    ]
    assert tenant_update["value"] is None
    assert report_tenant_update["choices"] == tenant_update["choices"]
    assert "Access granted" in message
    assert group_update["visible"] is False


def test_denied_login_does_not_authenticate() -> None:
    handlers = _handlers(auth_service=_auth_service())

    authed, session, message, group_update, tenant_update, report_tenant_update = (
        handlers.check_password("agent", "wrong")
    )

    assert authed is False
    assert session == {}
    assert "Incorrect" in message
    assert group_update["visible"] is True
    assert tenant_update["choices"] == []
    assert report_tenant_update["choices"] == []


def test_authenticated_filter_rejects_unauthorized_tenant_before_service_call() -> None:
    call_log_service = _CallLogService()
    handlers = _handlers(auth_service=_auth_service(), call_log_service=call_log_service)
    _, session, *_ = handlers.check_password("agent", "secret")

    result = handlers.filter_calls(
        "2026-07-01",
        "",
        "",
        "",
        True,
        "tenant-c",
        session,
    )

    assert "Access denied" in result[3]
    assert call_log_service.list_calls_requests == []


def test_legacy_bool_authed_path_still_uses_tenant_text_input() -> None:
    call_log_service = _CallLogService()
    handlers = _handlers(auth_service=None, call_log_service=call_log_service)

    result = handlers.filter_calls(
        "2026-07-01",
        "",
        "",
        "",
        True,
        "typed-tenant",
    )

    assert result[3] == "Calls found: 1"
    assert call_log_service.list_calls_requests[0][1].tenant_id == "typed-tenant"


def test_legacy_play_audio_still_uses_tenant_text_input() -> None:
    call_log_service = _CallLogService()
    handlers = _handlers(auth_service=None, call_log_service=call_log_service)
    df = pd.DataFrame([{"UniqueId": "call-1"}])

    html, audio_uri, status = handlers.play_audio(
        "0",
        df,
        "typed-tenant",
        True,
    )

    assert "https://example.test/typed-tenant/call-1.mp3" in html
    assert audio_uri == "C:/tmp/call-1.mp3"
    assert status.startswith("Ready")
    assert call_log_service.ensure_recording_requests[0][1].tenant_id == "typed-tenant"


def test_auth_service_play_audio_requires_active_session_before_recording_call() -> None:
    call_log_service = _CallLogService()
    handlers = _handlers(auth_service=_auth_service(), call_log_service=call_log_service)
    df = pd.DataFrame([{"UniqueId": "call-1"}])

    html, audio_uri, status = handlers.play_audio(
        "0",
        df,
        "tenant-a",
        True,
        auth_session=None,
    )

    assert "Access denied" in html
    assert audio_uri is None
    assert status == ""
    assert call_log_service.ensure_recording_requests == []


def test_auth_service_play_audio_allows_valid_session_for_allowed_tenant() -> None:
    call_log_service = _CallLogService()
    handlers = _handlers(auth_service=_auth_service(), call_log_service=call_log_service)
    _, session, *_ = handlers.check_password("agent", "secret")
    df = pd.DataFrame([{"UniqueId": "call-1"}])

    html, audio_uri, status = handlers.play_audio(
        "0",
        df,
        "tenant-a",
        True,
        session,
    )

    assert "https://example.test/tenant-a/call-1.mp3" in html
    assert audio_uri == "C:/tmp/call-1.mp3"
    assert status.startswith("Ready")
    assert call_log_service.ensure_recording_requests[0][1].tenant_id == "tenant-a"


def test_auth_service_analyze_bridge_requires_active_session_before_analysis_call() -> None:
    analysis_service = _AnalysisService()
    handlers = _handlers(auth_service=_auth_service(), analysis_service=analysis_service)

    output = list(
        handlers.analyze_bridge(
            "call-1",
            pd.DataFrame(),
            pd.DataFrame(),
            "simple",
            "",
            "en",
            "fake-model",
            "tenant-a",
            "",
            True,
            auth_session=None,
        )
    )

    assert output == ["Access denied. Sign in to continue."]
    assert analysis_service.calls == []


def test_auth_service_analyze_bridge_allows_valid_session_for_allowed_tenant() -> None:
    analysis_service = _AnalysisService()
    handlers = _handlers(auth_service=_auth_service(), analysis_service=analysis_service)
    _, session, *_ = handlers.check_password("agent", "secret")

    output = list(
        handlers.analyze_bridge(
            "call-1",
            pd.DataFrame(),
            pd.DataFrame(),
            "simple",
            "",
            "en",
            "fake-model",
            "tenant-a",
            "",
            True,
            session,
        )
    )

    assert output[-1] == "### Analysis result\n\nanalysis for call-1"
    assert analysis_service.calls[0][1].tenant_id == "tenant-a"


def test_auth_service_batch_row_select_requires_active_session_before_recording_call() -> None:
    call_log_service = _CallLogService()
    handlers = _handlers(auth_service=_auth_service(), call_log_service=call_log_service)
    full_df = pd.DataFrame(
        [
            {
                "Start": "2026-07-01T09:00:00",
                "Caller": "caller",
                "Destination": "support",
                "Duration (s)": 30,
                "UniqueId": "call-1",
            }
        ]
    )

    result = handlers.on_batch_row_select(
        full_df,
        full_df,
        "tenant-a",
        True,
        SimpleNamespace(index=0),
        auth_session=None,
    )

    assert "Access denied" in result[3]
    assert result[4] is None
    assert call_log_service.ensure_recording_requests == []


def test_auth_service_batch_row_select_allows_valid_session_for_allowed_tenant() -> None:
    call_log_service = _CallLogService()
    handlers = _handlers(auth_service=_auth_service(), call_log_service=call_log_service)
    _, session, *_ = handlers.check_password("agent", "secret")
    full_df = pd.DataFrame(
        [
            {
                "Start": "2026-07-01T09:00:00",
                "Caller": "caller",
                "Destination": "support",
                "Duration (s)": 30,
                "UniqueId": "call-1",
            }
        ]
    )

    result = handlers.on_batch_row_select(
        full_df,
        full_df,
        "tenant-a",
        True,
        SimpleNamespace(index=0),
        session,
    )

    assert "https://example.test/tenant-a/call-1.mp3" in result[3]
    assert result[4] == "C:/tmp/call-1.mp3"
    assert call_log_service.ensure_recording_requests[0][1].tenant_id == "tenant-a"
