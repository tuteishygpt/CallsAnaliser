from __future__ import annotations

import datetime as dt
import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest

os.environ.setdefault("DEFAULT_TENANT_ID", "tenant")
os.environ.setdefault("TENANT_VOCHI_API_KEY", "test-vochi-key")
os.environ.setdefault("TENANT_VOCHI_BASE_URL", "https://bot.example/api/v1")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

import app
from calls_analyser.ui import utils


class _StubTenantService:
    def __init__(self, tenant: SimpleNamespace) -> None:
        self._tenant = tenant

    def resolve(self, tenant_id: str | None = None) -> SimpleNamespace:
        return self._tenant


class _StubCallLogService:
    def __init__(self, entries: list[SimpleNamespace]) -> None:
        self._entries = entries

    def list_calls(self, *_, **__) -> list[SimpleNamespace]:
        return list(self._entries)


class _StubAnalysisService:
    def __init__(self, responses: dict[str, str | Exception]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, SimpleNamespace, object]] = []

    def analyze_call(self, unique_id: str, tenant: SimpleNamespace, lang, options):  # noqa: ANN001
        self.calls.append((unique_id, tenant, options))
        response = self._responses[unique_id]
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response)


class _RecordingEmailReportService:
    def __init__(self) -> None:
        self.calls = []

    def send(self, results, **kwargs) -> None:  # noqa: ANN001
        self.calls.append((results.copy(), kwargs))


def _configure_batch_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entries: list[SimpleNamespace],
    responses: dict[str, str | Exception] | None = None,
) -> tuple[SimpleNamespace, _StubAnalysisService | None]:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        provider="vochi",
        vochi_base_url="https://bot.example/api/v1",
        recording_url=lambda unique_id: f"https://bot.example/api/v1/recording/{unique_id}",
    )

    monkeypatch.setattr(app, "PROJECT_IMPORTS_AVAILABLE", True)
    monkeypatch.setattr(app, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(app, "call_log_service", _StubCallLogService(entries))
    monkeypatch.setattr(app, "ai_registry", {"fake-model": object()})
    monkeypatch.setattr(app, "BATCH_MODEL_KEY", "fake-model")
    monkeypatch.setattr(app, "BATCH_PROMPT_KEY", "batch")
    monkeypatch.setattr(app, "BATCH_PROMPT_TEXT", "")
    monkeypatch.setattr(app, "BATCH_LANGUAGE", app.Language.ENGLISH)

    analysis: _StubAnalysisService | None = None
    if responses is not None:
        analysis = _StubAnalysisService(responses)
        monkeypatch.setattr(app, "analysis_service", analysis)

    return tenant, analysis


def test_ui_mass_analyze_requires_authentication() -> None:
    result = list(app.ui_mass_analyze("2024-01-01", "", "", "", "tenant", False))

    assert len(result) == 1
    df_update, message, file_update, filter_update = result[0]
    assert df_update["visible"] is False
    assert file_update["visible"] is False
    assert filter_update["visible"] is False
    assert "Enter the password" in message


def test_ui_mass_analyze_reports_absence_of_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_batch_environment(monkeypatch, entries=[])

    result = list(app.ui_mass_analyze("2024-02-10", "", "", "", "tenant", True))

    assert len(result) == 1
    df_update, message, file_update, filter_update = result[0]
    assert isinstance(df_update["value"], pd.DataFrame)
    assert df_update["value"].empty
    assert df_update["visible"] is False
    assert message == "### ℹ️ No calls for the selected filter."
    assert file_update["visible"] is False
    assert filter_update["visible"] is False


def test_ui_mass_analyze_streams_partial_and_final_results(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        SimpleNamespace(
            started_at=dt.datetime(2024, 2, 15, 9, 30),
            caller_id="Alice",
            destination="Support",
            duration_seconds=123,
            unique_id="call-1",
            raw={"recording_url": "https://bot.example/permanent/call-1"},
        ),
        SimpleNamespace(
            started_at=dt.datetime(2024, 2, 15, 10, 0),
            caller_id="Bob",
            destination="Sales",
            duration_seconds=45,
            unique_id="call-2",
            raw={},
        ),
    ]
    responses = {
        "call-1": json.dumps({"needs_follow_up": True, "reason": "Schedule callback"}),
        "call-2": RuntimeError("network down"),
    }
    tenant, analysis = _configure_batch_environment(
        monkeypatch, entries=entries, responses=responses
    )

    result = list(app.ui_mass_analyze("2024-02-15", "", "", "", tenant.tenant_id, True))

    assert len(result) == 4

    initial_df_update, initial_message, _, initial_filter = result[0]
    assert initial_message == "### Starting batch analysis for 2 call(s)..."
    assert initial_df_update["visible"] is False
    assert initial_filter["visible"] is False

    partial_df_update, partial_message, _, partial_filter = result[1]
    assert "Analyzing 1/2" in partial_message
    assert partial_filter["visible"] is False
    partial_df = partial_df_update["value"]
    assert list(partial_df["Status"]) == ["✅"]
    assert list(partial_df["Needs follow-up"]) == ["Yes"]
    assert list(partial_df["Reason"]) == ["Schedule callback"]
    assert partial_df.iloc[0]["Link"] == (
        "<a href=\"https://bot.example/permanent/call-1\" target=\"_blank\">Listen</a>"
    )

    error_df_update, error_message, _, error_filter = result[2]
    assert "Analyzing 2/2" in error_message
    assert error_filter["visible"] is False
    error_df = error_df_update["value"]
    assert list(error_df["Status"]) == ["✅", "❌"]
    assert error_df.iloc[1]["Reason"].startswith("❌ network down")
    assert error_df.iloc[1]["Link"] == ""

    final_df_update, final_message, final_file, final_filter = result[3]
    assert final_message == "## ✅ Batch analysis completed. Found: 2, processed successfully: 1"
    final_df = final_df_update["value"]
    assert isinstance(final_df, pd.DataFrame)
    assert list(final_df["Status"]) == ["✅", "❌"]
    assert final_file["visible"] is False
    assert final_filter["visible"] is True

    assert analysis is not None
    assert [call[0] for call in analysis.calls] == ["call-1", "call-2"]
    for _, _, options in analysis.calls:
        assert options.model_key == "fake-model"
        assert options.prompt_key == "batch"


def test_send_results_email_uses_selected_filter_and_full_results(monkeypatch) -> None:
    report_service = _RecordingEmailReportService()
    monkeypatch.setattr(app.handlers.deps, "email_report_service", report_service, raising=False)
    full_results = pd.DataFrame(
        [
            {"UniqueId": "call-1", "Needs follow-up": "Yes"},
            {"UniqueId": "call-2", "Needs follow-up": "No"},
        ]
    )

    status = app.handlers.send_results_email(
        full_results,
        "No follow-up",
        "2026-06-22",
        "lix",
        True,
    )

    assert status == "✅ Email sent to tuttstt@gmail.com."
    sent_results, options = report_service.calls[0]
    assert list(sent_results["UniqueId"]) == ["call-1", "call-2"]
    assert options == {
        "filter_option": "No follow-up",
        "report_date": "2026-06-22",
        "tenant_id": "lix",
    }


def test_batch_row_select_uses_full_results_when_unique_id_is_hidden(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    handle = SimpleNamespace(
        local_uri="C:/tmp/call-1.mp3",
        source_uri="https://bot.example/permanent/call-1",
    )
    call_log_service = SimpleNamespace(ensure_recording=lambda *_args: handle)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(app.handlers.deps, "call_log_service", call_log_service)

    full_results = pd.DataFrame(
        [
            {
                "Start": "2026-06-23T18:53:25",
                "Caller": "+3753335105",
                "Destination": "150",
                "Duration (s)": 164,
                "UniqueId": "call-1",
                "Needs follow-up": "Yes",
                "Reason": "Needs callback",
                "Link": "",
                "Status": "✅",
            }
        ]
    )
    displayed_results = utils.prepare_results_display(full_results)
    assert "UniqueId" not in displayed_results.columns

    result = app.handlers.on_batch_row_select(
        displayed_results,
        full_results,
        "tenant",
        SimpleNamespace(index=(0, 0)),
    )

    dropdown_update, selected_uid, uid_markdown, listen_html, audio_uri, status_msg = result
    assert dropdown_update["value"] == "call-1"
    assert selected_uid == "call-1"
    assert "call-1" in uid_markdown
    assert "https://bot.example/permanent/call-1" in listen_html
    assert audio_uri == "C:/tmp/call-1.mp3"
    assert status_msg == "Ready ✅"


def test_batch_row_select_matches_filtered_view_after_index_reset(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    handle = SimpleNamespace(
        local_uri="C:/tmp/call-2.mp3",
        source_uri="https://bot.example/permanent/call-2",
    )
    call_log_service = SimpleNamespace(ensure_recording=lambda *_args: handle)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(app.handlers.deps, "call_log_service", call_log_service)

    full_results = pd.DataFrame(
        [
            {
                "Start": "2026-06-23T18:00:00",
                "Caller": "+3751111111",
                "Destination": "150",
                "Duration (s)": 10,
                "UniqueId": "call-1",
                "Needs follow-up": "No",
                "Reason": "Resolved",
                "Link": "",
                "Status": "✅",
            },
            {
                "Start": "2026-06-23T19:00:00",
                "Caller": "+3752222222",
                "Destination": "152",
                "Duration (s)": 20,
                "UniqueId": "call-2",
                "Needs follow-up": "Yes",
                "Reason": "Needs callback",
                "Link": "",
                "Status": "✅",
            },
        ]
    )
    filtered_view = utils.prepare_results_display(
        full_results[full_results["Needs follow-up"] == "Yes"]
    ).reset_index(drop=True)

    result = app.handlers.on_batch_row_select(
        filtered_view,
        full_results,
        "tenant",
        SimpleNamespace(index=(0, 0)),
    )

    assert result[1] == "call-2"
    assert result[4] == "C:/tmp/call-2.mp3"
