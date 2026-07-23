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
        self.list_calls_count = 0

    def list_calls(self, *_, **__) -> list[SimpleNamespace]:
        self.list_calls_count += 1
        return list(self._entries)


class _StubAnalysisService:
    def __init__(self, responses: dict[str, str | Exception]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, SimpleNamespace, object]] = []
        self.languages: list[object] = []

    def analyze_call(self, unique_id: str, tenant: SimpleNamespace, lang, options):  # noqa: ANN001
        self.calls.append((unique_id, tenant, options))
        self.languages.append(lang)
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
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "tenant_settings_service", None, raising=False)

    analysis: _StubAnalysisService | None = None
    if responses is not None:
        analysis = _StubAnalysisService(responses)
        monkeypatch.setattr(app, "analysis_service", analysis)

    return tenant, analysis


class _StubTenantSettingsService:
    def __init__(self, settings: object = None, error: Exception | None = None) -> None:
        self.settings = settings
        self.error = error
        self.calls: list[str] = []

    def resolve(self, tenant_id: str) -> object:
        self.calls.append(tenant_id)
        if self.error is not None:
            raise self.error
        return self.settings


def _batch_entry(unique_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        started_at=dt.datetime(2024, 2, 15, 9, 30),
        caller_id="Alice",
        destination="Support",
        duration_seconds=123,
        unique_id=unique_id,
        raw={},
    )


def test_ui_mass_analyze_requires_authentication() -> None:
    result = list(app.ui_mass_analyze("2024-01-01", "", "", "", "tenant", False))

    assert len(result) == 1
    df_update, state_df, message, file_update, filter_update = result[0]
    assert df_update["visible"] is False
    assert isinstance(state_df, pd.DataFrame)
    assert state_df.empty
    assert file_update["visible"] is False
    assert filter_update["visible"] is False
    assert "Enter the password" in message


def test_ui_mass_analyze_reports_absence_of_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_batch_environment(monkeypatch, entries=[])

    result = list(app.ui_mass_analyze("2024-02-10", "", "", "", "tenant", True))

    assert len(result) == 1
    df_update, state_df, message, file_update, filter_update = result[0]
    assert isinstance(df_update["value"], pd.DataFrame)
    assert df_update["value"].empty
    assert isinstance(state_df, pd.DataFrame)
    assert state_df.empty
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

    initial_df_update, initial_state, initial_message, _, initial_filter = result[0]
    assert initial_message == "### Starting batch analysis for 2 call(s)..."
    assert initial_df_update["visible"] is False
    assert initial_state.empty
    assert initial_filter["visible"] is False

    partial_df_update, partial_state, partial_message, _, partial_filter = result[1]
    assert "Analyzing 1/2" in partial_message
    assert partial_filter["visible"] is False
    partial_df = partial_df_update["value"]
    assert "UniqueId" not in partial_df.columns
    assert list(partial_state["UniqueId"]) == ["call-1"]
    assert list(partial_df["Status"]) == ["✅"]
    assert list(partial_df["Needs follow-up"]) == ["Yes"]
    assert list(partial_df["Reason"]) == ["Schedule callback"]
    assert partial_df.iloc[0]["Link"] == (
        "<a href=\"https://bot.example/permanent/call-1\" target=\"_blank\">Listen</a>"
    )

    error_df_update, error_state, error_message, _, error_filter = result[2]
    assert "Analyzing 2/2" in error_message
    assert error_filter["visible"] is False
    error_df = error_df_update["value"]
    assert "UniqueId" not in error_df.columns
    assert list(error_state["UniqueId"]) == ["call-1", "call-2"]
    assert list(error_df["Status"]) == ["✅", "❌"]
    assert error_df.iloc[1]["Reason"].startswith("❌ network down")
    assert error_df.iloc[1]["Link"] == ""

    final_df_update, final_state, final_message, final_file, final_filter = result[3]
    assert final_message == "## ✅ Batch analysis completed. Found: 2, processed successfully: 1"
    final_df = final_df_update["value"]
    assert isinstance(final_df, pd.DataFrame)
    assert "UniqueId" not in final_df.columns
    assert list(final_state["UniqueId"]) == ["call-1", "call-2"]
    assert list(final_df["Status"]) == ["✅", "❌"]
    assert final_file["visible"] is False
    assert final_filter["visible"] is True

    assert analysis is not None
    assert [call[0] for call in analysis.calls] == ["call-1", "call-2"]
    for _, _, options in analysis.calls:
        assert options.model_key == "fake-model"
        assert options.prompt_key == "batch"


def test_ui_mass_analyze_resolves_tenant_batch_model_and_language_once(monkeypatch) -> None:
    entries = [_batch_entry("call-1"), _batch_entry("call-2")]
    tenant, analysis = _configure_batch_environment(
        monkeypatch,
        entries=entries,
        responses={"call-1": "{}", "call-2": "{}"},
    )
    settings_service = _StubTenantSettingsService(
        SimpleNamespace(batch_model_key="tenant-model", batch_language_code="be")
    )
    monkeypatch.setattr(
        app, "ai_registry", {"fake-model": object(), "tenant-model": object()}
    )
    monkeypatch.setattr(app.handlers.deps, "tenant_settings_service", settings_service)

    list(app.ui_mass_analyze("2024-02-15", "", "", "", tenant.tenant_id, True))

    assert settings_service.calls == [tenant.tenant_id]
    assert analysis is not None
    assert [call[2].model_key for call in analysis.calls] == [
        "tenant-model",
        "tenant-model",
    ]
    assert analysis.languages == [app.Language.BELARUSIAN, app.Language.BELARUSIAN]


@pytest.mark.parametrize("language_code", ["", "invalid"])
def test_ui_mass_analyze_falls_back_for_blank_or_invalid_tenant_language(
    monkeypatch, language_code
) -> None:
    tenant, analysis = _configure_batch_environment(
        monkeypatch,
        entries=[_batch_entry("call-1")],
        responses={"call-1": "{}"},
    )
    settings_service = _StubTenantSettingsService(
        SimpleNamespace(batch_model_key="", batch_language_code=language_code)
    )
    monkeypatch.setattr(app.handlers.deps, "tenant_settings_service", settings_service)

    list(app.ui_mass_analyze("2024-02-15", "", "", "", tenant.tenant_id, True))

    assert analysis is not None
    assert analysis.calls[0][2].model_key == "fake-model"
    assert analysis.languages == [app.Language.ENGLISH]


@pytest.mark.parametrize("language_code", ["auto", "default"])
def test_ui_mass_analyze_maps_auto_language_aliases(monkeypatch, language_code) -> None:
    tenant, analysis = _configure_batch_environment(
        monkeypatch,
        entries=[_batch_entry("call-1")],
        responses={"call-1": "{}"},
    )
    settings_service = _StubTenantSettingsService(
        SimpleNamespace(batch_model_key="", batch_language_code=language_code)
    )
    monkeypatch.setattr(app.handlers.deps, "tenant_settings_service", settings_service)

    list(app.ui_mass_analyze("2024-02-15", "", "", "", tenant.tenant_id, True))

    assert analysis is not None
    assert analysis.languages == [app.Language.AUTO]


def test_ui_mass_analyze_falls_back_when_tenant_settings_resolution_throws(monkeypatch) -> None:
    tenant, analysis = _configure_batch_environment(
        monkeypatch,
        entries=[_batch_entry("call-1")],
        responses={"call-1": "{}"},
    )
    settings_service = _StubTenantSettingsService(error=RuntimeError("settings unavailable"))
    monkeypatch.setattr(app.handlers.deps, "tenant_settings_service", settings_service)

    list(app.ui_mass_analyze("2024-02-15", "", "", "", tenant.tenant_id, True))

    assert settings_service.calls == [tenant.tenant_id]
    assert analysis is not None
    assert analysis.calls[0][2].model_key == "fake-model"
    assert analysis.languages == [app.Language.ENGLISH]


def test_ui_mass_analyze_rejects_unknown_tenant_model_before_listing_calls(monkeypatch) -> None:
    tenant, _ = _configure_batch_environment(monkeypatch, entries=[_batch_entry("call-1")])
    settings_service = _StubTenantSettingsService(
        SimpleNamespace(batch_model_key="unknown-model", batch_language_code="en")
    )
    monkeypatch.setattr(app.handlers.deps, "tenant_settings_service", settings_service)

    result = list(app.ui_mass_analyze("2024-02-15", "", "", "", tenant.tenant_id, True))

    assert result[0][2] == "## ❌ Configured batch model 'unknown-model' is not available."
    assert app.call_log_service.list_calls_count == 0


def test_ui_mass_analyze_does_not_mask_unexpected_registry_errors(monkeypatch) -> None:
    tenant, _ = _configure_batch_environment(monkeypatch, entries=[_batch_entry("call-1")])

    class BrokenRegistry:
        def get(self, key):  # noqa: ANN001
            raise RuntimeError("registry exploded")

    monkeypatch.setattr(app, "ai_registry", BrokenRegistry())

    result = list(app.ui_mass_analyze("2024-02-15", "", "", "", tenant.tenant_id, True))

    assert result[0][2] == "## ❌ Analysis failed: registry exploded"
    assert app.call_log_service.list_calls_count == 0


def test_send_results_email_uses_selected_filter_and_full_results(monkeypatch) -> None:
    report_service = _RecordingEmailReportService()
    monkeypatch.setattr(app.handlers.deps, "email_report_service", report_service, raising=False)
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
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
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
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
        True,
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
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
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
        True,
        SimpleNamespace(index=(0, 0)),
    )

    assert result[1] == "call-2"
    assert result[4] == "C:/tmp/call-2.mp3"


def test_batch_row_select_accepts_plain_row_index(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    handle = SimpleNamespace(
        local_uri="C:/tmp/call-2.mp3",
        source_uri="https://bot.example/permanent/call-2",
    )
    call_log_service = SimpleNamespace(ensure_recording=lambda *_args: handle)
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
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
        True,
        SimpleNamespace(index=0),
    )

    assert result[1] == "call-2"
    assert result[4] == "C:/tmp/call-2.mp3"


def test_batch_row_select_recovers_uid_from_link_when_state_hides_unique_id(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    handle = SimpleNamespace(
        local_uri="C:/tmp/call-2.mp3",
        source_uri="https://bot.example/permanent/call-2",
    )
    call_log_service = SimpleNamespace(ensure_recording=lambda *_args: handle)
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(app.handlers.deps, "call_log_service", call_log_service)

    full_results = pd.DataFrame(
        [
            {
                "Start": "2026-07-09T17:37:18",
                "Caller": "375293249850",
                "Destination": "user",
                "user": "muravitskaya_viktoriya",
                "Duration (s)": 120,
                "UniqueId": "call-2",
                "Needs follow-up": "Yes",
                "Reason": "Needs callback",
                "Link": '<a href="https://bot.example/recording/call-2" target="_blank">Listen</a>',
                "Status": "âœ…",
            }
        ]
    )
    displayed_results = utils.prepare_results_display(full_results)
    state_without_unique_id = displayed_results.copy()
    assert "UniqueId" not in state_without_unique_id.columns

    result = app.handlers.on_batch_row_select(
        displayed_results,
        state_without_unique_id,
        "tenant",
        True,
        SimpleNamespace(index=(0, 1)),
    )

    assert result[1] == "call-2"
    assert result[4] == "C:/tmp/call-2.mp3"


def test_batch_row_select_recovers_uid_from_visible_link_when_state_is_empty(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    handle = SimpleNamespace(
        local_uri="C:/tmp/call-2.mp3",
        source_uri="https://bot.example/permanent/call-2",
    )
    call_log_service = SimpleNamespace(ensure_recording=lambda *_args: handle)
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(app.handlers.deps, "call_log_service", call_log_service)

    displayed_results = pd.DataFrame(
        [
            {
                "Start": "2026-07-09 17:37:18",
                "Caller": "375293249850",
                "Destination": "user",
                "user": "muravitskaya_viktoriya",
                "Duration (s)": 120,
                "Needs follow-up": "Yes",
                "Reason": "Needs callback",
                "Link": '<a href="https://bot.example/recording/call-2" target="_blank">Listen</a>',
                "Status": "âœ…",
            }
        ]
    )

    result = app.handlers.on_batch_row_select(
        displayed_results,
        pd.DataFrame(),
        "tenant",
        True,
        SimpleNamespace(index=(0, 1)),
    )

    assert result[1] == "call-2"
    assert result[4] == "C:/tmp/call-2.mp3"


def test_batch_row_select_strips_mp3_suffix_from_recovered_record_uid(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        recording_url=lambda unique_id: f"https://mts.example/crmapi/v1/history/record/{unique_id}",
    )
    calls = []

    def ensure_recording(unique_id, _tenant, *_args):  # noqa: ANN001
        calls.append(unique_id)
        return SimpleNamespace(
            local_uri=f"C:/tmp/{unique_id}.mp3",
            source_uri=f"https://mts.example/crmapi/v1/history/record/{unique_id}",
        )

    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(
        app.handlers.deps,
        "call_log_service",
        SimpleNamespace(ensure_recording=ensure_recording),
    )

    displayed_results = pd.DataFrame(
        [
            {
                "Start": "2026-07-09 17:47:43",
                "Caller": "375293332636",
                "Destination": "user",
                "Duration (s)": 13,
                "Link": (
                    '<a href="https://mts.example/crmapi/v1/history/record/'
                    '375293332636_in_375336809226_2026_07_09-20_47_49_kepn.mp3" '
                    'target="_blank">Listen</a>'
                ),
            }
        ]
    )

    result = app.handlers.on_batch_row_select(
        displayed_results,
        pd.DataFrame(),
        "tenant",
        True,
        SimpleNamespace(index=(0, 0)),
    )

    assert calls == ["375293332636_in_375336809226_2026_07_09-20_47_49_kepn"]
    assert result[1] == "375293332636_in_375336809226_2026_07_09-20_47_49_kepn"


def test_batch_row_select_passes_recovered_record_url_to_call_log_service(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        recording_url=lambda unique_id: f"https://mts.example/crmapi/v1/history/record/{unique_id}",
    )
    calls = []
    record_url = (
        "https://mts.example/crmapi/v1/history/record/"
        "375293332636_in_375336809226_2026_07_09-20_47_49_kepn.mp3"
    )

    def ensure_recording(unique_id, _tenant, recording_url=None):  # noqa: ANN001
        calls.append((unique_id, recording_url))
        return SimpleNamespace(local_uri="C:/tmp/call.mp3", source_uri=recording_url)

    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(
        app.handlers.deps,
        "call_log_service",
        SimpleNamespace(ensure_recording=ensure_recording),
    )

    displayed_results = pd.DataFrame(
        [
            {
                "Start": "2026-07-09 17:47:43",
                "Caller": "375293332636",
                "Destination": "user",
                "Duration (s)": 13,
                "Link": f'<a href="{record_url}" target="_blank">Listen</a>',
            }
        ]
    )

    result = app.handlers.on_batch_row_select(
        displayed_results,
        pd.DataFrame(),
        "tenant",
        True,
        SimpleNamespace(index=(0, 0)),
    )

    assert calls == [
        ("375293332636_in_375336809226_2026_07_09-20_47_49_kepn", record_url)
    ]
    assert result[4] == "C:/tmp/call.mp3"


def test_play_audio_after_batch_uses_current_uid_when_dropdown_value_is_label(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    handle = SimpleNamespace(
        local_uri="C:/tmp/call-1.mp3",
        source_uri="https://bot.example/permanent/call-1",
    )
    call_log_service = SimpleNamespace(ensure_recording=lambda *_args: handle)
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "project_imports_available", True)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(app.handlers.deps, "call_log_service", call_log_service)

    html, audio_uri, status = app.handlers.play_audio(
        "Batch: 2026-07-06 19:24:01 | +375297857324 -> 150 (141s)",
        pd.DataFrame(),
        "tenant",
        True,
        current_uid="call-1",
    )

    assert "https://bot.example/permanent/call-1" in html
    assert audio_uri == "C:/tmp/call-1.mp3"
    assert status.startswith("Ready")


def test_direct_analysis_uses_batch_result_row_for_usage_metadata(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        provider="vochi",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    analysis = _StubAnalysisService({"call-1": "analysis"})
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "project_imports_available", True)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(app.handlers.deps, "ai_registry", {"fake-model": object()})
    monkeypatch.setattr(app.handlers.deps, "analysis_service", analysis)

    batch_results = pd.DataFrame(
        [
            {
                "Start": "2026-06-25T09:57:17+00:00",
                "Caller": "+375292873510",
                "Destination": "150",
                "Duration (s)": 66,
                "UniqueId": "call-1",
                "Needs follow-up": "No",
                "Reason": "",
                "Link": "",
                "Status": "✅",
            }
        ]
    )

    output = list(
        app.handlers.analyze_bridge(
            "call-1",
            pd.DataFrame(),
            batch_results,
            "simple",
            "",
            app.Language.ENGLISH,
            "fake-model",
            "tenant",
            "call-1",
            True,
        )
    )

    assert output[-1] == "### Analysis result\n\nanalysis"
    options = analysis.calls[0][2]
    assert options.mode == "ui_direct"
    assert options.call_entry.unique_id == "call-1"
    assert options.call_entry.started_at.isoformat() == "2026-06-25T09:57:17+00:00"
    assert options.call_entry.caller_id == "+375292873510"
    assert options.call_entry.destination == "150"
    assert options.call_entry.duration_seconds == 66


def test_direct_analysis_uses_current_uid_when_batch_display_hides_unique_id(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        provider="vochi",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    analysis = _StubAnalysisService({"call-1": "analysis"})
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "project_imports_available", True)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(app.handlers.deps, "ai_registry", {"fake-model": object()})
    monkeypatch.setattr(app.handlers.deps, "analysis_service", analysis)

    batch_results = pd.DataFrame(
        [
            {
                "Start": "2026-06-25T09:57:17+00:00",
                "Caller": "+375292873510",
                "Destination": "150",
                "Duration (s)": 66,
                "UniqueId": "call-1",
                "Needs follow-up": "No",
                "Reason": "",
                "Link": "",
                "Status": "✅",
            }
        ]
    )
    displayed_batch_results = utils.prepare_results_display(batch_results)
    assert "UniqueId" not in displayed_batch_results.columns

    output = list(
        app.handlers.analyze_bridge(
            None,
            pd.DataFrame(),
            displayed_batch_results,
            "simple",
            "",
            app.Language.ENGLISH,
            "fake-model",
            "tenant",
            "call-1",
            True,
        )
    )

    assert output[-1] == "### Analysis result\n\nanalysis"
    options = analysis.calls[0][2]
    assert options.call_entry is not None
    assert options.call_entry.unique_id == "call-1"


def test_direct_analysis_accepts_unique_id_dropdown_value_when_state_is_empty(monkeypatch) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        provider="vochi",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    analysis = _StubAnalysisService({"call-1": "analysis"})
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "project_imports_available", True)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(app.handlers.deps, "ai_registry", {"fake-model": object()})
    monkeypatch.setattr(app.handlers.deps, "analysis_service", analysis)

    output = list(
        app.handlers.analyze_bridge(
            "call-1",
            pd.DataFrame(),
            pd.DataFrame(),
            "simple",
            "",
            app.Language.ENGLISH,
            "fake-model",
            "tenant",
            "",
            True,
        )
    )

    assert output[-1] == "### Analysis result\n\nanalysis"
    assert analysis.calls[0][0] == "call-1"


def test_direct_analysis_uses_dropdown_model_without_resolving_tenant_batch_settings(
    monkeypatch,
) -> None:
    tenant = SimpleNamespace(
        tenant_id="tenant",
        provider="vochi",
        recording_url=lambda unique_id: f"https://bot.example/recording/{unique_id}",
    )
    analysis = _StubAnalysisService({"call-1": "analysis"})
    settings_service = _StubTenantSettingsService(
        SimpleNamespace(batch_model_key="tenant-batch-model", batch_language_code="be")
    )
    monkeypatch.setattr(app.handlers.deps, "auth_service", None, raising=False)
    monkeypatch.setattr(app.handlers.deps, "project_imports_available", True)
    monkeypatch.setattr(app.handlers.deps, "tenant_service", _StubTenantService(tenant))
    monkeypatch.setattr(
        app.handlers.deps,
        "ai_registry",
        {"dropdown-model": object(), "tenant-batch-model": object()},
    )
    monkeypatch.setattr(app.handlers.deps, "analysis_service", analysis)
    monkeypatch.setattr(app.handlers.deps, "tenant_settings_service", settings_service)

    output = list(
        app.handlers.analyze_bridge(
            "call-1",
            pd.DataFrame(),
            pd.DataFrame(),
            "simple",
            "",
            app.Language.ENGLISH,
            "dropdown-model",
            tenant.tenant_id,
            "",
            True,
        )
    )

    assert output[-1] == "### Analysis result\n\nanalysis"
    assert settings_service.calls == []
    assert analysis.calls[0][2].model_key == "dropdown-model"
    assert analysis.languages == [app.Language.ENGLISH]
