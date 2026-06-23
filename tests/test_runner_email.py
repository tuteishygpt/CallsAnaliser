from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

from calls_analyser.domain.models import AnalysisResult
from calls_analyser.runner import run_batch_process


class _Registry:
    def __init__(self, provider) -> None:  # noqa: ANN001
        self._provider = provider

    def get(self, _key: str):
        return self._provider


class _CallLogService:
    def __init__(self, entries) -> None:  # noqa: ANN001
        self._entries = entries

    def list_calls(self, *_args, **_kwargs):
        return list(self._entries)

    def ensure_recording(self, _unique_id, _tenant):
        raise AssertionError("Cached result must not download recording")


class _TenantService:
    def __init__(self, tenant) -> None:  # noqa: ANN001
        self._tenant = tenant

    def resolve(self, _tenant_id=None):
        return self._tenant


class _RecordingEmailReportService:
    def __init__(self) -> None:
        self.calls = []

    def send(self, results, **kwargs) -> None:  # noqa: ANN001
        self.calls.append((results.copy(), kwargs))


def test_run_batch_sends_cached_results_by_email() -> None:
    day = dt.date(2026, 6, 22)
    tenant = SimpleNamespace(
        tenant_id="lix",
        provider="vochi",
        recording_url=lambda unique_id: f"https://example.test/recording/{unique_id}",
    )
    entry = SimpleNamespace(
        started_at=dt.datetime(2026, 6, 22, 9, 0),
        caller_id="Client",
        destination="Support",
        duration_seconds=90,
        unique_id="call-1",
        raw={"recording_url": "https://example.test/permanent/call-1"},
    )
    provider = SimpleNamespace(provider_name="gemini")
    cache_key = (
        "lix",
        "call-1",
        "BATCH_PROMPT",
        "gemini",
        "models/gemini-test",
        "",
    )
    cache = {
        cache_key: AnalysisResult(
            text=json.dumps({"needs_follow_up": True, "reason": "Call the client"}),
            model="models/gemini-test",
            provider="gemini",
        )
    }
    email_service = _RecordingEmailReportService()
    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        call_log_service=_CallLogService([entry]),
        batch_language=SimpleNamespace(value="en"),
        batch_prompt_text="prompt",
        batch_prompt_key="BATCH_PROMPT",
        batch_model_key="models/gemini-test",
        ai_registry=_Registry(provider),
        analysis_service=SimpleNamespace(_cache=cache),
        email_report_service=email_service,
    )

    results = run_batch_process(deps, day, None, None, "", "lix")

    assert list(results["UniqueId"]) == ["call-1"]
    assert list(results["Needs follow-up"]) == ["Yes"]
    assert list(results["Reason"]) == ["Call the client"]
    assert len(email_service.calls) == 1
    emailed_results, options = email_service.calls[0]
    assert list(emailed_results["UniqueId"]) == ["call-1"]
    assert options == {
        "filter_option": "Needs follow-up",
        "report_date": "2026-06-22",
        "tenant_id": "lix",
    }
