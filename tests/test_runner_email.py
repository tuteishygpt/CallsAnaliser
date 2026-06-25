from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

from calls_analyser.domain.models import AnalysisResult
from calls_analyser.runner import run_batch_process
from calls_analyser.services.gemini_batch import BatchAnalysisResult


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


class _PreparingCallLogService(_CallLogService):
    def ensure_recording(self, unique_id, _tenant):
        return SimpleNamespace(local_uri=f"{unique_id}.wav")


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


class _BulkOnlyCache:
    def __init__(self, results) -> None:  # noqa: ANN001
        self._results = results
        self.requested_keys = []

    def get_many(self, keys):  # noqa: ANN001
        self.requested_keys = list(keys)
        return {key: self._results[key] for key in self.requested_keys if key in self._results}

    def get(self, _key):  # noqa: ANN001
        raise AssertionError("runner should use bulk cache lookup")


class _RecordingUsageTracker:
    def __init__(self) -> None:
        self.calls = []

    def record(self, **kwargs) -> None:  # noqa: ANN003
        self.calls.append(kwargs)


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


def test_run_batch_uses_bulk_cache_lookup_for_cached_results() -> None:
    day = dt.date(2026, 6, 22)
    tenant = SimpleNamespace(
        tenant_id="lix",
        provider="vochi",
        recording_url=lambda unique_id: f"https://example.test/recording/{unique_id}",
    )
    entries = [
        SimpleNamespace(
            started_at=dt.datetime(2026, 6, 22, 9, idx),
            caller_id=f"Client {idx}",
            destination="Support",
            duration_seconds=90,
            unique_id=f"call-{idx}",
            raw={"recording_url": f"https://example.test/permanent/call-{idx}"},
        )
        for idx in range(2)
    ]
    provider = SimpleNamespace(provider_name="gemini")
    cache_keys = [
        (
            "lix",
            entry.unique_id,
            "BATCH_PROMPT",
            "gemini",
            "models/gemini-test",
            "",
        )
        for entry in entries
    ]
    cache = _BulkOnlyCache(
        {
            key: AnalysisResult(
                text=json.dumps({"needs_follow_up": False, "reason": key[1]}),
                model="models/gemini-test",
                provider="gemini",
            )
            for key in cache_keys
        }
    )
    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        call_log_service=_CallLogService(entries),
        batch_language=SimpleNamespace(value="en"),
        batch_prompt_text="prompt",
        batch_prompt_key="BATCH_PROMPT",
        batch_model_key="models/gemini-test",
        ai_registry=_Registry(provider),
        analysis_service=SimpleNamespace(_cache=cache),
        email_report_service=None,
    )

    results = run_batch_process(deps, day, None, None, "", "lix")

    assert cache.requested_keys == cache_keys
    assert list(results["UniqueId"]) == ["call-0", "call-1"]
    assert list(results["Reason"]) == ["call-0", "call-1"]


def test_run_batch_skips_email_when_batch_failure_has_no_successes(monkeypatch) -> None:
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
    email_service = _RecordingEmailReportService()

    class FailingBatchRunner:
        def __init__(self, *, model):  # noqa: ANN001
            self.model = model

        def run_batch(self, *_args, **_kwargs):
            raise RuntimeError("batch timed out")

    monkeypatch.setattr("calls_analyser.runner.VertexBatchRunner", FailingBatchRunner)

    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        call_log_service=_PreparingCallLogService([entry]),
        batch_language=SimpleNamespace(value="en"),
        batch_prompt_text="prompt",
        batch_prompt_key="BATCH_PROMPT",
        batch_model_key="models/gemini-test",
        ai_registry=_Registry(provider),
        analysis_service=SimpleNamespace(_cache={}),
        email_report_service=email_service,
    )

    results = run_batch_process(deps, day, None, None, "", "lix")

    assert list(results["UniqueId"]) == ["call-1"]
    assert email_service.calls == []


def test_run_batch_records_usage_for_processed_vertex_batch_result(monkeypatch) -> None:
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
        raw={"recording_url": "https://example.test/permanent/call-1", "user": "agent"},
    )
    provider = SimpleNamespace(provider_name="gemini")
    usage_tracker = _RecordingUsageTracker()

    class SuccessfulBatchRunner:
        def __init__(self, *, model):  # noqa: ANN001
            self.model = model

        def run_batch_results(self, *_args, **_kwargs):
            return {
                "call-1": BatchAnalysisResult(
                    text=json.dumps({"needs_follow_up": False, "reason": ""}),
                    usage_metadata={
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 20,
                        "totalTokenCount": 120,
                        "thoughtsTokenCount": 0,
                    },
                )
            }

    monkeypatch.setattr("calls_analyser.runner.VertexBatchRunner", SuccessfulBatchRunner)

    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        call_log_service=_PreparingCallLogService([entry]),
        batch_language=SimpleNamespace(value="en"),
        batch_prompt_text="prompt",
        batch_prompt_key="BATCH_PROMPT",
        batch_model_key="models/gemini-test",
        ai_registry=_Registry(provider),
        analysis_service=SimpleNamespace(_cache={}),
        email_report_service=None,
        usage_tracker=usage_tracker,
    )

    run_batch_process(deps, day, None, None, "", "lix")

    assert len(usage_tracker.calls) == 1
    call = usage_tracker.calls[0]
    assert call["entry"] is entry
    assert call["tenant"] is tenant
    assert call["mode"] == "scheduler_batch"
    assert call["usage"].total_token_count == 120
