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


class _RecordingRegistry:
    def __init__(self, providers) -> None:  # noqa: ANN001
        self._providers = providers
        self.requested_keys = []

    def get(self, key: str):
        self.requested_keys.append(key)
        return self._providers.get(key)


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


class _TenantSettingsService:
    def __init__(self, runtime_settings) -> None:  # noqa: ANN001
        self._runtime_settings = runtime_settings
        self.resolved_tenant_ids = []

    def resolve(self, tenant_id: str):
        self.resolved_tenant_ids.append(tenant_id)
        return self._runtime_settings


class _CapturingCallLogService:
    def __init__(self, entries=None) -> None:  # noqa: ANN001
        self._entries = list(entries or [])
        self.calls = []

    def list_calls(self, day, tenant, **kwargs):  # noqa: ANN001
        self.calls.append((day, tenant, kwargs))
        return list(self._entries)


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


class _PromptService:
    def get_prompt(self, _key: str, tenant_id=None):  # noqa: ANN001
        return SimpleNamespace(version=1)


def test_run_batch_uses_tenant_prompt_body_and_version_for_vertex_batch(monkeypatch) -> None:
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
    cache = {}
    usage_tracker = _RecordingUsageTracker()
    captured_prompts = []

    class TenantPromptService:
        def __init__(self) -> None:
            self.calls = []

        def get_prompt(self, key: str, tenant_id=None):  # noqa: ANN001
            self.calls.append((key, tenant_id))
            return SimpleNamespace(
                body="tenant-specific batch prompt",
                version=42,
            )

    class SuccessfulBatchRunner:
        def __init__(self, *, model):  # noqa: ANN001
            self.model = model

        def run_batch_results(self, _tasks, prompt, *, chunk_size):  # noqa: ANN001
            captured_prompts.append(prompt)
            return {
                "call-1": BatchAnalysisResult(
                    text=json.dumps({"needs_follow_up": False, "reason": "No action"}),
                    usage_metadata={
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 20,
                        "totalTokenCount": 120,
                    },
                )
            }

    monkeypatch.setattr("calls_analyser.runner.VertexBatchRunner", SuccessfulBatchRunner)

    prompt_service = TenantPromptService()
    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        call_log_service=_PreparingCallLogService([entry]),
        batch_language=SimpleNamespace(value="en"),
        batch_prompt_text="global batch prompt",
        batch_prompt_key="BATCH_PROMPT",
        batch_model_key="models/gemini-test",
        ai_registry=_Registry(provider),
        prompt_service=prompt_service,
        analysis_service=SimpleNamespace(_cache=cache),
        email_report_service=None,
        usage_tracker=usage_tracker,
    )

    run_batch_process(deps, day, None, None, "", "lix")

    expected_cache_key = (
        "lix",
        "call-1",
        "BATCH_PROMPT",
        42,
        "gemini",
        "models/gemini-test",
        "",
    )
    assert prompt_service.calls == [("BATCH_PROMPT", "lix")]
    assert len(captured_prompts) == 1
    assert "tenant-specific batch prompt" in captured_prompts[0]
    assert "global batch prompt" not in captured_prompts[0]
    assert expected_cache_key in cache
    assert usage_tracker.calls[0]["cache_key"] == expected_cache_key


def test_run_batch_falls_back_to_global_prompt_when_tenant_prompt_body_is_empty(monkeypatch) -> None:
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
    cache = {}
    captured_prompts = []

    class TenantPromptService:
        def __init__(self) -> None:
            self.calls = []

        def get_prompt(self, key: str, tenant_id=None):  # noqa: ANN001
            self.calls.append((key, tenant_id))
            return SimpleNamespace(
                body="   ",
                version=99,
            )

    class SuccessfulBatchRunner:
        def __init__(self, *, model):  # noqa: ANN001
            self.model = model

        def run_batch_results(self, _tasks, prompt, *, chunk_size):  # noqa: ANN001
            captured_prompts.append(prompt)
            return {
                "call-1": BatchAnalysisResult(
                    text=json.dumps({"needs_follow_up": False, "reason": "No action"}),
                    usage_metadata=None,
                )
            }

    monkeypatch.setattr("calls_analyser.runner.VertexBatchRunner", SuccessfulBatchRunner)

    prompt_service = TenantPromptService()
    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        call_log_service=_PreparingCallLogService([entry]),
        batch_language=SimpleNamespace(value="en"),
        batch_prompt_text="global batch prompt",
        batch_prompt_key="BATCH_PROMPT",
        batch_model_key="models/gemini-test",
        ai_registry=_Registry(provider),
        prompt_service=prompt_service,
        analysis_service=SimpleNamespace(_cache=cache),
        email_report_service=None,
    )

    run_batch_process(deps, day, None, None, "", "lix")

    expected_cache_key = (
        "lix",
        "call-1",
        "BATCH_PROMPT",
        99,
        "gemini",
        "models/gemini-test",
        "",
    )
    assert prompt_service.calls == [("BATCH_PROMPT", "lix")]
    assert len(captured_prompts) == 1
    assert "global batch prompt" in captured_prompts[0]
    assert "   " not in captured_prompts[0]
    assert expected_cache_key in cache


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
        1,
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
        prompt_service=_PromptService(),
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
            1,
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
        prompt_service=_PromptService(),
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
        prompt_service=_PromptService(),
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
                    text=json.dumps({"needs_follow_up": False, "reason": "No action"}),
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
        prompt_service=_PromptService(),
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


def test_run_batch_uses_tenant_runtime_model_and_batch_size_for_processed_result(monkeypatch) -> None:
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
    registry = _RecordingRegistry({"models/gemini-tenant": provider})
    cache = {}
    runner_models = []
    runner_chunk_sizes = []

    class SuccessfulBatchRunner:
        def __init__(self, *, model):  # noqa: ANN001
            runner_models.append(model)

        def run_batch_results(self, tasks, _prompt, *, chunk_size):  # noqa: ANN001
            runner_chunk_sizes.append(chunk_size)
            assert [task.key for task in tasks] == ["call-1"]
            return {
                "call-1": BatchAnalysisResult(
                    text=json.dumps({"needs_follow_up": False, "reason": "No action"}),
                    usage_metadata=None,
                )
            }

    monkeypatch.setattr("calls_analyser.runner.VertexBatchRunner", SuccessfulBatchRunner)

    runtime_settings = SimpleNamespace(
        batch_enabled=True,
        batch_model_key="models/gemini-tenant",
        batch_language_code="",
        batch_size=7,
        scheduler_filters={},
    )
    tenant_settings_service = _TenantSettingsService(runtime_settings)
    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        call_log_service=_PreparingCallLogService([entry]),
        batch_language=SimpleNamespace(value="en"),
        batch_prompt_text="prompt",
        batch_prompt_key="BATCH_PROMPT",
        batch_model_key="models/gemini-default",
        ai_registry=registry,
        prompt_service=_PromptService(),
        analysis_service=SimpleNamespace(_cache=cache),
        email_report_service=None,
        tenant_settings_service=tenant_settings_service,
    )

    results = run_batch_process(deps, day, None, None, "", "lix")

    expected_cache_key = (
        "lix",
        "call-1",
        "BATCH_PROMPT",
        1,
        "gemini",
        "models/gemini-tenant",
        "",
    )
    assert tenant_settings_service.resolved_tenant_ids == ["lix"]
    assert registry.requested_keys == ["models/gemini-tenant"]
    assert runner_models == ["models/gemini-tenant"]
    assert runner_chunk_sizes == [7]
    assert expected_cache_key in cache
    assert cache[expected_cache_key].model == "models/gemini-tenant"
    assert list(results["UniqueId"]) == ["call-1"]


def test_run_batch_uses_tenant_scheduler_filters_when_explicit_args_are_empty() -> None:
    day = dt.date(2026, 6, 22)
    tenant = SimpleNamespace(
        tenant_id="lix",
        provider="vochi",
        recording_url=lambda unique_id: f"https://example.test/recording/{unique_id}",
    )
    call_log_service = _CapturingCallLogService()
    runtime_settings = SimpleNamespace(
        batch_enabled=True,
        batch_model_key="",
        batch_language_code="",
        batch_size=25,
        scheduler_filters={
            "time_from": "09:30",
            "time_to": "17:45",
            "call_type": "Outbound",
        },
    )
    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        call_log_service=call_log_service,
        batch_language=SimpleNamespace(value="en"),
        batch_prompt_text="prompt",
        batch_prompt_key="BATCH_PROMPT",
        batch_model_key="models/gemini-test",
        ai_registry=_Registry(SimpleNamespace(provider_name="gemini")),
        prompt_service=_PromptService(),
        analysis_service=SimpleNamespace(_cache={}),
        email_report_service=None,
        tenant_settings_service=_TenantSettingsService(runtime_settings),
    )

    run_batch_process(deps, day, "", None, "", "lix")

    assert len(call_log_service.calls) == 1
    _day, _tenant, filters = call_log_service.calls[0]
    assert filters == {
        "time_from": dt.time(9, 30),
        "time_to": dt.time(17, 45),
        "call_type": 1,
    }


def test_run_batch_skips_processing_when_tenant_settings_disable_batch() -> None:
    day = dt.date(2026, 6, 22)
    tenant = SimpleNamespace(
        tenant_id="lix",
        provider="vochi",
        recording_url=lambda unique_id: f"https://example.test/recording/{unique_id}",
    )
    call_log_service = _CapturingCallLogService()
    runtime_settings = SimpleNamespace(
        batch_enabled=False,
        batch_model_key="models/gemini-tenant",
        batch_language_code="en",
        batch_size=7,
        scheduler_filters={},
    )
    tenant_settings_service = _TenantSettingsService(runtime_settings)
    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        call_log_service=call_log_service,
        batch_language=SimpleNamespace(value="en"),
        batch_prompt_text="prompt",
        batch_prompt_key="BATCH_PROMPT",
        batch_model_key="models/gemini-test",
        ai_registry=_Registry(SimpleNamespace(provider_name="gemini")),
        prompt_service=_PromptService(),
        analysis_service=SimpleNamespace(_cache={}),
        email_report_service=None,
        tenant_settings_service=tenant_settings_service,
    )

    result = run_batch_process(deps, day, None, None, "", "lix")

    assert result is None
    assert tenant_settings_service.resolved_tenant_ids == ["lix"]
    assert call_log_service.calls == []


def test_run_batch_enforces_verification_and_emails_audit_columns(monkeypatch) -> None:
    day = dt.date(2026, 6, 22)
    tenant = SimpleNamespace(
        tenant_id="lix", provider="vochi",
        recording_url=lambda unique_id: f"https://example.test/{unique_id}",
    )
    entry = SimpleNamespace(
        started_at=dt.datetime(2026, 6, 22, 9), caller_id="Client",
        destination="Support", duration_seconds=90, unique_id="call-1",
        raw={"recording_url": "https://example.test/call-1"},
    )
    rounds = []

    class TwoRoundRunner:
        def __init__(self, *, model):  # noqa: ANN001
            self.model = model

        def run_batch_results(self, tasks, _prompt, *, chunk_size):  # noqa: ANN001
            rounds.append((self.model, [task.key for task in tasks], chunk_size))
            decision = self.model == "verify-model"
            return {
                "call-1": BatchAnalysisResult(
                    text=json.dumps({
                        "needs_follow_up": not decision,
                        "reason": "primary reason" if not decision else "cleared",
                    }),
                    usage_metadata=None,
                )
            }

    monkeypatch.setattr("calls_analyser.runner.VertexBatchRunner", TwoRoundRunner)
    email = _RecordingEmailReportService()
    settings = SimpleNamespace(
        batch_enabled=True, batch_model_key="primary-model", batch_language_code="en",
        batch_size=7, scheduler_filters={}, follow_up_verification_mode="enforce",
        follow_up_verification_model_key="verify-model",
        follow_up_verification_prompt_key="VERIFY_PROMPT",
    )
    prompt_service = SimpleNamespace(get_prompt=lambda key, tenant_id=None: SimpleNamespace(
        key=key, version=1, body=f"{key} body",
    ))
    deps = SimpleNamespace(
        project_imports_available=True,
        batch_params=SimpleNamespace(enable_gemini_batch=True, batch_size=25),
        tenant_service=_TenantService(tenant),
        tenant_settings_service=_TenantSettingsService(settings),
        call_log_service=_PreparingCallLogService([entry]),
        batch_language=SimpleNamespace(value="en"), batch_prompt_text="primary",
        batch_prompt_key="BATCH_PROMPT", batch_model_key="default-model",
        ai_registry=_RecordingRegistry({
            "primary-model": SimpleNamespace(provider_name="gemini"),
            "verify-model": SimpleNamespace(provider_name="gemini"),
        }),
        prompt_service=prompt_service, analysis_service=SimpleNamespace(_cache={}),
        email_report_service=email,
    )

    results = run_batch_process(deps, day, None, None, "", "lix")

    assert rounds == [
        ("primary-model", ["call-1"], 7),
        ("verify-model", ["call-1"], 7),
    ]
    assert results.loc[0, "Needs follow-up"] == "No"
    assert results.loc[0, "Initial needs follow-up"] == "Yes"
    assert results.loc[0, "Verification needs follow-up"] == "No"
    assert results.loc[0, "Verification status"] == "complete"
    assert len(email.calls) == 1
