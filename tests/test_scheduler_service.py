from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from calls_analyser.services.scheduler import (
    ScheduledTenantFailure,
    ScheduledTenantSuccess,
    run_scheduled_batches_for_enabled_tenants,
)


class FakeTenantSettingsService:
    def __init__(self, enabled_tenants, runtime_settings_by_tenant=None) -> None:
        self._enabled_tenants = list(enabled_tenants)
        self._runtime_settings_by_tenant = runtime_settings_by_tenant or {}
        self.resolved_tenant_ids = []

    def list_scheduler_enabled_tenants(self):
        return list(self._enabled_tenants)

    def resolve(self, tenant_id):
        self.resolved_tenant_ids.append(tenant_id)
        return self._runtime_settings_by_tenant.get(tenant_id, SimpleNamespace())


class RecordingRunner:
    def __init__(self, failures=None) -> None:
        self.calls = []
        self._failures = failures or {}

    def __call__(self, deps, day, time_from_str, time_to_str, call_type_str, tenant_id_arg):
        self.calls.append(
            {
                "deps": deps,
                "day": day,
                "time_from_str": time_from_str,
                "time_to_str": time_to_str,
                "call_type_str": call_type_str,
                "tenant_id_arg": tenant_id_arg,
            }
        )
        if tenant_id_arg in self._failures:
            raise self._failures[tenant_id_arg]
        return f"result:{tenant_id_arg}"


def test_runs_runner_once_per_enabled_tenant_with_tenant_id_arg_set() -> None:
    day = dt.date(2026, 7, 6)
    deps = object()
    settings = FakeTenantSettingsService(["tenant-a", "tenant-b"])
    runner = RecordingRunner()

    summary = run_scheduled_batches_for_enabled_tenants(
        tenant_settings_service=settings,
        runner=runner,
        deps=deps,
        day=day,
    )

    assert [call["tenant_id_arg"] for call in runner.calls] == ["tenant-a", "tenant-b"]
    assert all(call["deps"] is deps for call in runner.calls)
    assert all(call["day"] == day for call in runner.calls)
    assert settings.resolved_tenant_ids == ["tenant-a", "tenant-b"]
    assert summary.successes == [
        ScheduledTenantSuccess(tenant_id="tenant-a", result="result:tenant-a"),
        ScheduledTenantSuccess(tenant_id="tenant-b", result="result:tenant-b"),
    ]
    assert summary.failures == []


def test_passes_per_tenant_filters_and_runtime_fallbacks() -> None:
    day = dt.date(2026, 7, 6)
    settings = FakeTenantSettingsService(
        ["tenant-a", "tenant-b", "tenant-c"],
        {
            "tenant-a": SimpleNamespace(
                scheduler_filters={
                    "time_from": "09:00",
                    "time_to": "18:00",
                    "call_type": "ANSWERED",
                }
            ),
            "tenant-b": SimpleNamespace(
                scheduler_filters={},
                filter_time_from="10:00",
                filter_time_to="17:30",
                filter_call_type="MISSED",
            ),
            "tenant-c": SimpleNamespace(),
        },
    )
    runner = RecordingRunner()

    run_scheduled_batches_for_enabled_tenants(
        tenant_settings_service=settings,
        runner=runner,
        deps=object(),
        day=day,
    )

    assert [
        (call["time_from_str"], call["time_to_str"], call["call_type_str"])
        for call in runner.calls
    ] == [
        ("09:00", "18:00", "ANSWERED"),
        ("10:00", "17:30", "MISSED"),
        (None, None, ""),
    ]


def test_records_failure_for_one_tenant_and_continues_to_another() -> None:
    day = dt.date(2026, 7, 6)
    settings = FakeTenantSettingsService(["tenant-bad", "tenant-good"])
    runner = RecordingRunner({"tenant-bad": RuntimeError("batch failed")})

    summary = run_scheduled_batches_for_enabled_tenants(
        tenant_settings_service=settings,
        runner=runner,
        deps=object(),
        day=day,
    )

    assert [call["tenant_id_arg"] for call in runner.calls] == ["tenant-bad", "tenant-good"]
    assert summary.successes == [
        ScheduledTenantSuccess(tenant_id="tenant-good", result="result:tenant-good"),
    ]
    assert summary.failures == [
        ScheduledTenantFailure(
            tenant_id="tenant-bad",
            error="batch failed",
            exception_type="RuntimeError",
        ),
    ]


def test_returns_empty_summary_when_no_tenants_enabled() -> None:
    settings = FakeTenantSettingsService([])
    runner = RecordingRunner()

    summary = run_scheduled_batches_for_enabled_tenants(
        tenant_settings_service=settings,
        runner=runner,
        deps=object(),
        day=dt.date(2026, 7, 6),
    )

    assert runner.calls == []
    assert settings.resolved_tenant_ids == []
    assert summary.successes == []
    assert summary.failures == []
