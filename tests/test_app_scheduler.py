from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import app


class FrozenDate(dt.date):
    @classmethod
    def today(cls) -> dt.date:
        return cls(2026, 7, 7)


def _batch_params(**overrides) -> SimpleNamespace:
    values = {
        "filter_time_from": "09:00",
        "filter_time_to": "18:00",
        "filter_call_type": "missed",
        "scheduler_enabled": True,
        "scheduler_mode": "cron",
        "scheduler_cron_time": "02:30",
        "scheduler_interval_minutes": 45,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RecordingScheduler:
    def __init__(self) -> None:
        self.jobs = []
        self.started = False

    def add_job(self, job, trigger, **kwargs):  # noqa: ANN001
        self.jobs.append({"job": job, "trigger": trigger, **kwargs})

    def start(self) -> None:
        self.started = True


class FakeTenantSettingsService:
    def __init__(self, enabled_tenants) -> None:  # noqa: ANN001
        self._enabled_tenants = list(enabled_tenants)
        self.list_calls = 0

    def list_scheduler_enabled_tenants(self):
        self.list_calls += 1
        return list(self._enabled_tenants)


def test_scheduled_job_uses_multi_tenant_scheduler_service(monkeypatch) -> None:
    tenant_settings_service = object()
    deps = SimpleNamespace(
        batch_params=_batch_params(),
        tenant_settings_service=tenant_settings_service,
    )
    calls = []

    def fake_run_scheduled_batches_for_enabled_tenants(
        tenant_settings_service,
        runner,
        deps,
        day,
    ):  # noqa: ANN001
        calls.append(
            {
                "tenant_settings_service": tenant_settings_service,
                "runner": runner,
                "deps": deps,
                "day": day,
            }
        )
        return SimpleNamespace(
            successes=[object(), object()],
            failures=[
                SimpleNamespace(
                    tenant_id="tenant-bad",
                    exception_type="RuntimeError",
                    error="batch failed",
                )
            ],
        )

    monkeypatch.setattr(app, "deps", deps)
    monkeypatch.setattr(app.datetime, "date", FrozenDate)
    monkeypatch.setattr(
        app,
        "run_scheduled_batches_for_enabled_tenants",
        fake_run_scheduled_batches_for_enabled_tenants,
        raising=False,
    )

    app.run_scheduled_job()

    assert calls == [
        {
            "tenant_settings_service": tenant_settings_service,
            "runner": app.daily_runner.run_batch_process,
            "deps": deps,
            "day": dt.date(2026, 7, 6),
        }
    ]


def test_scheduled_job_preserves_legacy_runner_call_without_tenant_settings_service(
    monkeypatch,
) -> None:
    deps = SimpleNamespace(batch_params=_batch_params(), tenant_settings_service=None)
    calls = []

    def fake_run_batch_process(
        deps,
        day,
        time_from_str,
        time_to_str,
        call_type_str,
        tenant_id_arg,
    ):  # noqa: ANN001
        calls.append(
            {
                "deps": deps,
                "day": day,
                "time_from_str": time_from_str,
                "time_to_str": time_to_str,
                "call_type_str": call_type_str,
                "tenant_id_arg": tenant_id_arg,
            }
        )

    monkeypatch.setattr(app, "deps", deps)
    monkeypatch.setattr(app.datetime, "date", FrozenDate)
    monkeypatch.setattr(app.daily_runner, "run_batch_process", fake_run_batch_process)

    app.run_scheduled_job()

    assert calls == [
        {
            "deps": deps,
            "day": dt.date(2026, 7, 6),
            "time_from_str": "09:00",
            "time_to_str": "18:00",
            "call_type_str": "missed",
            "tenant_id_arg": None,
        }
    ]


def test_scheduler_registration_uses_tenant_enabled_list_when_global_disabled() -> None:
    tenant_settings_service = FakeTenantSettingsService(["tenant-a"])
    deps = SimpleNamespace(
        batch_params=_batch_params(scheduler_enabled=False),
        tenant_settings_service=tenant_settings_service,
    )
    scheduler = RecordingScheduler()
    job = object()

    registered = app._register_scheduler_job_if_available(scheduler, deps, job)

    assert registered is True
    assert scheduler.started is True
    assert scheduler.jobs == [
        {
            "job": job,
            "trigger": "cron",
            "hour": 2,
            "minute": 30,
        }
    ]
    assert tenant_settings_service.list_calls == 1


def test_scheduler_registration_skips_multi_tenant_scheduler_when_no_tenants_enabled() -> None:
    tenant_settings_service = FakeTenantSettingsService([])
    deps = SimpleNamespace(
        batch_params=_batch_params(scheduler_enabled=True),
        tenant_settings_service=tenant_settings_service,
    )
    scheduler = RecordingScheduler()

    registered = app._register_scheduler_job_if_available(scheduler, deps, object())

    assert registered is False
    assert scheduler.started is False
    assert scheduler.jobs == []
    assert tenant_settings_service.list_calls == 1


def test_scheduler_registration_without_tenant_settings_service_follows_global_switch() -> None:
    enabled_scheduler = RecordingScheduler()
    disabled_scheduler = RecordingScheduler()

    enabled = app._register_scheduler_job_if_available(
        enabled_scheduler,
        SimpleNamespace(
            batch_params=_batch_params(scheduler_enabled=True),
            tenant_settings_service=None,
        ),
        object(),
    )
    disabled = app._register_scheduler_job_if_available(
        disabled_scheduler,
        SimpleNamespace(
            batch_params=_batch_params(scheduler_enabled=False),
            tenant_settings_service=None,
        ),
        object(),
    )

    assert enabled is True
    assert enabled_scheduler.started is True
    assert len(enabled_scheduler.jobs) == 1
    assert disabled is False
    assert disabled_scheduler.started is False
    assert disabled_scheduler.jobs == []
