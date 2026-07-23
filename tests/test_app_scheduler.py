from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import app


class RecordingScheduler:
    def __init__(self) -> None:
        self.jobs: list[dict[str, object]] = []
        self.start_calls = 0

    def add_job(self, job, trigger, **kwargs):  # noqa: ANN001
        self.jobs.append({"job": job, "trigger": trigger, **kwargs})

    def start(self) -> None:
        self.start_calls += 1


class FakeTenantSettingsService:
    def __init__(self, settings_by_tenant) -> None:  # noqa: ANN001
        self.settings_by_tenant = dict(settings_by_tenant)
        self.list_calls = 0
        self.resolve_calls: list[str] = []

    def list_scheduler_enabled_tenants(self):
        self.list_calls += 1
        return list(self.settings_by_tenant)

    def resolve(self, tenant_id: str):
        self.resolve_calls.append(tenant_id)
        return self.settings_by_tenant[tenant_id]


def _settings(**overrides) -> SimpleNamespace:
    values = {
        "scheduler_mode": "cron",
        "scheduler_cron_time": "02:30",
        "scheduler_interval_minutes": 45,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_registers_one_tenant_specific_job_for_each_enabled_tenant(monkeypatch) -> None:
    tenant_settings = FakeTenantSettingsService(
        {
            "tenant-cron": _settings(),
            "tenant-interval": _settings(
                scheduler_mode="interval",
                scheduler_interval_minutes=17,
            ),
        }
    )
    deps = SimpleNamespace(
        tenant_settings_service=tenant_settings,
        scheduler_run_repository=object(),
    )
    scheduler = RecordingScheduler()
    monkeypatch.setenv("SCHEDULER_TIMEZONE", "Europe/Nicosia")

    registered = app._register_scheduler_jobs_if_available(
        scheduler,
        deps,
        runner=object(),
    )

    assert registered is True
    assert scheduler.start_calls == 1
    assert tenant_settings.list_calls == 1
    assert tenant_settings.resolve_calls == ["tenant-cron", "tenant-interval"]
    assert len(scheduler.jobs) == 2
    cron_job, interval_job = scheduler.jobs
    assert cron_job == {
        "job": cron_job["job"],
        "trigger": "cron",
        "hour": 2,
        "minute": 30,
        "timezone": ZoneInfo("Europe/Nicosia"),
        "id": "scheduler:tenant-cron",
        "replace_existing": True,
        "max_instances": 1,
    }
    assert interval_job == {
        "job": interval_job["job"],
        "trigger": "interval",
        "minutes": 17,
        "timezone": ZoneInfo("Europe/Nicosia"),
        "id": "scheduler:tenant-interval",
        "replace_existing": True,
        "max_instances": 1,
    }


def test_delayed_cron_job_uses_planned_minute_and_frozen_runtime_settings(
    monkeypatch,
) -> None:
    runtime_settings = _settings(scheduler_cron_time="02:30")
    tenant_settings = FakeTenantSettingsService({"tenant-a": runtime_settings})
    repository = object()
    runner = object()
    deps = SimpleNamespace(
        tenant_settings_service=tenant_settings,
        scheduler_run_repository=repository,
    )
    scheduler = RecordingScheduler()
    calls: list[dict[str, object]] = []
    delayed_now = dt.datetime(2026, 7, 23, 2, 37, tzinfo=ZoneInfo("Europe/Nicosia"))

    monkeypatch.setenv("SCHEDULER_TIMEZONE", "Europe/Nicosia")
    monkeypatch.setattr(app, "_scheduler_now", lambda timezone: delayed_now)
    monkeypatch.setattr(
        app,
        "run_scheduled_batch_for_tenant",
        lambda **kwargs: calls.append(kwargs),
    )

    app._register_scheduler_jobs_if_available(scheduler, deps, runner=runner)
    scheduler.jobs[0]["job"]()

    assert calls == [
        {
            "tenant_id": "tenant-a",
            "runtime_settings": runtime_settings,
            "scheduled_for": dt.datetime(
                2026,
                7,
                22,
                23,
                30,
                tzinfo=dt.timezone.utc,
            ),
            "now": delayed_now,
            "run_repository": repository,
            "runner": runner,
            "deps": deps,
        }
    ]


def test_same_mode_tenant_closures_keep_their_own_tenant_settings_and_trigger(
    monkeypatch,
) -> None:
    settings_a = _settings(scheduler_cron_time="01:15")
    settings_b = _settings(scheduler_cron_time="04:45")
    deps = SimpleNamespace(
        tenant_settings_service=FakeTenantSettingsService(
            {"tenant-a": settings_a, "tenant-b": settings_b}
        ),
        scheduler_run_repository=object(),
    )
    scheduler = RecordingScheduler()
    now = dt.datetime(2026, 7, 23, 5, 0, tzinfo=dt.timezone.utc)
    calls = []
    monkeypatch.delenv("SCHEDULER_TIMEZONE", raising=False)
    monkeypatch.setattr(app, "_scheduler_now", lambda timezone: now)
    monkeypatch.setattr(
        app,
        "run_scheduled_batch_for_tenant",
        lambda **kwargs: calls.append(kwargs),
    )

    app._register_scheduler_jobs_if_available(scheduler, deps, runner="runner")
    for registered_job in scheduler.jobs:
        registered_job["job"]()

    assert [job["id"] for job in scheduler.jobs] == [
        "scheduler:tenant-a",
        "scheduler:tenant-b",
    ]
    assert [(job["hour"], job["minute"]) for job in scheduler.jobs] == [
        (1, 15),
        (4, 45),
    ]
    assert [call["tenant_id"] for call in calls] == ["tenant-a", "tenant-b"]
    assert [
        call["runtime_settings"].scheduler_cron_time for call in calls
    ] == ["01:15", "04:45"]
    assert [call["scheduled_for"].time() for call in calls] == [
        dt.time(1, 15),
        dt.time(4, 45),
    ]


def test_registration_copies_runtime_settings_before_source_mutation(monkeypatch) -> None:
    source_settings = _settings(
        scheduler_cron_time="02:30",
        batch_model_key="startup-model",
    )
    deps = SimpleNamespace(
        tenant_settings_service=FakeTenantSettingsService(
            {"tenant-a": source_settings}
        ),
        scheduler_run_repository=object(),
    )
    scheduler = RecordingScheduler()
    now = dt.datetime(2026, 7, 23, 2, 37, tzinfo=dt.timezone.utc)
    calls = []
    monkeypatch.delenv("SCHEDULER_TIMEZONE", raising=False)
    monkeypatch.setattr(app, "_scheduler_now", lambda timezone: now)
    monkeypatch.setattr(
        app,
        "run_scheduled_batch_for_tenant",
        lambda **kwargs: calls.append(kwargs),
    )

    app._register_scheduler_jobs_if_available(scheduler, deps, runner="runner")
    source_settings.scheduler_cron_time = "05:45"
    source_settings.batch_model_key = "mutated-model"
    scheduler.jobs[0]["job"]()

    captured_settings = calls[0]["runtime_settings"]
    assert captured_settings is not source_settings
    assert captured_settings.scheduler_cron_time == "02:30"
    assert captured_settings.batch_model_key == "startup-model"
    assert calls[0]["scheduled_for"] == dt.datetime(
        2026,
        7,
        23,
        2,
        30,
        tzinfo=dt.timezone.utc,
    )


def test_interval_job_computes_current_planned_bucket(monkeypatch) -> None:
    runtime_settings = _settings(
        scheduler_mode="interval",
        scheduler_interval_minutes=15,
    )
    deps = SimpleNamespace(
        tenant_settings_service=FakeTenantSettingsService({"tenant-a": runtime_settings}),
        scheduler_run_repository=object(),
    )
    scheduler = RecordingScheduler()
    now = dt.datetime(2026, 7, 23, 2, 37, tzinfo=dt.timezone.utc)
    calls = []
    monkeypatch.delenv("SCHEDULER_TIMEZONE", raising=False)
    monkeypatch.setattr(app, "_scheduler_now", lambda timezone: now)
    monkeypatch.setattr(
        app,
        "run_scheduled_batch_for_tenant",
        lambda **kwargs: calls.append(kwargs),
    )

    app._register_scheduler_jobs_if_available(scheduler, deps, runner="runner")
    scheduler.jobs[0]["job"]()

    assert calls[0]["scheduled_for"] == dt.datetime(
        2026,
        7,
        23,
        2,
        30,
        tzinfo=dt.timezone.utc,
    )


def test_no_enabled_tenants_does_not_start_scheduler() -> None:
    tenant_settings = FakeTenantSettingsService({})
    scheduler = RecordingScheduler()

    registered = app._register_scheduler_jobs_if_available(
        scheduler,
        SimpleNamespace(
            tenant_settings_service=tenant_settings,
            scheduler_run_repository=object(),
        ),
        runner=object(),
    )

    assert registered is False
    assert scheduler.start_calls == 0
    assert scheduler.jobs == []
    assert tenant_settings.list_calls == 1


def test_missing_run_repository_fails_closed_without_listing_or_registering() -> None:
    tenant_settings = FakeTenantSettingsService({"tenant-a": _settings()})
    scheduler = RecordingScheduler()

    registered = app._register_scheduler_jobs_if_available(
        scheduler,
        SimpleNamespace(
            tenant_settings_service=tenant_settings,
            scheduler_run_repository=None,
        ),
        runner=object(),
    )

    assert registered is False
    assert scheduler.start_calls == 0
    assert scheduler.jobs == []
    assert tenant_settings.list_calls == 0
