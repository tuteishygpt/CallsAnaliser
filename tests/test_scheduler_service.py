from __future__ import annotations

import datetime as dt
import threading
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from calls_analyser.runner import BatchExecutionContext
from calls_analyser.services.batch_results import BatchRunResult
from calls_analyser.services import scheduler


UTC = dt.timezone.utc


def _context(
    *,
    tenant_id: str = "tenant-a",
    prompt_key: str = "FOLLOW_UP",
    prompt_version: int = 7,
    model_key: str = "gemini-batch",
) -> BatchExecutionContext:
    return BatchExecutionContext(
        tenant=SimpleNamespace(tenant_id=tenant_id),
        prompt_key=prompt_key,
        batch_model_key=model_key,
        provider_name="gemini",
        batch_size=20,
        batch_language="auto",
        merged_prompt="frozen prompt",
        prompt_version=prompt_version,
        time_from=dt.time(9),
        time_to=dt.time(18),
        call_type="answered",
    )


class RecordingRepository:
    def __init__(self, *, claimed: bool = True, claim_error: Exception | None = None) -> None:
        self.claimed = claimed
        self.claim_error = claim_error
        self.claimed_keys = []
        self.finished = []

    def claim(self, key):
        self.claimed_keys.append(key)
        if self.claim_error is not None:
            raise self.claim_error
        return self.claimed

    def finish(self, key, result):
        self.finished.append((key, result))


class RecordingRunner:
    def __init__(self, result: BatchRunResult | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = []

    def __call__(
        self,
        deps,
        day,
        time_from_str,
        time_to_str,
        call_type_str,
        tenant_id_arg,
        *,
        execution_context,
    ):
        self.calls.append(
            {
                "deps": deps,
                "day": day,
                "time_from_str": time_from_str,
                "time_to_str": time_to_str,
                "call_type_str": call_type_str,
                "tenant_id_arg": tenant_id_arg,
                "execution_context": execution_context,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result


def test_scheduler_timezone_defaults_to_utc_and_parses_iana_zone() -> None:
    assert scheduler.scheduler_timezone(None) == ZoneInfo("UTC")
    assert scheduler.scheduler_timezone("") == ZoneInfo("UTC")
    assert scheduler.scheduler_timezone("Asia/Nicosia") == ZoneInfo("Asia/Nicosia")


def test_cron_scheduled_for_returns_latest_planned_minute_in_utc() -> None:
    timezone = ZoneInfo("Asia/Nicosia")

    delayed = scheduler.cron_scheduled_for(
        dt.datetime(2026, 7, 23, 2, 37, tzinfo=timezone),
        dt.time(2, 30),
    )
    before_today_slot = scheduler.cron_scheduled_for(
        dt.datetime(2026, 7, 23, 2, 29, tzinfo=timezone),
        dt.time(2, 30),
    )

    assert delayed == dt.datetime(2026, 7, 22, 23, 30, tzinfo=UTC)
    assert before_today_slot == dt.datetime(2026, 7, 21, 23, 30, tzinfo=UTC)


def test_cron_scheduled_for_uses_second_occurrence_after_dst_fallback() -> None:
    timezone = ZoneInfo("Europe/Berlin")
    now = dt.datetime(2026, 10, 25, 2, 45, tzinfo=timezone, fold=1)

    scheduled_for = scheduler.cron_scheduled_for(now, dt.time(2, 30))

    assert scheduled_for == dt.datetime(2026, 10, 25, 1, 30, tzinfo=UTC)


def test_interval_scheduled_for_floors_local_bucket_and_returns_utc() -> None:
    timezone = ZoneInfo("Asia/Nicosia")

    scheduled_for = scheduler.interval_scheduled_for(
        dt.datetime(2026, 7, 23, 2, 37, 59, tzinfo=timezone),
        90,
    )

    assert scheduled_for == dt.datetime(2026, 7, 22, 22, 30, tzinfo=UTC)


def test_interval_scheduled_for_uses_utc_day_bucket_boundaries() -> None:
    timezone = ZoneInfo("Asia/Nicosia")

    scheduled_for = scheduler.interval_scheduled_for(
        dt.datetime(2026, 1, 23, 2, 37, tzinfo=timezone),
        90,
    )

    assert scheduled_for == dt.datetime(2026, 1, 23, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "result",
    [
        BatchRunResult("success", 3, 3, 0, 1),
        BatchRunResult("partial", 3, 2, 1, 1),
        BatchRunResult("failed", 3, 0, 3, 0),
    ],
)
def test_claims_before_work_runs_yesterday_and_finishes_exact_result(
    monkeypatch,
    result: BatchRunResult,
) -> None:
    context = _context()
    monkeypatch.setattr(scheduler, "resolve_batch_execution_context", lambda *args, **kwargs: context)
    repository = RecordingRepository()
    runner = RecordingRunner(result=result)
    deps = SimpleNamespace(batch_prompt_key="mutable")
    now = dt.datetime(2026, 7, 23, 0, 15, tzinfo=ZoneInfo("Asia/Nicosia"))

    returned = scheduler.run_scheduled_batch_for_tenant(
        tenant_id="tenant-a",
        runtime_settings=SimpleNamespace(),
        scheduled_for=dt.datetime(2026, 7, 22, 21, 0, tzinfo=UTC),
        now=now,
        run_repository=repository,
        runner=runner,
        deps=deps,
    )

    assert returned is result
    assert runner.calls[0]["day"] == dt.date(2026, 7, 22)
    assert runner.calls[0]["tenant_id_arg"] == "tenant-a"
    assert runner.calls[0]["execution_context"] is context
    assert repository.finished == [(repository.claimed_keys[0], result)]
    key = repository.claimed_keys[0]
    assert (
        key.tenant_id,
        key.prompt_key,
        key.prompt_version,
        key.model_key,
        key.scheduled_for,
    ) == (
        "tenant-a",
        "FOLLOW_UP",
        7,
        "gemini-batch",
        dt.datetime(2026, 7, 22, 21, 0, tzinfo=UTC),
    )


def test_duplicate_claim_returns_none_without_runner_or_finish(monkeypatch) -> None:
    monkeypatch.setattr(
        scheduler,
        "resolve_batch_execution_context",
        lambda *args, **kwargs: _context(),
    )
    repository = RecordingRepository(claimed=False)
    runner = RecordingRunner(BatchRunResult("success", 0, 0, 0))

    returned = scheduler.run_scheduled_batch_for_tenant(
        tenant_id="tenant-a",
        runtime_settings=SimpleNamespace(),
        scheduled_for=dt.datetime(2026, 7, 23, tzinfo=UTC),
        now=dt.datetime(2026, 7, 23, tzinfo=UTC),
        run_repository=repository,
        runner=runner,
        deps=object(),
    )

    assert returned is None
    assert runner.calls == []
    assert repository.finished == []


def test_missing_repository_fails_closed_without_runner(monkeypatch) -> None:
    resolver_calls = []
    monkeypatch.setattr(
        scheduler,
        "resolve_batch_execution_context",
        lambda *args, **kwargs: resolver_calls.append(1) or _context(),
    )
    runner = RecordingRunner()

    with pytest.raises(RuntimeError, match="repository"):
        scheduler.run_scheduled_batch_for_tenant(
            tenant_id="tenant-a",
            runtime_settings=SimpleNamespace(),
            scheduled_for=dt.datetime(2026, 7, 23, tzinfo=UTC),
            now=dt.datetime(2026, 7, 23, tzinfo=UTC),
            run_repository=None,
            runner=runner,
            deps=object(),
        )

    assert resolver_calls == []
    assert runner.calls == []


def test_disabled_batch_fails_before_context_claim_or_runner(monkeypatch) -> None:
    resolver_calls = []
    monkeypatch.setattr(
        scheduler,
        "resolve_batch_execution_context",
        lambda *args, **kwargs: resolver_calls.append(1) or _context(),
    )
    repository = RecordingRepository()
    runner = RecordingRunner()

    with pytest.raises(RuntimeError, match="disabled"):
        scheduler.run_scheduled_batch_for_tenant(
            tenant_id="tenant-a",
            runtime_settings=SimpleNamespace(batch_enabled=False),
            scheduled_for=dt.datetime(2026, 7, 23, tzinfo=UTC),
            now=dt.datetime(2026, 7, 23, tzinfo=UTC),
            run_repository=repository,
            runner=runner,
            deps=object(),
        )

    assert resolver_calls == []
    assert repository.claimed_keys == []
    assert runner.calls == []


def test_naive_now_fails_before_context_claim_or_runner(monkeypatch) -> None:
    resolver_calls = []
    monkeypatch.setattr(
        scheduler,
        "resolve_batch_execution_context",
        lambda *args, **kwargs: resolver_calls.append(1) or _context(),
    )
    repository = RecordingRepository()
    runner = RecordingRunner()

    with pytest.raises(ValueError, match="now must be timezone-aware"):
        scheduler.run_scheduled_batch_for_tenant(
            tenant_id="tenant-a",
            runtime_settings=SimpleNamespace(),
            scheduled_for=dt.datetime(2026, 7, 23, tzinfo=UTC),
            now=dt.datetime(2026, 7, 23),
            run_repository=repository,
            runner=runner,
            deps=object(),
        )

    assert resolver_calls == []
    assert repository.claimed_keys == []
    assert runner.calls == []


def test_nonduplicate_claim_error_fails_closed_without_runner(monkeypatch) -> None:
    monkeypatch.setattr(
        scheduler,
        "resolve_batch_execution_context",
        lambda *args, **kwargs: _context(),
    )
    repository = RecordingRepository(claim_error=ConnectionError("guard unavailable"))
    runner = RecordingRunner()

    with pytest.raises(ConnectionError, match="guard unavailable"):
        scheduler.run_scheduled_batch_for_tenant(
            tenant_id="tenant-a",
            runtime_settings=SimpleNamespace(),
            scheduled_for=dt.datetime(2026, 7, 23, tzinfo=UTC),
            now=dt.datetime(2026, 7, 23, tzinfo=UTC),
            run_repository=repository,
            runner=runner,
            deps=object(),
        )

    assert runner.calls == []
    assert repository.finished == []


def test_runner_exception_finishes_failed_with_zero_counts_and_reraises(monkeypatch) -> None:
    monkeypatch.setattr(
        scheduler,
        "resolve_batch_execution_context",
        lambda *args, **kwargs: _context(),
    )
    repository = RecordingRepository()
    runner = RecordingRunner(error=RuntimeError("vertex exploded"))

    with pytest.raises(RuntimeError, match="vertex exploded"):
        scheduler.run_scheduled_batch_for_tenant(
            tenant_id="tenant-a",
            runtime_settings=SimpleNamespace(),
            scheduled_for=dt.datetime(2026, 7, 23, tzinfo=UTC),
            now=dt.datetime(2026, 7, 23, tzinfo=UTC),
            run_repository=repository,
            runner=runner,
            deps=object(),
        )

    assert repository.finished == [
        (
            repository.claimed_keys[0],
            BatchRunResult("failed", 0, 0, 0, 0),
        )
    ]


def test_claim_mutation_cannot_change_frozen_context_or_identity(monkeypatch) -> None:
    context = _context()
    monkeypatch.setattr(scheduler, "resolve_batch_execution_context", lambda *args, **kwargs: context)
    runtime_settings = SimpleNamespace(batch_model_key="gemini-batch")
    active_prompt = SimpleNamespace(body="frozen prompt", version=7)
    deps = SimpleNamespace(
        batch_prompt_key="FOLLOW_UP",
        prompt_service=SimpleNamespace(active_prompt=active_prompt),
    )

    class MutatingRepository(RecordingRepository):
        def claim(self, key):
            self.claimed_keys.append(key)
            runtime_settings.batch_model_key = "changed-model"
            deps.batch_prompt_key = "CHANGED_PROMPT"
            deps.prompt_service.active_prompt = SimpleNamespace(
                body="changed prompt",
                version=99,
            )
            return True

    repository = MutatingRepository()
    result = BatchRunResult("success", 1, 1, 0)
    runner = RecordingRunner(result)

    scheduler.run_scheduled_batch_for_tenant(
        tenant_id="requested-alias",
        runtime_settings=runtime_settings,
        scheduled_for=dt.datetime(2026, 7, 23, tzinfo=UTC),
        now=dt.datetime(2026, 7, 23, tzinfo=UTC),
        run_repository=repository,
        runner=runner,
        deps=deps,
    )

    key = repository.claimed_keys[0]
    assert (key.tenant_id, key.prompt_key, key.prompt_version, key.model_key) == (
        "tenant-a",
        "FOLLOW_UP",
        7,
        "gemini-batch",
    )
    assert runner.calls[0]["execution_context"] is context
    assert runner.calls[0]["execution_context"].prompt_key == key.prompt_key
    assert runner.calls[0]["execution_context"].prompt_version == key.prompt_version
    assert runner.calls[0]["execution_context"].batch_model_key == key.model_key
    assert runner.calls[0]["execution_context"].tenant.tenant_id == key.tenant_id
    assert runner.calls[0]["tenant_id_arg"] == key.tenant_id


def test_two_simultaneous_attempts_with_same_identity_run_once(monkeypatch) -> None:
    context = _context()
    monkeypatch.setattr(scheduler, "resolve_batch_execution_context", lambda *args, **kwargs: context)

    class AtomicRepository(RecordingRepository):
        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()
            self._claimed = False

        def claim(self, key):
            with self._lock:
                self.claimed_keys.append(key)
                if self._claimed:
                    return False
                self._claimed = True
                return True

    repository = AtomicRepository()
    runner = RecordingRunner(BatchRunResult("success", 0, 0, 0))
    barrier = threading.Barrier(2)
    returns = []

    def attempt() -> None:
        barrier.wait()
        returns.append(
            scheduler.run_scheduled_batch_for_tenant(
                tenant_id="tenant-a",
                runtime_settings=SimpleNamespace(),
                scheduled_for=dt.datetime(2026, 7, 23, tzinfo=UTC),
                now=dt.datetime(2026, 7, 23, tzinfo=UTC),
                run_repository=repository,
                runner=runner,
                deps=object(),
            )
        )

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(runner.calls) == 1
    assert len(repository.finished) == 1
    assert returns.count(None) == 1
