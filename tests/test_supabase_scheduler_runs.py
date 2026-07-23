from __future__ import annotations

import datetime as dt
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from postgrest.exceptions import APIError

from calls_analyser.adapters.storage.supabase_scheduler_runs import (
    SchedulerRunKey,
    SupabaseSchedulerRunRepository,
)
from calls_analyser.services.batch_results import BatchRunResult


class _Query:
    def __init__(self, table: "_FakeSchedulerRunsTable", operation: str, payload: dict) -> None:
        self._table = table
        self._operation = operation
        self._payload = payload
        self._filters: list[tuple[str, object]] = []

    def eq(self, column: str, value: object) -> "_Query":
        self._filters.append((column, value))
        return self

    def execute(self) -> SimpleNamespace:
        if self._operation == "insert":
            self._table.execute_insert(self._payload)
            return SimpleNamespace(data=[])
        else:
            self._table.updates.append((self._payload, self._filters))
            return SimpleNamespace(data=self._table.update_response_data)


class _FakeSchedulerRunsTable:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.updates: list[tuple[dict, list[tuple[str, object]]]] = []
        self.update_response_data: list[dict] = [{"tenant_id": "tenant-a"}]
        self._lock = threading.Lock()

    def insert(self, payload: dict) -> _Query:
        return _Query(self, "insert", payload)

    def update(self, payload: dict) -> _Query:
        return _Query(self, "update", payload)

    def execute_insert(self, payload: dict) -> None:
        identity = (
            payload["tenant_id"],
            payload["scheduled_for"],
            payload["prompt_key"],
            payload["prompt_version"],
            payload["model_key"],
        )
        with self._lock:
            if any(
                (
                    row["tenant_id"],
                    row["scheduled_for"],
                    row["prompt_key"],
                    row["prompt_version"],
                    row["model_key"],
                )
                == identity
                for row in self.rows
            ):
                raise APIError(
                    {
                        "code": "23505",
                        "message": "duplicate key value violates unique constraint",
                        "details": None,
                        "hint": None,
                    }
                )
            self.rows.append(dict(payload))


class _FakeClient:
    def __init__(self, table: _FakeSchedulerRunsTable) -> None:
        self.scheduler_runs = table

    def table(self, name: str) -> _FakeSchedulerRunsTable:
        assert name == "scheduler_runs"
        return self.scheduler_runs


@pytest.fixture
def key() -> SchedulerRunKey:
    return SchedulerRunKey(
        tenant_id="tenant-a",
        scheduled_for=dt.datetime(
            2026, 7, 23, 2, 30, tzinfo=dt.timezone(dt.timedelta(hours=3))
        ),
        prompt_key="follow-up",
        prompt_version=7,
        model_key="gemini-2.5-pro",
    )


def _repository(table: _FakeSchedulerRunsTable) -> SupabaseSchedulerRunRepository:
    with patch(
        "calls_analyser.adapters.storage.supabase_scheduler_runs.create_client",
        return_value=_FakeClient(table),
    ):
        return SupabaseSchedulerRunRepository("https://example.supabase.co", "service-key")


def test_claim_inserts_one_running_row_with_utc_iso_timestamp(key: SchedulerRunKey) -> None:
    table = _FakeSchedulerRunsTable()
    repository = _repository(table)

    assert repository.claim(key) is True

    assert table.rows == [
        {
            "tenant_id": "tenant-a",
            "scheduled_for": "2026-07-22T23:30:00+00:00",
            "prompt_key": "follow-up",
            "prompt_version": 7,
            "model_key": "gemini-2.5-pro",
            "status": "running",
        }
    ]


def test_claim_rejects_naive_scheduled_for() -> None:
    table = _FakeSchedulerRunsTable()
    repository = _repository(table)
    naive_key = SchedulerRunKey(
        tenant_id="tenant-a",
        scheduled_for=dt.datetime(2026, 7, 23, 2, 30),
        prompt_key="follow-up",
        prompt_version=7,
        model_key="gemini-2.5-pro",
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        repository.claim(naive_key)

    assert table.rows == []


def test_claim_returns_false_only_for_postgres_unique_violation(
    key: SchedulerRunKey,
) -> None:
    table = _FakeSchedulerRunsTable()
    repository = _repository(table)

    assert repository.claim(key) is True
    assert repository.claim(key) is False

    auth_error = APIError(
        {
            "code": "42501",
            "message": "permission denied",
            "details": None,
            "hint": None,
        }
    )
    with patch.object(table, "execute_insert", side_effect=auth_error):
        with pytest.raises(APIError) as raised:
            repository.claim(key)
    assert raised.value is auth_error


def test_claim_does_not_treat_duplicate_words_without_23505_as_uniqueness(
    key: SchedulerRunKey,
) -> None:
    table = _FakeSchedulerRunsTable()
    repository = _repository(table)
    error = RuntimeError("duplicate key response from an unavailable gateway")

    with patch.object(table, "execute_insert", side_effect=error):
        with pytest.raises(RuntimeError) as raised:
            repository.claim(key)

    assert raised.value is error


def test_finish_filters_all_key_columns_and_writes_result_counters(
    key: SchedulerRunKey,
) -> None:
    table = _FakeSchedulerRunsTable()
    repository = _repository(table)
    result = BatchRunResult(
        status="partial",
        total_count=12,
        success_count=8,
        failure_count=4,
        cached_count=3,
    )

    before = dt.datetime.now(dt.timezone.utc)
    repository.finish(key, result)
    after = dt.datetime.now(dt.timezone.utc)

    assert len(table.updates) == 1
    payload, filters = table.updates[0]
    finished_at = dt.datetime.fromisoformat(payload.pop("finished_at"))
    assert before <= finished_at <= after
    assert payload == {
        "status": "partial",
        "total_count": 12,
        "success_count": 8,
        "failure_count": 4,
        "cached_count": 3,
    }
    assert filters == [
        ("tenant_id", "tenant-a"),
        ("scheduled_for", "2026-07-22T23:30:00+00:00"),
        ("prompt_key", "follow-up"),
        ("prompt_version", 7),
        ("model_key", "gemini-2.5-pro"),
    ]


@pytest.mark.parametrize("updated_rows", [[], [{}, {}]])
def test_finish_requires_exactly_one_updated_row(
    key: SchedulerRunKey,
    updated_rows: list[dict],
) -> None:
    table = _FakeSchedulerRunsTable()
    table.update_response_data = updated_rows
    repository = _repository(table)
    result = BatchRunResult.from_counts(
        total_count=1,
        success_count=1,
        failure_count=0,
    )

    with pytest.raises(RuntimeError, match="exactly one scheduler run"):
        repository.finish(key, result)


def test_concurrent_claim_is_atomic_and_only_one_worker_wins(
    key: SchedulerRunKey,
) -> None:
    table = _FakeSchedulerRunsTable()
    repository = _repository(table)
    barrier = threading.Barrier(2)

    def claim_after_barrier() -> bool:
        barrier.wait()
        return repository.claim(key)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: claim_after_barrier(), range(2)))

    assert sorted(results) == [False, True]
    assert len(table.rows) == 1
    assert table.rows[0]["status"] == "running"
