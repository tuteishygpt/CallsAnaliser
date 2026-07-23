from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from calls_analyser.adapters.storage.supabase_usage_report import (
    SupabaseUsageReportRepository,
)
from calls_analyser.services.usage_report import UsageReportFilters


class _FakeTable:
    def __init__(self, pages=None) -> None:  # noqa: ANN001
        self.pages = pages or [[]]
        self.calls = []
        self.execute_count = 0

    def select(self, value):  # noqa: ANN001
        self.calls.append(("select", value))
        return self

    def eq(self, key, value):  # noqa: ANN001
        self.calls.append(("eq", key, value))
        return self

    def gte(self, key, value):  # noqa: ANN001
        self.calls.append(("gte", key, value))
        return self

    def lte(self, key, value):  # noqa: ANN001
        self.calls.append(("lte", key, value))
        return self

    def or_(self, value):  # noqa: ANN001
        self.calls.append(("or", value))
        return self

    def order(self, key, desc=False):  # noqa: ANN001
        self.calls.append(("order", key, desc))
        return self

    def limit(self, value):  # noqa: ANN001
        self.calls.append(("limit", value))
        return self

    def range(self, start, end):  # noqa: A003, ANN001
        self.calls.append(("range", start, end))
        return self

    def execute(self):
        page_index = min(self.execute_count, len(self.pages) - 1)
        self.execute_count += 1
        return SimpleNamespace(data=self.pages[page_index])


class _FakeClient:
    def __init__(self, table: _FakeTable) -> None:
        self.usage = table

    def table(self, name):  # noqa: ANN001
        if name != "analysis_usage":
            raise AssertionError(name)
        return self.usage


def _build_repo(table: _FakeTable) -> SupabaseUsageReportRepository:
    fake_client = _FakeClient(table)
    with patch(
        "calls_analyser.adapters.storage.supabase_usage_report.create_client",
        return_value=fake_client,
    ):
        return SupabaseUsageReportRepository("https://example.supabase.co", "key")


def test_supabase_usage_report_fetches_usage_with_filters() -> None:
    table = _FakeTable(pages=[[{"id": 1}], []])
    repo = _build_repo(table)

    rows = repo.list_usage(
        UsageReportFilters(
            tenant_id="lix",
            date_from="2026-06-01",
            date_to="2026-06-27",
            mode="ui_mass",
            model_key="models/gemini-2.5-flash-lite",
            call_user="agent-1",
        )
    )

    assert rows == [{"id": 1}]
    assert ("select", "*") in table.calls
    assert ("eq", "tenant_id", "lix") in table.calls
    assert ("gte", "created_at", "2026-06-01T00:00:00") in table.calls
    assert ("lte", "created_at", "2026-06-27T23:59:59") in table.calls
    assert ("eq", "mode", "ui_mass") in table.calls
    assert ("eq", "model_key", "models/gemini-2.5-flash-lite") in table.calls
    assert ("eq", "call_user", "agent-1") in table.calls
    assert ("eq", "cache_hit", False) in table.calls
    assert ("order", "created_at", True) in table.calls
    assert ("range", 0, 999) in table.calls


def test_supabase_usage_report_skips_all_filters_and_can_include_cache_hits() -> None:
    table = _FakeTable(pages=[[]])
    repo = _build_repo(table)

    repo.list_usage(
        UsageReportFilters(
            mode="All",
            model_key="All",
            call_user="All",
            include_cache_hits=True,
        )
    )

    assert not [call for call in table.calls if call[:2] == ("eq", "mode")]
    assert not [call for call in table.calls if call[:2] == ("eq", "model_key")]
    assert not [call for call in table.calls if call[:2] == ("eq", "call_user")]
    assert ("eq", "cache_hit", False) not in table.calls


def test_supabase_usage_report_maps_unknown_user_filter_to_missing_values() -> None:
    table = _FakeTable(pages=[[]])
    repo = _build_repo(table)

    repo.list_usage(UsageReportFilters(call_user="(unknown)"))

    assert ("eq", "call_user", "(unknown)") not in table.calls
    assert ("or", "call_user.is.null,call_user.eq.") in table.calls


def test_supabase_usage_report_paginates_until_short_page() -> None:
    first_page = [{"id": idx} for idx in range(1000)]
    second_page = [{"id": 1000}]
    table = _FakeTable(pages=[first_page, second_page])
    repo = _build_repo(table)

    rows = repo.list_usage(UsageReportFilters())

    assert len(rows) == 1001
    assert ("range", 0, 999) in table.calls
    assert ("range", 1000, 1999) in table.calls


def test_supabase_usage_report_derives_filter_values_from_recent_rows() -> None:
    table = _FakeTable(
        pages=[
            [
                {
                    "model_key": "models/gemini-2.5-pro",
                    "mode": "ui_direct",
                    "call_user": None,
                },
                {
                    "model_key": "models/gemini-2.5-flash-lite",
                    "mode": "ui_mass",
                    "call_user": "agent-1",
                },
                {
                    "model_key": "models/gemini-2.5-flash-lite",
                    "mode": "scheduler_batch",
                    "call_user": "",
                },
            ]
        ]
    )
    repo = _build_repo(table)

    values = repo.list_filter_values("lix")

    assert values == {
        "models": ["All", "models/gemini-2.5-flash-lite", "models/gemini-2.5-pro"],
        "modes": ["All", "scheduler_batch", "ui_direct", "ui_mass"],
        "users": ["All", "(unknown)", "agent-1"],
    }
    assert ("select", "model_key,mode,call_user") in table.calls
    assert ("eq", "tenant_id", "lix") in table.calls
    assert ("order", "created_at", True) in table.calls
    assert ("limit", 5000) in table.calls
