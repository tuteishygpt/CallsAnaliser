from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from calls_analyser.ui.handlers import UIHandlers


class _FakeUsageReportRepository:
    def __init__(self, rows=None, values=None) -> None:  # noqa: ANN001
        self.rows = rows or []
        self.values = values or {
            "models": ["All"],
            "modes": ["All"],
            "users": ["All"],
        }
        self.list_usage_calls = []
        self.list_filter_values_calls = []

    def list_usage(self, filters):  # noqa: ANN001
        self.list_usage_calls.append(filters)
        return self.rows

    def list_filter_values(self, tenant_id=None):  # noqa: ANN001
        self.list_filter_values_calls.append(tenant_id)
        return self.values


def _handlers(repository=None) -> UIHandlers:  # noqa: ANN001
    return UIHandlers(
        SimpleNamespace(
            project_imports_available=True,
            usage_report_repository=repository,
        )
    )


def test_load_usage_report_requires_authentication() -> None:
    result = _handlers(_FakeUsageReportRepository()).load_usage_report(
        "lix",
        "2026-06-01",
        "2026-06-27",
        "All",
        "All",
        "All",
        False,
    )

    summary, by_model_mode, by_user, details, state = result
    assert "Enter the password" in summary
    assert by_model_mode.empty
    assert by_user.empty
    assert details.empty
    assert state.empty


def test_load_usage_report_requires_repository() -> None:
    result = _handlers(None).load_usage_report(
        "lix",
        "",
        "",
        "All",
        "All",
        "All",
        True,
    )

    assert "Usage reporting is not configured" in result[0]
    assert result[1].empty


def test_load_usage_report_returns_summary_tables_and_export_state() -> None:
    repository = _FakeUsageReportRepository(
        [
            {
                "tenant_id": "lix",
                "model_key": "models/gemini-2.5-flash-lite",
                "mode": "ui_mass",
                "call_user": "agent-1",
                "duration_seconds": 60,
                "prompt_token_count": 100,
                "candidates_token_count": 25,
                "total_token_count": 125,
                "estimated_cost": 0.01,
                "estimated_client_price": 0.02,
                "currency": "USD",
            }
        ]
    )

    summary, by_model_mode, by_user, details, state = _handlers(repository).load_usage_report(
        "lix",
        "2026-06-01",
        "2026-06-27",
        "ui_mass",
        "models/gemini-2.5-flash-lite",
        "agent-1",
        True,
    )

    assert "### Usage summary" in summary
    assert "Calls: 1" in summary
    assert "Tokens: 125 total (prompt 100, output 25)" in summary
    assert "Internal cost: 0.01 USD" in summary
    assert by_model_mode.iloc[0]["total_calls"] == 1
    assert by_user.iloc[0]["call_user"] == "agent-1"
    assert details.iloc[0]["margin"] == 0.01
    assert state.equals(details)

    filters = repository.list_usage_calls[0]
    assert filters.tenant_id == "lix"
    assert filters.mode == "ui_mass"
    assert filters.model_key == "models/gemini-2.5-flash-lite"
    assert filters.call_user == "agent-1"


def test_export_usage_report_writes_csv_file() -> None:
    details = pd.DataFrame([{"call_unique_id": "call-1", "total_token_count": 125}])

    file_update, message = _handlers().export_usage_report(details)

    path = Path(file_update["value"])
    assert file_update["visible"] is True
    assert path.exists()
    assert "File is ready" in message
    assert "call-1" in path.read_text(encoding="utf-8")


def test_export_usage_report_rejects_empty_data() -> None:
    file_update, message = _handlers().export_usage_report(pd.DataFrame())

    assert file_update["visible"] is False
    assert "No report data to export" in message


def test_load_usage_report_filter_choices() -> None:
    repository = _FakeUsageReportRepository(
        values={
            "models": ["All", "models/gemini-2.5-pro"],
            "modes": ["All", "ui_direct"],
            "users": ["All", "(unknown)", "agent-1"],
        }
    )

    model_update, mode_update, user_update, message = _handlers(
        repository
    ).load_usage_report_filter_choices("lix", True)

    assert model_update["choices"] == ["All", "models/gemini-2.5-pro"]
    assert mode_update["choices"] == ["All", "ui_direct"]
    assert user_update["choices"] == ["All", "(unknown)", "agent-1"]
    assert message == ""
    assert repository.list_filter_values_calls == ["lix"]
