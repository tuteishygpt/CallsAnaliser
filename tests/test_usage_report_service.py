from __future__ import annotations

import pandas as pd

from calls_analyser.services.usage_report import (
    DETAIL_COLUMNS,
    UsageReportFilters,
    build_usage_report,
)


def test_usage_report_summarizes_cost_tokens_and_margin() -> None:
    rows = [
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
            "cache_hit": False,
        },
        {
            "tenant_id": "lix",
            "model_key": "models/gemini-2.5-pro",
            "mode": "ui_direct",
            "call_user": None,
            "duration_seconds": 120,
            "prompt_token_count": 200,
            "candidates_token_count": 50,
            "total_token_count": 250,
            "estimated_cost": 0.03,
            "estimated_client_price": 0.06,
            "cache_hit": False,
        },
    ]

    report = build_usage_report(rows)

    assert report.summary["total_calls"] == 2
    assert report.summary["total_duration_seconds"] == 180
    assert report.summary["total_duration_minutes"] == 3.0
    assert report.summary["prompt_tokens"] == 300
    assert report.summary["output_tokens"] == 75
    assert report.summary["total_tokens"] == 375
    assert report.summary["estimated_cost"] == 0.04
    assert report.summary["estimated_client_price"] == 0.08
    assert report.summary["margin"] == 0.04


def test_usage_report_groups_by_model_mode_and_user() -> None:
    rows = [
        {
            "model_key": "models/gemini-2.5-flash-lite",
            "mode": "ui_mass",
            "call_user": "agent-1",
            "duration_seconds": 60,
            "prompt_token_count": 100,
            "candidates_token_count": 25,
            "total_token_count": 125,
            "estimated_cost": 0.01,
            "estimated_client_price": 0.02,
        },
        {
            "model_key": "models/gemini-2.5-flash-lite",
            "mode": "ui_mass",
            "call_user": "",
            "duration_seconds": 120,
            "prompt_token_count": 200,
            "candidates_token_count": 50,
            "total_token_count": 250,
            "estimated_cost": 0.03,
            "estimated_client_price": 0.06,
        },
    ]

    report = build_usage_report(rows)

    assert list(report.by_model_mode.columns) == [
        "model_key",
        "mode",
        "total_calls",
        "total_duration_seconds",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost",
        "estimated_client_price",
        "margin",
    ]
    assert report.by_model_mode.to_dict("records") == [
        {
            "model_key": "models/gemini-2.5-flash-lite",
            "mode": "ui_mass",
            "total_calls": 2,
            "total_duration_seconds": 180,
            "prompt_tokens": 300,
            "output_tokens": 75,
            "total_tokens": 375,
            "estimated_cost": 0.04,
            "estimated_client_price": 0.08,
            "margin": 0.04,
        }
    ]
    assert list(report.by_user["call_user"]) == ["(unknown)", "agent-1"]


def test_usage_report_details_have_stable_columns_and_no_none_values() -> None:
    report = build_usage_report(
        [
            {
                "created_at": "2026-06-27T10:00:00Z",
                "tenant_id": "lix",
                "call_user": None,
                "model_key": None,
                "mode": None,
                "estimated_cost": 0.01,
                "estimated_client_price": 0.02,
            }
        ]
    )

    assert list(report.details.columns) == DETAIL_COLUMNS
    detail = report.details.iloc[0]
    assert detail["call_user"] == "(unknown)"
    assert detail["model_key"] == "(unknown)"
    assert detail["mode"] == "(unknown)"
    assert detail["margin"] == 0.01
    assert not any(pd.isna(detail[column]) for column in DETAIL_COLUMNS)


def test_usage_report_filters_convert_dates_to_iso_bounds() -> None:
    filters = UsageReportFilters(
        tenant_id="lix",
        date_from="2026-06-01",
        date_to="2026-06-27",
        mode="All",
        model_key="All",
        call_user="All",
    )

    assert filters.date_from_iso == "2026-06-01T00:00:00"
    assert filters.date_to_iso == "2026-06-27T23:59:59"
