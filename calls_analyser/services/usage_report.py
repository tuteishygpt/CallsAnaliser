"""Read-only reporting helpers for Gemini usage rows."""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import Iterable

import pandas as pd


UNKNOWN_VALUE = "(unknown)"
ALL_VALUE = "All"

DETAIL_COLUMNS = [
    "created_at",
    "tenant_id",
    "call_started_at",
    "call_user",
    "caller_id",
    "destination",
    "duration_seconds",
    "prompt_key",
    "provider_name",
    "model_key",
    "mode",
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
    "estimated_cost",
    "estimated_client_price",
    "margin",
    "currency",
    "call_unique_id",
]

NUMERIC_COLUMNS = [
    "duration_seconds",
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
    "estimated_cost",
    "estimated_client_price",
]

TEXT_COLUMNS = [
    "created_at",
    "tenant_id",
    "call_started_at",
    "call_user",
    "caller_id",
    "destination",
    "prompt_key",
    "provider_name",
    "model_key",
    "mode",
    "currency",
    "call_unique_id",
]

GROUP_COLUMNS = [
    "total_calls",
    "total_duration_seconds",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_cost",
    "estimated_client_price",
    "margin",
]


@dataclass(frozen=True)
class UsageReportFilters:
    tenant_id: str | None = None
    date_from: str | dt.date | None = None
    date_to: str | dt.date | None = None
    mode: str | None = ALL_VALUE
    model_key: str | None = ALL_VALUE
    call_user: str | None = ALL_VALUE
    include_cache_hits: bool = False

    @property
    def date_from_iso(self) -> str | None:
        date_value = _parse_date(self.date_from)
        if date_value is None:
            return None
        return dt.datetime.combine(date_value, dt.time.min).isoformat()

    @property
    def date_to_iso(self) -> str | None:
        date_value = _parse_date(self.date_to)
        if date_value is None:
            return None
        return dt.datetime.combine(date_value, dt.time(23, 59, 59)).isoformat()


@dataclass(frozen=True)
class UsageReport:
    summary: dict[str, object]
    by_model_mode: pd.DataFrame
    by_user: pd.DataFrame
    details: pd.DataFrame


def build_usage_report(rows: Iterable[dict]) -> UsageReport:
    details = _normalize_rows(rows)
    return UsageReport(
        summary=_build_summary(details),
        by_model_mode=_group_usage(details, ["model_key", "mode"]),
        by_user=_group_usage(details, ["call_user"]),
        details=details,
    )


def _parse_date(value: str | dt.date | None) -> dt.date | None:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _normalize_rows(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for column in DETAIL_COLUMNS:
        if column not in df.columns:
            df[column] = 0 if column in NUMERIC_COLUMNS else ""

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    for column in ("call_user", "model_key", "mode"):
        df[column] = df[column].replace("", UNKNOWN_VALUE)

    df["margin"] = df["estimated_client_price"] - df["estimated_cost"]
    return df[DETAIL_COLUMNS].fillna("")


def _build_summary(details: pd.DataFrame) -> dict[str, object]:
    duration_seconds = _sum(details, "duration_seconds")
    estimated_cost = _sum(details, "estimated_cost")
    estimated_client_price = _sum(details, "estimated_client_price")
    margin = estimated_client_price - estimated_cost
    return {
        "total_calls": int(len(details)),
        "total_duration_seconds": duration_seconds,
        "total_duration_minutes": round(duration_seconds / 60, 2),
        "prompt_tokens": _sum(details, "prompt_token_count"),
        "output_tokens": _sum(details, "candidates_token_count"),
        "total_tokens": _sum(details, "total_token_count"),
        "estimated_cost": round(estimated_cost, 8),
        "estimated_client_price": round(estimated_client_price, 8),
        "margin": round(margin, 8),
    }


def _group_usage(details: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    columns = by + GROUP_COLUMNS
    if details.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        details.groupby(by, dropna=False, as_index=False)
        .agg(
            total_calls=("total_token_count", "size"),
            total_duration_seconds=("duration_seconds", "sum"),
            prompt_tokens=("prompt_token_count", "sum"),
            output_tokens=("candidates_token_count", "sum"),
            total_tokens=("total_token_count", "sum"),
            estimated_cost=("estimated_cost", "sum"),
            estimated_client_price=("estimated_client_price", "sum"),
            margin=("margin", "sum"),
        )
        .sort_values(by)
        .reset_index(drop=True)
    )
    for column in ("estimated_cost", "estimated_client_price", "margin"):
        grouped[column] = grouped[column].round(8)
    return grouped[columns]


def _sum(details: pd.DataFrame, column: str) -> int | float:
    value = details[column].sum()
    if column in {"estimated_cost", "estimated_client_price"}:
        return float(value)
    return int(value)
