"""Supabase-backed read repository for token usage reports."""
from __future__ import annotations

from typing import Any

from supabase import Client, create_client

from calls_analyser.services.usage_report import (
    ALL_VALUE,
    UNKNOWN_VALUE,
    UsageReportFilters,
)


PAGE_SIZE = 1000
FILTER_VALUE_LIMIT = 5000


class SupabaseUsageReportRepository:
    """Reads usage rows from Supabase for reporting."""

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        self._client: Client = create_client(supabase_url, supabase_key)
        self._usage_table = self._client.table("analysis_usage")

    def list_usage(self, filters: UsageReportFilters) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        start = 0

        while True:
            query = self._usage_table.select("*")
            query = self._apply_usage_filters(query, filters)
            response = (
                query.order("created_at", desc=True)
                .range(start, start + PAGE_SIZE - 1)
                .execute()
            )
            page = response.data or []
            rows.extend(page)

            if len(page) < PAGE_SIZE:
                break
            start += PAGE_SIZE

        return rows

    def list_filter_values(self, tenant_id: str | None = None) -> dict[str, list[str]]:
        query = self._usage_table.select("model_key,mode,call_user")
        if tenant_id:
            query = query.eq("tenant_id", tenant_id)

        response = query.order("created_at", desc=True).limit(FILTER_VALUE_LIMIT).execute()
        rows = response.data or []

        return {
            "models": [ALL_VALUE, *_sorted_values(rows, "model_key")],
            "modes": [ALL_VALUE, *_sorted_values(rows, "mode")],
            "users": [ALL_VALUE, *_sorted_values(rows, "call_user", unknown=True)],
        }

    def _apply_usage_filters(self, query: Any, filters: UsageReportFilters) -> Any:
        if filters.tenant_id:
            query = query.eq("tenant_id", filters.tenant_id)
        if filters.date_from_iso:
            query = query.gte("created_at", filters.date_from_iso)
        if filters.date_to_iso:
            query = query.lte("created_at", filters.date_to_iso)
        if _is_concrete_filter(filters.mode):
            query = query.eq("mode", filters.mode)
        if _is_concrete_filter(filters.model_key):
            query = query.eq("model_key", filters.model_key)
        if filters.call_user == UNKNOWN_VALUE:
            query = query.or_("call_user.is.null,call_user.eq.")
        elif _is_concrete_filter(filters.call_user):
            query = query.eq("call_user", filters.call_user)
        if not filters.include_cache_hits:
            query = query.eq("cache_hit", False)
        return query


def _is_concrete_filter(value: str | None) -> bool:
    return bool(value) and value != ALL_VALUE


def _sorted_values(
    rows: list[dict[str, Any]],
    key: str,
    *,
    unknown: bool = False,
) -> list[str]:
    values: set[str] = set()
    for row in rows:
        value = row.get(key)
        if unknown and value in (None, ""):
            value = UNKNOWN_VALUE
        if value not in (None, ""):
            values.add(str(value))
    return sorted(values)
