"""Safely remove cached analysis results for a selected set of call IDs."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def cleanup_analysis_results(
    table: Any,
    *,
    tenant_id: str,
    call_unique_ids: Iterable[str],
    execute: bool,
) -> list[str]:
    """Return matching IDs and delete them only when ``execute`` is true."""
    unique_ids = list(dict.fromkeys(call_unique_ids))
    if not unique_ids:
        return []

    response = (
        table.select("call_unique_id")
        .eq("tenant_id", tenant_id)
        .in_("call_unique_id", unique_ids)
        .execute()
    )
    requested_ids = set(unique_ids)
    matching_ids = [
        record["call_unique_id"]
        for record in (response.data or [])
        if record.get("call_unique_id") in requested_ids
    ]
    if execute and matching_ids:
        (
            table.delete()
            .eq("tenant_id", tenant_id)
            .in_("call_unique_id", matching_ids)
            .execute()
        )
    return matching_ids
