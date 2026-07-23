"""Supabase persistence for atomic scheduler-run claims and completion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from supabase import Client, create_client

from calls_analyser.services.batch_results import BatchRunResult


@dataclass(frozen=True)
class SchedulerRunKey:
    """Identity of one protected scheduler slot."""

    tenant_id: str
    scheduled_for: datetime
    prompt_key: str
    prompt_version: int
    model_key: str


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scheduled_for must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


class SupabaseSchedulerRunRepository:
    """Atomically claims scheduler slots using the table's composite primary key."""

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        self._client: Client = create_client(supabase_url, supabase_key)
        self._table = self._client.table("scheduler_runs")

    def claim(self, key: SchedulerRunKey) -> bool:
        """Insert one running slot; return false only when its identity already exists."""
        row = {
            "tenant_id": key.tenant_id,
            "scheduled_for": _utc_iso(key.scheduled_for),
            "prompt_key": key.prompt_key,
            "prompt_version": key.prompt_version,
            "model_key": key.model_key,
            "status": "running",
        }
        try:
            self._table.insert(row).execute()
        except Exception as exc:
            if str(getattr(exc, "code", "")) == "23505":
                return False
            raise
        return True

    def finish(self, key: SchedulerRunKey, result: BatchRunResult) -> None:
        """Finalize the exact claimed slot with its terminal result counters."""
        response = (
            self._table.update(
                {
                    "status": result.status,
                    "total_count": result.total_count,
                    "success_count": result.success_count,
                    "failure_count": result.failure_count,
                    "cached_count": result.cached_count,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            .eq("tenant_id", key.tenant_id)
            .eq("scheduled_for", _utc_iso(key.scheduled_for))
            .eq("prompt_key", key.prompt_key)
            .eq("prompt_version", key.prompt_version)
            .eq("model_key", key.model_key)
            .execute()
        )
        updated_rows = response.data or []
        if len(updated_rows) != 1:
            raise RuntimeError(
                "Expected to finish exactly one scheduler run, "
                f"updated {len(updated_rows)} rows"
            )
