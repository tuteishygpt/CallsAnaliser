"""Time and guarded-execution helpers for scheduled tenant batches."""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from calls_analyser.adapters.storage.supabase_scheduler_runs import SchedulerRunKey
from calls_analyser.runner import resolve_batch_execution_context
from calls_analyser.services.batch_results import BatchRunResult


logger = logging.getLogger(__name__)
UTC = timezone.utc


def scheduler_timezone(value: str | None) -> ZoneInfo:
    """Return the configured global scheduler timezone, defaulting to UTC."""
    return ZoneInfo(str(value).strip() if value and str(value).strip() else "UTC")


def cron_scheduled_for(now: datetime, cron_time: time) -> datetime:
    """Return the latest planned cron occurrence at or before ``now`` in UTC."""
    _require_aware(now, "now")
    planned_time = cron_time.replace(tzinfo=None)
    now_utc = now.astimezone(UTC)
    candidates = [
        occurrence
        for planned_date in (now.date(), now.date() - timedelta(days=1))
        for occurrence in _local_occurrences(planned_date, planned_time, now.tzinfo)
        if occurrence <= now_utc
    ]
    return max(candidates)


def interval_scheduled_for(now: datetime, minutes: int) -> datetime:
    """Floor ``now`` to its current UTC-day interval bucket."""
    _require_aware(now, "now")
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes <= 0:
        raise ValueError("Interval minutes must be a positive integer")

    now_utc = now.astimezone(UTC)
    utc_midnight = datetime.combine(now_utc.date(), time.min, tzinfo=UTC)
    elapsed_seconds = (now_utc - utc_midnight).total_seconds()
    interval_seconds = minutes * 60
    bucket = int(elapsed_seconds // interval_seconds)
    return utc_midnight + timedelta(seconds=bucket * interval_seconds)


def run_scheduled_batch_for_tenant(
    *,
    tenant_id: str,
    runtime_settings: Any,
    scheduled_for: datetime,
    now: datetime,
    run_repository: Any,
    runner: Any,
    deps: Any,
) -> BatchRunResult | None:
    """Claim and execute one tenant slot; return ``None`` only for a duplicate."""
    if run_repository is None:
        raise RuntimeError("Scheduler run repository is unavailable")

    _require_aware(scheduled_for, "scheduled_for")
    _require_aware(now, "now")
    context = resolve_batch_execution_context(
        deps,
        tenant_id=tenant_id,
        runtime_settings=runtime_settings,
    )
    key = SchedulerRunKey(
        tenant_id=str(context.tenant.tenant_id),
        scheduled_for=scheduled_for.astimezone(UTC),
        prompt_key=context.prompt_key,
        prompt_version=context.prompt_version,
        model_key=context.batch_model_key,
    )

    if not run_repository.claim(key):
        return None

    target_day = now.date() - timedelta(days=1)
    try:
        result = runner(
            deps,
            target_day,
            None,
            None,
            "",
            key.tenant_id,
            execution_context=context,
        )
    except Exception:
        failed = BatchRunResult(
            status="failed",
            total_count=0,
            success_count=0,
            failure_count=0,
            cached_count=0,
        )
        try:
            run_repository.finish(key, failed)
        except Exception:
            logger.exception(
                "Failed to finalize errored scheduler slot for tenant %s",
                key.tenant_id,
            )
        logger.exception("Scheduled batch failed for tenant %s", key.tenant_id)
        raise

    run_repository.finish(key, result)
    return result


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _local_occurrences(planned_date, planned_time: time, tzinfo) -> list[datetime]:
    """Return the distinct real UTC instants for one local wall-clock minute."""
    occurrences: list[datetime] = []
    for fold in (0, 1):
        local = datetime.combine(
            planned_date,
            planned_time,
            tzinfo=tzinfo,
        ).replace(fold=fold)
        occurrence = local.astimezone(UTC)
        round_trip = occurrence.astimezone(tzinfo)
        if round_trip.date() != planned_date or round_trip.time().replace(tzinfo=None) != planned_time:
            continue
        if occurrence not in occurrences:
            occurrences.append(occurrence)
    return occurrences
