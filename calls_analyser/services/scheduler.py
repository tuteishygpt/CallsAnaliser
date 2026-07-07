"""Scheduler service helpers."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_MISSING = object()


@dataclass(frozen=True)
class ScheduledTenantSuccess:
    tenant_id: str
    result: Any = None


@dataclass(frozen=True)
class ScheduledTenantFailure:
    tenant_id: str
    error: str
    exception_type: str


@dataclass(frozen=True)
class SchedulerRunSummary:
    successes: list[ScheduledTenantSuccess] = field(default_factory=list)
    failures: list[ScheduledTenantFailure] = field(default_factory=list)


def run_scheduled_batches_for_enabled_tenants(
    *,
    tenant_settings_service,
    runner,
    deps,
    day,
) -> SchedulerRunSummary:
    successes: list[ScheduledTenantSuccess] = []
    failures: list[ScheduledTenantFailure] = []

    enabled_tenants = tenant_settings_service.list_scheduler_enabled_tenants() or []
    for tenant_ref in enabled_tenants:
        try:
            tenant_id = _tenant_id_from(tenant_ref)
        except Exception as exc:
            failures.append(
                ScheduledTenantFailure(
                    tenant_id=str(tenant_ref),
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )
            )
            continue

        try:
            runtime_settings = tenant_settings_service.resolve(tenant_id)
            time_from_str, time_to_str, call_type_str = _scheduler_filters(runtime_settings)
            result = runner(
                deps,
                day,
                time_from_str,
                time_to_str,
                call_type_str,
                tenant_id,
            )
        except Exception as exc:
            failures.append(
                ScheduledTenantFailure(
                    tenant_id=tenant_id,
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )
            )
            continue

        successes.append(ScheduledTenantSuccess(tenant_id=tenant_id, result=result))

    return SchedulerRunSummary(successes=successes, failures=failures)


def _scheduler_filters(runtime_settings) -> tuple[Any, Any, Any]:
    filters = _read_value(runtime_settings, "scheduler_filters", default={}) or {}
    time_from = _filter_value(
        filters,
        "time_from",
        runtime_settings,
        fallback_keys=("filter_time_from", "time_from"),
        default=None,
    )
    time_to = _filter_value(
        filters,
        "time_to",
        runtime_settings,
        fallback_keys=("filter_time_to", "time_to"),
        default=None,
    )
    call_type = _filter_value(
        filters,
        "call_type",
        runtime_settings,
        fallback_keys=("filter_call_type", "call_type"),
        default="",
    )
    if call_type is None:
        call_type = ""
    return time_from, time_to, call_type


def _filter_value(filters, key: str, runtime_settings, *, fallback_keys: tuple[str, ...], default):
    value = _read_value(filters, key, default=_MISSING)
    if value is not _MISSING and value is not None:
        return value

    for fallback_key in fallback_keys:
        value = _read_value(runtime_settings, fallback_key, default=_MISSING)
        if value is not _MISSING and value is not None:
            return value

    return default


def _tenant_id_from(tenant_ref) -> str:
    if isinstance(tenant_ref, str):
        tenant_id = tenant_ref
    else:
        tenant_id = _read_value(tenant_ref, "tenant_id", default=_MISSING)
        if tenant_id is _MISSING:
            tenant_id = _read_value(tenant_ref, "id", default=_MISSING)

    if tenant_id is _MISSING or tenant_id is None or str(tenant_id).strip() == "":
        raise ValueError("Enabled scheduler tenant is missing tenant id")
    return str(tenant_id)


def _read_value(source, key: str, *, default=None):
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)
