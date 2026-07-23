from __future__ import annotations

import re
from pathlib import Path


SCHEMA_PATH = Path("docs/supabase/multi_tenant_schema.sql")


def test_multi_tenant_schema_declares_required_tables_and_cache_version() -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8").lower()
    compact_sql = " ".join(sql.split())

    for table_name in (
        "tenants",
        "tenant_users",
        "tenant_user_access",
        "tenant_settings",
        "tenant_prompt_templates",
        "tenant_secrets",
        "analysis_results",
    ):
        assert f"create table if not exists public.{table_name}" in sql

    assert "prompt_version integer not null default 1" in sql
    assert (
        "tenant_id, call_unique_id, prompt_key, prompt_version, "
        "provider_name, model_key, custom_fragment"
    ) in compact_sql

    for table_name in (
        "tenants",
        "tenant_users",
        "tenant_user_access",
        "tenant_settings",
        "tenant_prompt_templates",
        "tenant_secrets",
        "analysis_results",
    ):
        assert f"alter table public.{table_name} enable row level security" in sql


def test_mvp_schema_keeps_prompt_index_non_unique_and_needs_no_admin_rpc() -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8").lower()

    assert "create index if not exists idx_tenant_prompt_templates_active" in sql
    assert "create unique index" not in sql
    assert "save_tenant_admin_settings" not in sql
    assert not Path("docs/supabase/migrations/20260712_tenant_admin_settings.sql").exists()


def test_scheduler_runs_has_exact_identity_counters_and_service_role_policy() -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8").lower()
    compact_sql = " ".join(sql.split())

    expected_definition = """
        tenant_id text not null references public.tenants(id) on delete cascade,
        scheduled_for timestamptz not null,
        prompt_key text not null,
        prompt_version integer not null,
        model_key text not null,
        status text not null check (status in ('running', 'success', 'partial', 'failed')),
        total_count integer not null default 0,
        success_count integer not null default 0,
        failure_count integer not null default 0,
        cached_count integer not null default 0,
        started_at timestamptz not null default now(),
        finished_at timestamptz,
        primary key (tenant_id, scheduled_for, prompt_key, prompt_version, model_key)
    """
    assert "create table if not exists public.scheduler_runs" in sql
    assert " ".join(expected_definition.split()) in compact_sql
    assert "alter table public.scheduler_runs enable row level security" in sql
    assert re.search(
        r"create policy .*? on public\.scheduler_runs\s+for all\s+to service_role"
        r"\s+using \(true\)\s+with check \(true\)",
        compact_sql,
    )


def test_schema_never_changes_historical_analysis_results_data() -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8").lower()

    forbidden = (
        r"\binsert\s+into\s+(?:public\.)?analysis_results\b",
        r"\bupdate\s+(?:public\.)?analysis_results\b",
        r"\bdelete\s+from\s+(?:public\.)?analysis_results\b",
        r"\btruncate(?:\s+table)?\s+(?:public\.)?analysis_results\b",
        r"\bmerge\s+into\s+(?:public\.)?analysis_results\b",
    )
    assert not any(re.search(pattern, sql) for pattern in forbidden)
