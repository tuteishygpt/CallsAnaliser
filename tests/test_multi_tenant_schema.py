from __future__ import annotations

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
