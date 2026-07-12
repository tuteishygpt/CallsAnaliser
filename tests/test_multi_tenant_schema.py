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


def test_multi_tenant_schema_documents_follow_up_verification_settings() -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8").lower()

    assert "follow_up_verification_mode" in sql
    assert "follow_up_verification_model_key" in sql
    assert "follow_up_verification_prompt_key" in sql
    assert "off, shadow, or enforce" in sql
    assert "defaults to off" in sql
