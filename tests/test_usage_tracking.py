from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from calls_analyser.services.usage import (
    PricingSnapshot,
    UsageMetadata,
    build_usage_record,
    extract_usage_metadata,
)


def test_extract_usage_metadata_accepts_gemini_camel_case_dict() -> None:
    usage = extract_usage_metadata(
        {
            "promptTokenCount": 4115,
            "candidatesTokenCount": 583,
            "totalTokenCount": 4698,
            "thoughtsTokenCount": 0,
        }
    )

    assert usage == UsageMetadata(
        prompt_token_count=4115,
        candidates_token_count=583,
        total_token_count=4698,
        thoughts_token_count=0,
    )


def test_extract_usage_metadata_accepts_genai_snake_case_object() -> None:
    usage = extract_usage_metadata(
        SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=170,
            thoughts_token_count=20,
        )
    )

    assert usage.prompt_token_count == 100
    assert usage.candidates_token_count == 50
    assert usage.total_token_count == 170
    assert usage.thoughts_token_count == 20


def test_build_usage_record_calculates_cost_and_client_price() -> None:
    entry = SimpleNamespace(
        unique_id="call-1",
        started_at=dt.datetime(2026, 6, 25, 9, 0),
        caller_id="375291112233",
        destination="101",
        duration_seconds=90,
        raw={"user": "agent-7"},
    )
    tenant = SimpleNamespace(tenant_id="lix")
    usage = UsageMetadata(
        prompt_token_count=1_000_000,
        candidates_token_count=500_000,
        total_token_count=1_500_000,
        thoughts_token_count=0,
    )
    pricing = PricingSnapshot(
        currency="USD",
        input_cost_per_1m=0.30,
        output_cost_per_1m=2.50,
        input_price_per_1m=0.60,
        output_price_per_1m=5.00,
    )

    record = build_usage_record(
        entry=entry,
        tenant=tenant,
        prompt_key="follow_up",
        custom_fragment="custom text",
        provider_name="gemini",
        model_key="models/gemini-test",
        mode="ui_mass",
        usage=usage,
        pricing=pricing,
        cache_key=("lix", "call-1", "follow_up", "gemini", "models/gemini-test", "custom text"),
    )

    assert record["tenant_id"] == "lix"
    assert record["call_unique_id"] == "call-1"
    assert record["call_user"] == "agent-7"
    assert record["duration_seconds"] == 90
    assert record["mode"] == "ui_mass"
    assert record["prompt_token_count"] == 1_000_000
    assert record["candidates_token_count"] == 500_000
    assert record["total_token_count"] == 1_500_000
    assert record["input_cost_per_1m_snapshot"] == 0.30
    assert record["output_cost_per_1m_snapshot"] == 2.50
    assert record["input_price_per_1m_snapshot"] == 0.60
    assert record["output_price_per_1m_snapshot"] == 5.00
    assert record["estimated_cost"] == 1.55
    assert record["estimated_client_price"] == 3.10
    assert record["custom_fragment_hash"]
    assert record["analysis_result_cache_key"]
