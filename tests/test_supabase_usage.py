from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import patch

from calls_analyser.adapters.storage.supabase_usage import SupabaseUsageTracker
from calls_analyser.services.usage import UsageMetadata


class _FakeTable:
    def __init__(self, response_data=None) -> None:  # noqa: ANN001
        self.response_data = response_data or []
        self.calls = []

    def select(self, value):  # noqa: ANN001
        self.calls.append(("select", value))
        return self

    def eq(self, key, value):  # noqa: ANN001
        self.calls.append(("eq", key, value))
        return self

    def lte(self, key, value):  # noqa: ANN001
        self.calls.append(("lte", key, value))
        return self

    def or_(self, value):  # noqa: ANN001
        self.calls.append(("or", value))
        return self

    def order(self, key, desc=False):  # noqa: ANN001
        self.calls.append(("order", key, desc))
        return self

    def limit(self, value):  # noqa: ANN001
        self.calls.append(("limit", value))
        return self

    def insert(self, value):  # noqa: ANN001
        self.calls.append(("insert", value))
        return self

    def execute(self):
        return SimpleNamespace(data=self.response_data)


class _FakeClient:
    def __init__(self) -> None:
        self.pricing = _FakeTable(
            [
                {
                    "currency": "USD",
                    "input_cost_per_1m": 0.30,
                    "output_cost_per_1m": 2.50,
                    "input_price_per_1m": 0.60,
                    "output_price_per_1m": 5.00,
                }
            ]
        )
        self.usage = _FakeTable()

    def table(self, name):
        if name == "model_pricing":
            return self.pricing
        if name == "analysis_usage":
            return self.usage
        raise AssertionError(name)


def test_supabase_usage_tracker_reads_pricing_and_inserts_usage() -> None:
    fake_client = _FakeClient()
    with patch(
        "calls_analyser.adapters.storage.supabase_usage.create_client",
        return_value=fake_client,
    ):
        tracker = SupabaseUsageTracker("https://example.supabase.co", "key")

    entry = SimpleNamespace(
        unique_id="call-1",
        started_at=dt.datetime(2026, 6, 25, 9, 0),
        caller_id="client",
        destination="agent",
        duration_seconds=120,
        raw={"user": "operator"},
    )
    tenant = SimpleNamespace(tenant_id="lix")

    tracker.record(
        entry=entry,
        tenant=tenant,
        prompt_key="follow_up",
        custom_fragment="",
        provider_name="gemini",
        model_key="models/gemini-test",
        mode="scheduler_batch",
        usage=UsageMetadata(
            prompt_token_count=1000,
            candidates_token_count=500,
            total_token_count=1500,
            thoughts_token_count=0,
        ),
        cache_key=("lix", "call-1", "follow_up", "gemini", "models/gemini-test", ""),
    )

    pricing_filters = fake_client.pricing.calls
    assert ("eq", "provider", "gemini") in pricing_filters
    assert ("eq", "model_key", "models/gemini-test") in pricing_filters
    inserted = [call for call in fake_client.usage.calls if call[0] == "insert"][0][1]
    assert inserted["tenant_id"] == "lix"
    assert inserted["call_unique_id"] == "call-1"
    assert inserted["mode"] == "scheduler_batch"
    assert inserted["estimated_cost"] > 0
    assert inserted["estimated_client_price"] > inserted["estimated_cost"]


def test_supabase_usage_tracker_skips_when_no_pricing_exists() -> None:
    fake_client = _FakeClient()
    fake_client.pricing.response_data = []
    with patch(
        "calls_analyser.adapters.storage.supabase_usage.create_client",
        return_value=fake_client,
    ):
        tracker = SupabaseUsageTracker("https://example.supabase.co", "key")

    tracker.record(
        entry=SimpleNamespace(unique_id="call-1", raw={}),
        tenant=SimpleNamespace(tenant_id="lix"),
        prompt_key="follow_up",
        custom_fragment="",
        provider_name="gemini",
        model_key="models/gemini-test",
        mode="ui_direct",
        usage=UsageMetadata(prompt_token_count=1, candidates_token_count=1, total_token_count=2),
        cache_key=("lix", "call-1", "follow_up", "gemini", "models/gemini-test", ""),
    )

    assert not [call for call in fake_client.usage.calls if call[0] == "insert"]
