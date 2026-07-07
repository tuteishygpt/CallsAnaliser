"""Token usage extraction and cost calculation helpers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class UsageMetadata:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0
    thoughts_token_count: int = 0


@dataclass(frozen=True)
class PricingSnapshot:
    currency: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    input_price_per_1m: float
    output_price_per_1m: float


def _read_int(source: Any, *names: str) -> int:
    for name in names:
        if isinstance(source, dict) and name in source:
            value = source.get(name)
        else:
            value = getattr(source, name, None)
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def extract_usage_metadata(source: Any) -> UsageMetadata | None:
    """Normalize Gemini usage metadata from dicts or SDK objects."""
    if source is None:
        return None

    usage = UsageMetadata(
        prompt_token_count=_read_int(source, "promptTokenCount", "prompt_token_count"),
        candidates_token_count=_read_int(
            source,
            "candidatesTokenCount",
            "candidates_token_count",
        ),
        total_token_count=_read_int(source, "totalTokenCount", "total_token_count"),
        thoughts_token_count=_read_int(source, "thoughtsTokenCount", "thoughts_token_count"),
    )
    if (
        usage.prompt_token_count == 0
        and usage.candidates_token_count == 0
        and usage.total_token_count == 0
        and usage.thoughts_token_count == 0
    ):
        return None
    return usage


def usage_metadata_to_dict(usage: UsageMetadata | None) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        "promptTokenCount": usage.prompt_token_count,
        "candidatesTokenCount": usage.candidates_token_count,
        "totalTokenCount": usage.total_token_count,
        "thoughtsTokenCount": usage.thoughts_token_count,
    }


def pricing_from_record(record: dict[str, Any]) -> PricingSnapshot:
    return PricingSnapshot(
        currency=str(record.get("currency") or "USD"),
        input_cost_per_1m=float(record.get("input_cost_per_1m") or 0),
        output_cost_per_1m=float(record.get("output_cost_per_1m") or 0),
        input_price_per_1m=float(record.get("input_price_per_1m") or 0),
        output_price_per_1m=float(record.get("output_price_per_1m") or 0),
    )


def _hash_text(value: str) -> str:
    clean = (value or "").strip()
    if not clean:
        return ""
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def _cache_key_json(cache_key: tuple[object, ...]) -> str:
    return json.dumps(list(cache_key), ensure_ascii=False, separators=(",", ":"))


def build_usage_record(
    *,
    entry: Any,
    tenant: Any,
    prompt_key: str,
    custom_fragment: str,
    provider_name: str,
    model_key: str,
    mode: str,
    usage: UsageMetadata,
    pricing: PricingSnapshot,
    cache_key: tuple[object, ...],
) -> dict[str, Any]:
    """Build the row inserted into ``analysis_usage``."""
    raw = getattr(entry, "raw", {}) or {}
    prompt_millions = usage.prompt_token_count / 1_000_000
    output_millions = usage.candidates_token_count / 1_000_000
    estimated_cost = (
        prompt_millions * pricing.input_cost_per_1m
        + output_millions * pricing.output_cost_per_1m
    )
    estimated_client_price = (
        prompt_millions * pricing.input_price_per_1m
        + output_millions * pricing.output_price_per_1m
    )
    started_at = getattr(entry, "started_at", None)

    return {
        "tenant_id": getattr(tenant, "tenant_id", ""),
        "call_unique_id": getattr(entry, "unique_id", ""),
        "call_started_at": started_at.isoformat() if started_at else None,
        "call_user": raw.get("user"),
        "caller_id": getattr(entry, "caller_id", None),
        "destination": getattr(entry, "destination", None),
        "duration_seconds": getattr(entry, "duration_seconds", None),
        "prompt_key": prompt_key,
        "custom_fragment_hash": _hash_text(custom_fragment),
        "provider_name": provider_name,
        "model_key": model_key,
        "mode": mode,
        "cache_hit": False,
        "prompt_token_count": usage.prompt_token_count,
        "candidates_token_count": usage.candidates_token_count,
        "thoughts_token_count": usage.thoughts_token_count,
        "total_token_count": usage.total_token_count,
        "input_cost_per_1m_snapshot": pricing.input_cost_per_1m,
        "output_cost_per_1m_snapshot": pricing.output_cost_per_1m,
        "input_price_per_1m_snapshot": pricing.input_price_per_1m,
        "output_price_per_1m_snapshot": pricing.output_price_per_1m,
        "estimated_cost": round(estimated_cost, 10),
        "estimated_client_price": round(estimated_client_price, 10),
        "currency": pricing.currency,
        "analysis_result_cache_key": _cache_key_json(cache_key),
    }
