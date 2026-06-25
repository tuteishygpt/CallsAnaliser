"""Supabase-backed token usage tracker."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from supabase import Client, create_client

from calls_analyser.services.usage import (
    UsageMetadata,
    build_usage_record,
    pricing_from_record,
)

logger = logging.getLogger(__name__)


class SupabaseUsageTracker:
    """Records paid Gemini usage rows using pricing from Supabase."""

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        self._client: Client = create_client(supabase_url, supabase_key)
        self._pricing_table = self._client.table("model_pricing")
        self._usage_table = self._client.table("analysis_usage")

    def _lookup_pricing(self, provider_name: str, model_key: str) -> dict[str, Any] | None:
        today = dt.date.today().isoformat()
        response = (
            self._pricing_table.select("*")
            .eq("provider", provider_name)
            .eq("model_key", model_key)
            .eq("is_active", True)
            .lte("effective_from", today)
            .or_(f"effective_to.is.null,effective_to.gte.{today}")
            .order("effective_from", desc=True)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    def record(
        self,
        *,
        entry: Any,
        tenant: Any,
        prompt_key: str,
        custom_fragment: str,
        provider_name: str,
        model_key: str,
        mode: str,
        usage: UsageMetadata | None,
        cache_key: tuple[str, str, str, str, str, str],
    ) -> None:
        """Insert one paid usage row; failures are logged and non-fatal."""
        if usage is None:
            return
        try:
            pricing_record = self._lookup_pricing(provider_name, model_key)
            if not pricing_record:
                logger.warning(
                    "Skipping usage record for %s: no active pricing for %s/%s",
                    getattr(entry, "unique_id", ""),
                    provider_name,
                    model_key,
                )
                return
            row = build_usage_record(
                entry=entry,
                tenant=tenant,
                prompt_key=prompt_key,
                custom_fragment=custom_fragment,
                provider_name=provider_name,
                model_key=model_key,
                mode=mode,
                usage=usage,
                pricing=pricing_from_record(pricing_record),
                cache_key=cache_key,
            )
            self._usage_table.insert(row).execute()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to record usage for %s: %s",
                getattr(entry, "unique_id", ""),
                exc,
            )
