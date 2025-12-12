"""Supabase-backed cache implementation."""
from __future__ import annotations

import json
from typing import Iterator, MutableMapping, Any

from supabase import Client, create_client
from postgrest.base_request_builder import APIResponse

from calls_analyser.domain.models import AnalysisResult
from calls_analyser.services.analysis import CacheKey


class SupabaseCache(MutableMapping[CacheKey, AnalysisResult]):
    """
    A persistent cache that stores analysis results in a Supabase database.
    
    It maps the CacheKey tuple to columns in the 'analysis_results' table.
    """

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        self._client: Client = create_client(supabase_url, supabase_key)
        self._table = self._client.table("analysis_results")
        # Local cache to reduce DB hits for repeated access in short term, 
        # specifically useful if the object is reused within a session.
        # However, for a proper distributed cache, we should query DB.
        # Implemented as a write-through cache.
        self._local_cache: dict[CacheKey, AnalysisResult] = {}

    def _key_to_dict(self, key: CacheKey) -> dict[str, str]:
        """Convert CacheKey tuple to dictionary for DB queries."""
        return {
            "tenant_id": key[0],
            "call_unique_id": key[1],
            "prompt_key": key[2],
            "provider_name": key[3],
            "model_key": key[4],
            "custom_fragment": key[5],
        }

    def _record_to_result(self, record: dict[str, Any]) -> AnalysisResult:
        """Convert DB record to AnalysisResult."""
        return AnalysisResult(
            text=record["result_text"],
            model=record["model_key"],
            provider=record["provider_name"],
            metadata=record.get("metadata", {}) or {},
        )

    def __getitem__(self, key: CacheKey) -> AnalysisResult:
        # Check local cache first (optional optimization)
        if key in self._local_cache:
            return self._local_cache[key]

        # Query Supabase
        query_params = self._key_to_dict(key)
        response: APIResponse = self._table.select("*").match(query_params).execute()

        if not response.data:
            raise KeyError(key)

        # Assume unique constraint ensures at most one result
        record = response.data[0]
        result = self._record_to_result(record)
        
        # Update local cache
        self._local_cache[key] = result
        return result

    def __setitem__(self, key: CacheKey, value: AnalysisResult) -> None:
        # Prepare data for insertion
        data = self._key_to_dict(key)
        data.update({
            "result_text": value.text,
            "metadata": value.metadata,
        })

        # Upsert into Supabase
        # on_conflict match the unique columns
        self._table.upsert(data, on_conflict="tenant_id, call_unique_id, prompt_key, provider_name, model_key, custom_fragment").execute()
        
        # Update local cache
        self._local_cache[key] = value

    def __delitem__(self, key: CacheKey) -> None:
        query_params = self._key_to_dict(key)
        self._table.delete().match(query_params).execute()
        
        if key in self._local_cache:
            del self._local_cache[key]

    def __iter__(self) -> Iterator[CacheKey]:
        # This is expensive and potentially dangerous for large tables.
        # Implementing for interface completeness but should be used with caution.
        # Fetching only keys.
        response = self._table.select("tenant_id, call_unique_id, prompt_key, provider_name, model_key, custom_fragment").execute()
        
        for record in response.data:
            yield (
                record["tenant_id"],
                record["call_unique_id"],
                record["prompt_key"],
                record["provider_name"],
                record["model_key"],
                record["custom_fragment"],
            )

    def __len__(self) -> int:
        # Use count="exact" header or similar efficient count if available.
        # Supabase-py count:
        response = self._table.select("*", count="exact").execute()
        return response.count if response.count is not None else 0
