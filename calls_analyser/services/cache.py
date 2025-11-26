"""File-backed cache implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, MutableMapping, Any

from calls_analyser.domain.models import AnalysisResult
from calls_analyser.services.analysis import CacheKey


class FileBackedCache(MutableMapping[CacheKey, AnalysisResult]):
    """
    A persistent cache that stores analysis results in a JSON file.
    
    It keeps the data in memory for fast access and syncs to disk on updates.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._data: dict[CacheKey, AnalysisResult] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not self._file_path.exists():
            return

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            for key_str, value_dict in raw_data.items():
                # Key is stored as a string representation of the tuple, or a joined string.
                # Since JSON keys must be strings, we need a reliable way to serialize/deserialize.
                # A simple way is to use a separator that is unlikely to be in the data, 
                # but the data includes prompts which can be anything.
                # However, the CacheKey components are:
                # (tenant_id, unique_id, prompt_key, provider_name, model_key, custom_fragment)
                # We can try to rely on JSON serialization of the list of keys if we store it as a list of entries instead of a dict in the file?
                # But for a dict-like usage, we need a hashable key.
                
                # Let's assume we store the key as a JSON string of the list.
                try:
                    key_parts = json.loads(key_str)
                    if isinstance(key_parts, list) and len(key_parts) == 6:
                        key = tuple(key_parts) # type: ignore
                        self._data[key] = AnalysisResult(**value_dict)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

        except (OSError, json.JSONDecodeError):
            # If file is corrupted or unreadable, start fresh
            pass

    def _save(self) -> None:
        """Save cache to disk."""
        # Ensure directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        raw_data = {}
        for key, value in self._data.items():
            # Serialize key as JSON string
            key_str = json.dumps(list(key))
            raw_data[key_str] = value.model_dump()

        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)
        except (OSError, TypeError, ValueError):
            # Best effort save - ignore if serialization fails
            pass

    def __getitem__(self, key: CacheKey) -> AnalysisResult:
        return self._data[key]

    def __setitem__(self, key: CacheKey, value: AnalysisResult) -> None:
        self._data[key] = value
        self._save()

    def __delitem__(self, key: CacheKey) -> None:
        del self._data[key]
        self._save()

    def __iter__(self) -> Iterator[CacheKey]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)
