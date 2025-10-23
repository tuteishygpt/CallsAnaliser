"""Local filesystem storage adapter."""
from __future__ import annotations

from pathlib import Path

from calls_analyser.domain.exceptions import StorageError
from calls_analyser.ports.storage import StoragePort


class LocalStorageAdapter(StoragePort):
    """Stores files under a base directory on the local filesystem."""

    def __init__(self, base_dir: str | Path = "/tmp") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def uri_for(self, file_name: str) -> str:
        return str(self._base_dir / file_name)

    def save(self, data: bytes, file_name: str) -> str:
        uri = self.uri_for(file_name)
        try:
            with open(uri, "wb") as fh:
                fh.write(data)
        except OSError as exc:
            raise StorageError(f"Failed to save recording to {uri}") from exc
        return uri

    def open(self, uri: str) -> bytes:
        try:
            with open(uri, "rb") as fh:
                return fh.read()
        except OSError as exc:
            raise StorageError(f"Failed to open recording at {uri}") from exc

    def exists(self, uri: str) -> bool:
        return Path(uri).exists()
