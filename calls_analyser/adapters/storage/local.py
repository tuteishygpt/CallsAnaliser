"""Local filesystem storage adapter."""
from __future__ import annotations

from os import PathLike, fspath
from pathlib import Path
from shutil import copyfileobj

from calls_analyser.domain.exceptions import StorageError
from calls_analyser.ports.storage import StoragePort


class LocalStorageAdapter(StoragePort):
    """Stores files under a base directory on the local filesystem."""

    def __init__(self, base_dir: str | Path = "/tmp") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def uri_for(self, file_name: str) -> str:
        return str(self._base_dir / file_name)

    def save_file(self, data: bytes | str | PathLike[str], file_name: str) -> str:
        """Store ``data`` or copy from an existing file into the base directory."""
        uri = self.uri_for(file_name)
        try:
            if isinstance(data, (bytes, bytearray)):
                with open(uri, "wb") as dest:
                    dest.write(data)
            else:
                source = Path(fspath(data))
                with open(source, "rb") as src, open(uri, "wb") as dest:
                    copyfileobj(src, dest)
        except (OSError, TypeError) as exc:
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
