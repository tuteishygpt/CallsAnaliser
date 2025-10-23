"""Storage port."""
from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike


class StoragePort(ABC):
    """Abstract storage backend for recordings."""

    @abstractmethod
    def uri_for(self, file_name: str) -> str:
        """Return the URI that would be used for ``file_name`` without writing."""

    @abstractmethod
    def save_file(self, data: bytes | str | PathLike[str], file_name: str) -> str:
        """Persist ``data`` or copy from the given path and return a URI."""

    @abstractmethod
    def open(self, uri: str) -> bytes:
        """Return the stored bytes for ``uri``."""

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Return whether the URI is present in storage."""
