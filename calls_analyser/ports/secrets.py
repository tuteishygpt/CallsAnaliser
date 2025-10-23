"""Secrets port."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class SecretsPort(ABC):
    """Interface for retrieving configuration secrets."""

    @abstractmethod
    def get_secret(self, key: str, tenant_id: Optional[str] = None) -> str:
        """Return the secret value or raise :class:`SecretsError`."""

    @abstractmethod
    def get_optional_secret(self, key: str, tenant_id: Optional[str] = None) -> Optional[str]:
        """Return the optional secret value if available."""
