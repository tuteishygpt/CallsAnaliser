"""Simple provider registry used for dependency lookups."""
from __future__ import annotations

from typing import Generic, Iterable, Iterator, MutableMapping, TypeVar

T = TypeVar("T")


class ProviderRegistry(Generic[T]):
    """Registry mapping provider keys to implementations."""

    def __init__(self) -> None:
        self._providers: MutableMapping[str, T] = {}

    def register(self, key: str, provider: T) -> None:
        """Register a provider with the given key."""

        self._providers[key] = provider

    def get(self, key: str) -> T:
        """Return the provider registered for ``key``."""

        try:
            return self._providers[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers)) or "<empty>"
            raise KeyError(f"Provider '{key}' is not registered. Available: {available}") from exc

    def __contains__(self, key: object) -> bool:
        return key in self._providers

    def keys(self) -> Iterable[str]:
        return self._providers.keys()

    def values(self) -> Iterable[T]:
        return self._providers.values()

    def items(self) -> Iterable[tuple[str, T]]:
        return self._providers.items()

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return iter(self._providers.items())

    def __len__(self) -> int:
        return len(self._providers)
