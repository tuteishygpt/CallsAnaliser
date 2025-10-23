"""Telephony port interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable

from calls_analyser.domain.models import CallLogEntry, Recording


class TelephonyPort(ABC):
    """Abstract port for telephony providers."""

    @abstractmethod
    def list_calls(self, day: date, tenant_id: str) -> Iterable[CallLogEntry]:
        """Return all call log entries for the given tenant and day."""

    @abstractmethod
    def get_recording(self, unique_id: str, tenant_id: str) -> Recording:
        """Return the raw recording for the given unique identifier."""
