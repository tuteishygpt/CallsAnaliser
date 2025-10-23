"""Domain specific exceptions."""
from __future__ import annotations


class CallsAnalyserError(Exception):
    """Base exception for the application."""


class TelephonyError(CallsAnalyserError):
    """Raised when a telephony adapter fails."""


class AIModelError(CallsAnalyserError):
    """Raised when an AI adapter fails."""


class StorageError(CallsAnalyserError):
    """Raised when a storage adapter fails."""


class SecretsError(CallsAnalyserError):
    """Raised when secrets cannot be retrieved."""
