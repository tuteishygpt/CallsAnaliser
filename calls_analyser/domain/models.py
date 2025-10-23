"""Domain models for the Calls Analyser application."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported languages for AI analysis."""

    AUTO = "default"
    RUSSIAN = "ru"
    BELARUSIAN = "be"
    ENGLISH = "en"


class CallLogEntry(BaseModel):
    """Represents a single call record returned by the telephony provider."""

    unique_id: str = Field(..., description="Unique identifier of the call recording.")
    started_at: Optional[datetime] = Field(
        default=None, description="Timestamp when the call started.",
    )
    caller_id: Optional[str] = Field(default=None, description="Caller phone number.")
    destination: Optional[str] = Field(default=None, description="Destination phone number.")
    duration_seconds: Optional[int] = Field(
        default=None, description="Duration of the call in seconds.",
    )
    raw: Dict[str, Any] = Field(default_factory=dict, description="Original payload from the provider.")


class Recording(BaseModel):
    """Raw audio recording returned by a telephony adapter."""

    unique_id: str
    content: bytes
    content_type: str = "audio/mpeg"
    source_uri: Optional[str] = None


class RecordingHandle(BaseModel):
    """Reference to a stored recording returned by :class:`CallLogService`."""

    unique_id: str
    local_uri: str
    source_uri: Optional[str] = None


class AnalysisResult(BaseModel):
    """Result returned by an AI model adapter."""

    text: str
    model: str
    provider: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
