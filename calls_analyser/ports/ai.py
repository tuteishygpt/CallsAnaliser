"""AI model port."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol

from calls_analyser.domain.models import AnalysisResult, Language


class AudioSource(Protocol):
    """Protocol describing how audio is passed to AI adapters."""

    path: str | None
    content: bytes | None


class AIModelPort(ABC):
    """Interface for AI model providers."""

    provider_name: str

    @abstractmethod
    def analyze(
        self,
        audio: AudioSource,
        prompt: str,
        lang: Language,
        options: Mapping[str, Any] | None = None,
    ) -> AnalysisResult:
        """Analyze the audio and return a structured result."""
