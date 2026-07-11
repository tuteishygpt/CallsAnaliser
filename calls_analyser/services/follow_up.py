"""Parse model output into validated follow-up decisions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import NoReturn


@dataclass(frozen=True, slots=True)
class FollowUpDecision:
    """A validated decision about whether a call needs follow-up."""

    needs_follow_up: bool
    reason: str


class FollowUpDecisionParser:
    """Parse strict model JSON and historical cached response formats."""

    _HISTORICAL_PATTERN = re.compile(
        r"^Needs follow-up:\s*(Yes|No)\s*(?:\r?\nSummary:\s*(.*))?$",
        re.DOTALL,
    )

    @classmethod
    def parse_strict(cls, text: str) -> FollowUpDecision | None:
        """Return a decision only for a strictly valid JSON object."""
        text_clean = cls._strip_markdown_fence(text)
        try:
            payload = json.loads(
                text_clean,
                parse_constant=cls._reject_nonstandard_json_constant,
            )
        except (TypeError, ValueError):
            return None

        if not isinstance(payload, dict):
            return None

        needs_follow_up = payload.get("needs_follow_up")
        reason = payload.get("reason")
        if type(needs_follow_up) is not bool:  # noqa: E721 - exact bool is required
            return None
        if not isinstance(reason, str) or not reason.strip():
            return None

        return FollowUpDecision(
            needs_follow_up=needs_follow_up,
            reason=reason.strip(),
        )

    @classmethod
    def parse_compatibility(cls, text: str) -> FollowUpDecision | None:
        """Parse strict JSON or the historical labeled text format."""
        strict_decision = cls.parse_strict(text)
        if strict_decision is not None:
            return strict_decision

        match = cls._HISTORICAL_PATTERN.fullmatch(cls._strip_markdown_fence(text))
        if match is None:
            return None

        return FollowUpDecision(
            needs_follow_up=match.group(1) == "Yes",
            reason=(match.group(2) or "").strip(),
        )

    @staticmethod
    def _reject_nonstandard_json_constant(value: str) -> NoReturn:
        raise ValueError(f"Invalid JSON constant: {value}")

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        text_clean = str(text or "").strip()
        lines = text_clean.splitlines()
        if (
            len(lines) >= 2
            and lines[0].strip().lower() in {"```", "```json"}
            and lines[-1].strip() == "```"
        ):
            return "\n".join(lines[1:-1]).strip()
        return text_clean
