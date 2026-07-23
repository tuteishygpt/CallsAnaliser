"""Convert call-analysis output into the shared batch result row format."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FollowUpResult:
    needs_follow_up: bool
    reason: str


_COMPLETE_JSON_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z",
    re.DOTALL,
)


def parse_follow_up_result(text: str) -> FollowUpResult:
    """Parse one complete plain/fenced JSON object or raise ValueError."""
    text_clean = str(text or "").strip()
    fenced = _COMPLETE_JSON_FENCE.fullmatch(text_clean)
    if fenced:
        text_clean = fenced.group("body").strip()

    try:
        payload = json.loads(text_clean)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("follow-up result must be one complete JSON object") from exc

    if not isinstance(payload, dict):
        raise ValueError("follow-up result must be a JSON object")
    if "needs_follow_up" not in payload or "reason" not in payload:
        raise ValueError("follow-up result requires needs_follow_up and reason")
    if type(payload["needs_follow_up"]) is not bool:
        raise ValueError("needs_follow_up must be a boolean")
    if not isinstance(payload["reason"], str):
        raise ValueError("reason must be a string")

    return FollowUpResult(
        needs_follow_up=payload["needs_follow_up"],
        reason=payload["reason"],
    )


def parse_follow_up_fields(text: str) -> tuple[str, str]:
    """Return display fields from a strictly parsed follow-up result."""
    result = parse_follow_up_result(text)
    return ("Yes" if result.needs_follow_up else "No", result.reason)


def build_success_row(
    entry,
    tenant,
    result: FollowUpResult,
) -> dict[str, object]:  # noqa: ANN001
    """Build a successful result row matching the Gradio batch table."""
    raw = getattr(entry, "raw", {}) or {}
    link_key = "record" if getattr(tenant, "provider", "").lower() == "mts_vats" else "recording_url"
    link = str(raw.get(link_key) or "").strip() or tenant.recording_url(entry.unique_id)
    user = raw.get("user")
    return {
        "Start": entry.started_at.isoformat() if entry.started_at else "",
        "Caller": entry.caller_id or "",
        "Destination": entry.destination or "",
        "Duration (s)": entry.duration_seconds,
        "UniqueId": entry.unique_id,
        "Needs follow-up": "Yes" if result.needs_follow_up else "No",
        "Reason": result.reason,
        "Link": f'<a href="{link}" target="_blank">Listen</a>' if link else "",
        "Status": "✅",
        **({"user": user} if user not in (None, "") else {}),
    }


BatchRunStatus = Literal["success", "partial", "failed"]


@dataclass(frozen=True)
class BatchRunResult:
    status: BatchRunStatus
    total_count: int
    success_count: int
    failure_count: int
    cached_count: int = 0

    @classmethod
    def from_counts(
        cls,
        *,
        total_count: int,
        success_count: int,
        failure_count: int,
        cached_count: int = 0,
    ) -> "BatchRunResult":
        status: BatchRunStatus = (
            "success"
            if failure_count == 0
            else "partial"
            if success_count > 0
            else "failed"
        )
        return cls(status, total_count, success_count, failure_count, cached_count)


def build_error_row(entry, reason: str) -> dict[str, object]:  # noqa: ANN001
    """Build a failed result row matching the Gradio batch table."""
    raw = getattr(entry, "raw", {}) or {}
    user = raw.get("user")
    return {
        "Start": entry.started_at.isoformat() if entry.started_at else "",
        "Caller": entry.caller_id or "",
        "Destination": entry.destination or "",
        "Duration (s)": entry.duration_seconds,
        "UniqueId": entry.unique_id,
        "Needs follow-up": "",
        "Reason": reason,
        "Link": "",
        "Status": "❌",
        **({"user": user} if user not in (None, "") else {}),
    }
