"""Convert call-analysis output into the shared batch result row format."""
from __future__ import annotations

from calls_analyser.services.follow_up import FollowUpDecisionParser


def parse_follow_up_fields(text: str) -> tuple[str, str]:
    """Extract follow-up decision and reason from supported model formats."""
    text_clean = str(text or "").strip()
    decision = FollowUpDecisionParser.parse_compatibility(text_clean)
    if decision is not None:
        return (
            "Yes" if decision.needs_follow_up else "No",
            decision.reason,
        )
    return "", text_clean


def build_success_row(entry, tenant, text: str) -> dict[str, object]:  # noqa: ANN001
    """Build a successful result row matching the Gradio batch table."""
    raw = getattr(entry, "raw", {}) or {}
    link_key = "record" if getattr(tenant, "provider", "").lower() == "mts_vats" else "recording_url"
    link = str(raw.get(link_key) or "").strip() or tenant.recording_url(entry.unique_id)
    needs, reason = parse_follow_up_fields(text)
    user = raw.get("user")
    return {
        "Start": entry.started_at.isoformat() if entry.started_at else "",
        "Caller": entry.caller_id or "",
        "Destination": entry.destination or "",
        "Duration (s)": entry.duration_seconds,
        "UniqueId": entry.unique_id,
        "Needs follow-up": needs,
        "Reason": reason,
        "Link": f'<a href="{link}" target="_blank">Listen</a>' if link else "",
        "Status": "✅",
        **({"user": user} if user not in (None, "") else {}),
    }


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

