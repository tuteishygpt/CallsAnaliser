"""Convert call-analysis output into the shared batch result row format."""
from __future__ import annotations

from html import escape
from urllib.parse import urlsplit

from calls_analyser.services.follow_up import FollowUpDecisionParser


INVALID_PRIMARY_REASON = "Invalid primary response: could not parse follow-up decision."
INVALID_VERIFICATION_REASON = (
    "Invalid verification response: could not parse follow-up decision."
)
CANONICAL_RESULT_COLUMNS = [
    "Start",
    "Caller",
    "Destination",
    "user",
    "Duration (s)",
    "UniqueId",
    "Needs follow-up",
    "Reason",
    "Initial needs follow-up",
    "Initial reason",
    "Verification needs follow-up",
    "Verification reason",
    "Verification status",
    "Link",
    "Status",
]
EXPORT_RESULT_COLUMNS = [
    column for column in CANONICAL_RESULT_COLUMNS if column != "UniqueId"
]


def _yes_no(value: bool | None) -> str:
    if value is None:
        return ""
    return "Yes" if value else "No"


def _recording_link(url: str) -> str:
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.netloc)
            and bool(parsed.hostname)
        )
    except ValueError:
        return ""
    if not valid:
        return ""
    return f'<a href="{escape(url, quote=True)}" target="_blank">Listen</a>'


def build_batch_item_row(item, tenant) -> dict[str, object]:  # noqa: ANN001
    """Project one orchestrated batch item into the canonical result row."""
    entry = item.entry
    raw = getattr(entry, "raw", {}) or {}
    link_key = "record" if getattr(tenant, "provider", "").lower() == "mts_vats" else "recording_url"
    link = str(raw.get(link_key) or "").strip() or tenant.recording_url(entry.unique_id)

    initial_reason = ""
    if item.primary_decision is not None:
        initial_reason = item.primary_decision.reason
    elif item.primary.execution_status != "success":
        initial_reason = item.primary.execution_error or ""
    elif item.primary_decision_status == "invalid" or item.final_status == "invalid":
        initial_reason = INVALID_PRIMARY_REASON

    verification_reason = ""
    if item.verification_decision is not None:
        verification_reason = item.verification_decision.reason
    elif item.verification is not None:
        verification_reason = item.verification.execution_error or ""
        if (
            not verification_reason
            and item.verification.execution_status == "success"
            and item.verification_decision_status == "invalid"
        ):
            verification_reason = INVALID_VERIFICATION_REASON

    if item.final_status == "pending":
        status = "⏳ Verification" if item.verification_status == "pending" else "⏳ Primary analysis"
        needs = ""
        reason = ""
    elif item.final_status == "complete":
        status = "✅"
        needs = _yes_no(item.final_decision)
        reason = item.final_reason or ""
    elif item.final_status == "fallback":
        status = "⚠️"
        needs = _yes_no(item.final_decision)
        reason = item.final_reason or ""
    else:
        status = "❌"
        needs = ""
        reason = initial_reason

    user = raw.get("user")
    values = {
        "Start": entry.started_at.isoformat() if entry.started_at else "",
        "Caller": entry.caller_id or "",
        "Destination": entry.destination or "",
        "user": user or "",
        "Duration (s)": entry.duration_seconds,
        "UniqueId": entry.unique_id,
        "Needs follow-up": needs,
        "Reason": reason,
        "Initial needs follow-up": _yes_no(
            item.primary_decision.needs_follow_up if item.primary_decision is not None else None
        ),
        "Initial reason": initial_reason,
        "Verification needs follow-up": _yes_no(
            item.verification_decision.needs_follow_up
            if item.verification_decision is not None
            else None
        ),
        "Verification reason": verification_reason,
        "Verification status": item.verification_status,
        "Link": _recording_link(link) if link else "",
        "Status": status,
    }
    return {column: values[column] for column in CANONICAL_RESULT_COLUMNS}


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

