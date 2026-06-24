"""Convert call-analysis output into the shared batch result row format."""
from __future__ import annotations

import json


def parse_follow_up_fields(text: str) -> tuple[str, str]:
    """Extract follow-up decision and reason from supported model formats."""
    text_clean = str(text or "").strip()
    if text_clean.startswith("```"):
        lines = text_clean.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        text_clean = "\n".join(lines).strip()
    try:
        left, right = text_clean.find("{"), text_clean.rfind("}")
        if left != -1 and right > left:
            text_clean = text_clean[left : right + 1]
        payload = json.loads(text_clean)
        return (
            "Yes" if payload.get("needs_follow_up") else "No",
            str(payload.get("reason") or ""),
        )
    except Exception:
        if "Needs follow-up:" in text_clean:
            parts = text_clean.split("Summary:", 1)
            return (
                parts[0].replace("Needs follow-up:", "").strip(),
                parts[1].strip() if len(parts) > 1 else "",
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

