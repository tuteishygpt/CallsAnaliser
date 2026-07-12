from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from calls_analyser.domain.models import CallLogEntry
from calls_analyser.services.batch_orchestrator import BatchItemResult, RoundExecutionResult
from calls_analyser.services.batch_results import (
    CANONICAL_RESULT_COLUMNS,
    EXPORT_RESULT_COLUMNS,
    build_batch_item_row,
)
from calls_analyser.services.follow_up import FollowUpDecision
from calls_analyser.services.tenant import TenantConfig


def _item() -> BatchItemResult:
    return BatchItemResult(
        entry=CallLogEntry(
            unique_id="call-1",
            started_at=datetime(2026, 7, 12, 9, 30),
            caller_id="100",
            destination="200",
            duration_seconds=42,
            raw={"user": "operator"},
        ),
        primary=RoundExecutionResult(raw_text='{"needs_follow_up": true, "reason": "Call back"}'),
    )


TENANT = TenantConfig(tenant_id="tenant", vochi_base_url="https://calls.test")


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (
            _item(),
            {
                "Status": "⏳ Primary analysis",
                "Needs follow-up": "",
                "Reason": "",
                "Initial needs follow-up": "",
                "Initial reason": "",
                "Verification needs follow-up": "",
                "Verification reason": "",
                "Verification status": "not_requested",
            },
        ),
        (
            replace(
                _item(),
                primary_decision=FollowUpDecision(True, "Call back"),
                primary_decision_status="valid",
                verification_status="pending",
            ),
            {
                "Status": "⏳ Verification",
                "Needs follow-up": "",
                "Reason": "",
                "Initial needs follow-up": "Yes",
                "Initial reason": "Call back",
                "Verification needs follow-up": "",
                "Verification reason": "",
                "Verification status": "pending",
            },
        ),
        (
            replace(
                _item(),
                primary_decision=FollowUpDecision(True, "Call back"),
                primary_decision_status="valid",
                verification=RoundExecutionResult(raw_text="verified"),
                verification_decision=FollowUpDecision(False, "Resolved"),
                verification_decision_status="valid",
                final_decision=False,
                final_reason="Resolved",
                final_status="complete",
                verification_status="complete",
            ),
            {
                "Status": "✅",
                "Needs follow-up": "No",
                "Reason": "Resolved",
                "Initial needs follow-up": "Yes",
                "Initial reason": "Call back",
                "Verification needs follow-up": "No",
                "Verification reason": "Resolved",
                "Verification status": "complete",
            },
        ),
        (
            replace(
                _item(),
                primary_decision=FollowUpDecision(True, "Call back"),
                primary_decision_status="valid",
                verification=RoundExecutionResult(execution_status="error", execution_error="timeout"),
                final_decision=True,
                final_reason="Call back",
                final_status="fallback",
                verification_status="failed",
            ),
            {
                "Status": "⚠️",
                "Needs follow-up": "Yes",
                "Reason": "Call back",
                "Initial needs follow-up": "Yes",
                "Initial reason": "Call back",
                "Verification needs follow-up": "",
                "Verification reason": "timeout",
                "Verification status": "failed",
            },
        ),
        (
            replace(
                _item(),
                primary=RoundExecutionResult(execution_status="error", execution_error="network down"),
                final_reason="network down",
                final_status="error",
            ),
            {
                "Status": "❌",
                "Needs follow-up": "",
                "Reason": "network down",
                "Initial needs follow-up": "",
                "Initial reason": "network down",
                "Verification needs follow-up": "",
                "Verification reason": "",
                "Verification status": "not_requested",
            },
        ),
        (
            replace(_item(), primary_decision_status="invalid", final_status="invalid"),
            {
                "Status": "❌",
                "Needs follow-up": "",
                "Reason": "Invalid primary response: could not parse follow-up decision.",
                "Initial needs follow-up": "",
                "Initial reason": "Invalid primary response: could not parse follow-up decision.",
                "Verification needs follow-up": "",
                "Verification reason": "",
                "Verification status": "not_requested",
            },
        ),
    ],
)
def test_build_batch_item_row_projects_state(item: BatchItemResult, expected: dict[str, str]) -> None:
    row = build_batch_item_row(item, TENANT)

    assert {key: row[key] for key in expected} == expected
    assert row["UniqueId"] == "call-1"
    assert row["user"] == "operator"
    assert row["Link"] == '<a href="https://calls.test/recording/call-1" target="_blank">Listen</a>'


def test_invalid_verification_keeps_diagnostic_reason() -> None:
    item = replace(
        _item(),
        primary_decision=FollowUpDecision(True, "Call back"),
        primary_decision_status="valid",
        verification=RoundExecutionResult(raw_text="not a strict decision"),
        verification_decision_status="invalid",
        final_decision=True,
        final_reason="Call back",
        final_status="fallback",
        verification_status="failed",
    )

    row = build_batch_item_row(item, TENANT)

    assert row["Verification reason"] == (
        "Invalid verification response: could not parse follow-up decision."
    )


def test_canonical_row_and_export_schemas_have_one_stable_order() -> None:
    row = build_batch_item_row(_item(), TENANT)

    assert list(row) == CANONICAL_RESULT_COLUMNS
    assert EXPORT_RESULT_COLUMNS == [
        column for column in CANONICAL_RESULT_COLUMNS if column != "UniqueId"
    ]


def test_recording_link_escapes_attribute_injection() -> None:
    entry = _item().entry.model_copy(
        update={"raw": {"recording_url": 'https://calls.test/recording/1" onmouseover="alert(1)'}},
    )

    row = build_batch_item_row(replace(_item(), entry=entry), TENANT)

    assert row["Link"] == (
        '<a href="https://calls.test/recording/1&quot; onmouseover=&quot;alert(1)" '
        'target="_blank">Listen</a>'
    )


def test_recording_link_rejects_non_http_scheme() -> None:
    entry = _item().entry.model_copy(update={"raw": {"recording_url": "javascript:alert(1)"}})

    row = build_batch_item_row(replace(_item(), entry=entry), TENANT)

    assert row["Link"] == ""


@pytest.mark.parametrize("url", ["http://[", "https:///recording/call-1"])
def test_recording_link_rejects_malformed_or_hostless_http_url(url: str) -> None:
    entry = _item().entry.model_copy(update={"raw": {"recording_url": url}})

    row = build_batch_item_row(replace(_item(), entry=entry), TENANT)

    assert row["Link"] == ""
