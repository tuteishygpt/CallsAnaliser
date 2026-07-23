from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from types import SimpleNamespace

import pytest

from calls_analyser.services.batch_results import (
    BatchRunResult,
    FollowUpResult,
    build_success_row,
    parse_follow_up_result,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '{"needs_follow_up": true, "reason": "Call tomorrow"}',
            FollowUpResult(needs_follow_up=True, reason="Call tomorrow"),
        ),
        (
            '{"needs_follow_up": false, "reason": ""}',
            FollowUpResult(needs_follow_up=False, reason=""),
        ),
        (
            '```json\n{"needs_follow_up": true, "reason": "Interested"}\n```',
            FollowUpResult(needs_follow_up=True, reason="Interested"),
        ),
        (
            '```\n{"needs_follow_up": false, "reason": "Resolved"}\n```',
            FollowUpResult(needs_follow_up=False, reason="Resolved"),
        ),
    ],
)
def test_parse_follow_up_result_accepts_complete_plain_or_fenced_json(
    text: str,
    expected: FollowUpResult,
) -> None:
    assert parse_follow_up_result(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        '{"needs_follow_up": true}',
        '{"reason": "Missing decision"}',
        '{"needs_follow_up": "true", "reason": "Wrong type"}',
        '{"needs_follow_up": 1, "reason": "Wrong type"}',
        '{"needs_follow_up": null, "reason": "Wrong type"}',
        '{"needs_follow_up": true, "reason": 7}',
        '{"needs_follow_up": true, "reason": null}',
        '["needs_follow_up", true]',
        '{"needs_follow_up": true, "reason": "Valid"} trailing prose',
        'prefix {"needs_follow_up": true, "reason": "Valid"}',
        '```json\n{"needs_follow_up": true, "reason": "Valid"}\n```\ntrailing',
        "Needs follow-up: Yes\nSummary: Call tomorrow",
    ],
)
def test_parse_follow_up_result_rejects_invalid_or_partial_formats(text: str) -> None:
    with pytest.raises(ValueError):
        parse_follow_up_result(text)


def test_follow_up_result_is_frozen() -> None:
    result = FollowUpResult(needs_follow_up=True, reason="Call")

    with pytest.raises(FrozenInstanceError):
        result.reason = "Changed"  # type: ignore[misc]


def test_build_success_row_uses_preparsed_result() -> None:
    entry = SimpleNamespace(
        started_at=datetime(2026, 7, 23, 10, 30),
        caller_id="+100",
        destination="+200",
        duration_seconds=42,
        unique_id="call-1",
        raw={},
    )
    tenant = SimpleNamespace(
        provider="vochi",
        recording_url=lambda unique_id: f"https://example.test/{unique_id}",
    )

    row = build_success_row(
        entry,
        tenant,
        FollowUpResult(needs_follow_up=False, reason="Resolved"),
    )

    assert row["Needs follow-up"] == "No"
    assert row["Reason"] == "Resolved"


@pytest.mark.parametrize(
    ("success_count", "failure_count", "expected_status"),
    [
        (0, 0, "success"),
        (3, 0, "success"),
        (2, 1, "partial"),
        (0, 3, "failed"),
    ],
)
def test_batch_run_result_derives_status_from_counts(
    success_count: int,
    failure_count: int,
    expected_status: str,
) -> None:
    result = BatchRunResult.from_counts(
        total_count=success_count + failure_count,
        success_count=success_count,
        failure_count=failure_count,
        cached_count=1,
    )

    assert result.status == expected_status
    assert result.cached_count == 1


def test_batch_run_result_is_frozen() -> None:
    result = BatchRunResult.from_counts(
        total_count=0,
        success_count=0,
        failure_count=0,
    )

    with pytest.raises(FrozenInstanceError):
        result.status = "failed"  # type: ignore[misc]
