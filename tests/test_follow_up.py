from __future__ import annotations

import pytest

from calls_analyser.services.batch_results import parse_follow_up_fields
from calls_analyser.services.follow_up import FollowUpDecision, FollowUpDecisionParser


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '{"needs_follow_up": true, "reason": "Call the customer"}',
            FollowUpDecision(needs_follow_up=True, reason="Call the customer"),
        ),
        (
            '{"needs_follow_up": false, "reason": "Issue resolved"}',
            FollowUpDecision(needs_follow_up=False, reason="Issue resolved"),
        ),
    ],
)
def test_parse_strict_accepts_json_booleans(
    text: str,
    expected: FollowUpDecision,
) -> None:
    assert FollowUpDecisionParser.parse_strict(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"needs_follow_up": true, "reason": "Call back"}\n```',
        '```\n{"needs_follow_up": true, "reason": "Call back"}\n```',
    ],
)
def test_parse_strict_accepts_markdown_fenced_json(text: str) -> None:
    assert FollowUpDecisionParser.parse_strict(text) == FollowUpDecision(
        needs_follow_up=True,
        reason="Call back",
    )


@pytest.mark.parametrize(
    "needs_follow_up",
    ['"true"', '"false"', "1", "0", "null"],
)
def test_parse_strict_rejects_non_boolean_decisions(needs_follow_up: str) -> None:
    text = f'{{"needs_follow_up": {needs_follow_up}, "reason": "A reason"}}'

    assert FollowUpDecisionParser.parse_strict(text) is None


@pytest.mark.parametrize(
    "text",
    [
        '{"reason": "A reason"}',
        '{"needs_follow_up": true}',
        '{"needs_follow_up": true, "reason": ""}',
        '{"needs_follow_up": true, "reason": "   "}',
        '{"needs_follow_up": true, "reason": 123}',
    ],
)
def test_parse_strict_rejects_missing_or_invalid_fields(text: str) -> None:
    assert FollowUpDecisionParser.parse_strict(text) is None


@pytest.mark.parametrize(
    "text",
    [
        '[{"needs_follow_up": true, "reason": "Call back"}]',
        "Needs follow-up: Yes\nSummary: Call back",
        'Result: {"needs_follow_up": true, "reason": "Call back"}',
    ],
)
def test_parse_strict_rejects_arrays_and_prose(text: str) -> None:
    assert FollowUpDecisionParser.parse_strict(text) is None


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize(
    "parse",
    [
        FollowUpDecisionParser.parse_strict,
        FollowUpDecisionParser.parse_compatibility,
    ],
)
def test_parsers_reject_nonstandard_json_constants(parse, constant: str) -> None:  # noqa: ANN001
    text = (
        '{"needs_follow_up": true, "reason": "Call back", '
        f'"confidence": {constant}}}'
    )

    assert parse(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Needs follow-up: Yes\nSummary: Call the customer",
            FollowUpDecision(needs_follow_up=True, reason="Call the customer"),
        ),
        (
            "Needs follow-up: No\nSummary: Issue resolved",
            FollowUpDecision(needs_follow_up=False, reason="Issue resolved"),
        ),
    ],
)
def test_parse_compatibility_accepts_historical_labeled_text(
    text: str,
    expected: FollowUpDecision,
) -> None:
    assert FollowUpDecisionParser.parse_compatibility(text) == expected


def test_batch_result_parser_keeps_its_tuple_contract() -> None:
    assert parse_follow_up_fields(
        "Needs follow-up: Yes\nSummary: Call the customer"
    ) == ("Yes", "Call the customer")


def test_batch_result_parser_preserves_invalid_raw_text_as_reason() -> None:
    assert parse_follow_up_fields("Unstructured response") == (
        "",
        "Unstructured response",
    )
