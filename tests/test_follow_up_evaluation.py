from __future__ import annotations

import json
import subprocess
import sys

import pytest

from calls_analyser.services.follow_up_evaluation import evaluate_follow_up_rows


def _sample_rows():  # noqa: ANN201
    return [
        {
            "manual_label": True,
            "primary_decision": True,
            "final_decision": True,
            "verification_decision": True,
            "verification_status": "complete",
            "primary_tokens": 100,
            "verification_tokens": 25,
            "primary_cost": 0.10,
            "verification_cost": 0.03,
            "primary_elapsed_seconds": 2.0,
            "verification_elapsed_seconds": 0.5,
        },
        {
            "manual_label": False,
            "primary_decision": True,
            "final_decision": False,
            "verification_decision": False,
            "verification_status": "complete",
            "primary_tokens": 80,
            "verification_tokens": 20,
            "primary_cost": 0.08,
            "verification_cost": 0.02,
            "primary_elapsed_seconds": 1.5,
            "verification_elapsed_seconds": 0.4,
        },
        {
            "manual_label": True,
            "primary_decision": True,
            "final_decision": True,
            "verification_decision": None,
            "verification_status": "failed",
        },
        {
            "manual_label": False,
            "primary_decision": False,
            "final_decision": False,
            "verification_decision": None,
            "verification_status": "not_candidate",
        },
    ]


def test_evaluation_reports_quality_and_operational_metrics() -> None:
    report = evaluate_follow_up_rows(_sample_rows())

    assert report["sample_size"] == 4
    assert report["primary_confusion_matrix"] == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}
    assert report["final_confusion_matrix"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}
    assert report["primary_precision"] == pytest.approx(2 / 3)
    assert report["final_precision"] == 1.0
    assert report["precision_delta"] == pytest.approx(1 / 3)
    assert report["primary_recall"] == 1.0
    assert report["final_recall"] == 1.0
    assert report["recall_delta"] == 0.0
    assert report["changed_to_false_percentage"] == pytest.approx(100 / 3)
    assert report["disagreement_rate"] == 0.5
    assert report["verification_failure_rate"] == pytest.approx(1 / 3)


def test_evaluation_sums_incremental_resource_totals() -> None:
    report = evaluate_follow_up_rows(_sample_rows())

    assert report["resources"] == {
        "primary_tokens": 180.0,
        "verification_tokens": 45.0,
        "incremental_tokens": 45.0,
        "primary_cost": pytest.approx(0.18),
        "verification_cost": pytest.approx(0.05),
        "incremental_cost": pytest.approx(0.05),
        "primary_elapsed_seconds": 3.5,
        "verification_elapsed_seconds": pytest.approx(0.9),
        "incremental_elapsed_seconds": pytest.approx(0.9),
    }


@pytest.mark.parametrize(
    ("precision_delta", "recall_delta", "failure_rate", "eligible"),
    [
        (0.01, -0.02, 0.019, True),
        (0.0, 0.0, 0.0, False),
        (0.01, -0.021, 0.0, False),
        (0.01, 0.0, 0.02, False),
    ],
)
def test_enforcement_thresholds_are_strict(
    precision_delta, recall_delta, failure_rate, eligible,
) -> None:
    rows = _sample_rows()
    report = evaluate_follow_up_rows(rows)
    report["precision_delta"] = precision_delta
    report["recall_delta"] = recall_delta
    report["verification_failure_rate"] = failure_rate

    from calls_analyser.services.follow_up_evaluation import is_eligible_for_enforcement

    assert is_eligible_for_enforcement(report) is eligible


def test_evaluation_rejects_missing_or_invalid_required_values() -> None:
    with pytest.raises(ValueError, match="manual_label"):
        evaluate_follow_up_rows([{"primary_decision": True, "final_decision": True}])
    with pytest.raises(ValueError, match="primary_decision"):
        evaluate_follow_up_rows([
            {"manual_label": "yes", "primary_decision": "maybe", "final_decision": "no"},
        ])


def test_csv_cli_emits_json_report_and_never_changes_tenant_configuration(tmp_path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "manual_label,primary_decision,final_decision,verification_decision,verification_status\n"
        "yes,yes,yes,yes,complete\n"
        "no,yes,no,no,complete\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_follow_up_verification.py", str(csv_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["eligible_for_enforcement"] is True
    assert "tenant configuration decision" in report["enforcement_notice"]
