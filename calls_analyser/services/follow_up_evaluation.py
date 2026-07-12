from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "y"}
_FALSE_VALUES = {"0", "false", "no", "n"}
_FAILURE_STATUSES = {"failed", "failure", "invalid"}


def _boolean(value: Any, field: str, *, optional: bool = False) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if optional:
            return None
        raise ValueError(f"{field} is required")
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{field} must be a boolean value")


def _number(value: Any, field: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def _confusion(labels: list[bool], decisions: list[bool]) -> dict[str, int]:
    return {
        "tp": sum(label and decision for label, decision in zip(labels, decisions, strict=True)),
        "fp": sum(not label and decision for label, decision in zip(labels, decisions, strict=True)),
        "tn": sum(not label and not decision for label, decision in zip(labels, decisions, strict=True)),
        "fn": sum(label and not decision for label, decision in zip(labels, decisions, strict=True)),
    }


def _precision(matrix: Mapping[str, int]) -> float:
    predicted_positive = matrix["tp"] + matrix["fp"]
    return matrix["tp"] / predicted_positive if predicted_positive else 0.0


def _recall(matrix: Mapping[str, int]) -> float:
    actual_positive = matrix["tp"] + matrix["fn"]
    return matrix["tp"] / actual_positive if actual_positive else 0.0


def is_eligible_for_enforcement(report: Mapping[str, Any]) -> bool:
    return (
        report["precision_delta"] > 0
        and report["recall_delta"] >= -0.02
        and report["verification_failure_rate"] < 0.02
    )


def evaluate_follow_up_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("evaluation sample must contain at least one row")

    labels: list[bool] = []
    primary: list[bool] = []
    final: list[bool] = []
    verification: list[bool | None] = []
    statuses: list[str] = []
    resource_fields = (
        "primary_tokens",
        "verification_tokens",
        "primary_cost",
        "verification_cost",
        "primary_elapsed_seconds",
        "verification_elapsed_seconds",
    )
    resources = {field: 0.0 for field in resource_fields}

    for row in materialized:
        labels.append(_boolean(row.get("manual_label"), "manual_label"))
        primary.append(_boolean(row.get("primary_decision"), "primary_decision"))
        final.append(_boolean(row.get("final_decision"), "final_decision"))
        verification.append(
            _boolean(row.get("verification_decision"), "verification_decision", optional=True),
        )
        statuses.append(str(row.get("verification_status", "")).strip().lower())
        for field in resource_fields:
            resources[field] += _number(row.get(field), field)

    primary_matrix = _confusion(labels, primary)
    final_matrix = _confusion(labels, final)
    primary_precision = _precision(primary_matrix)
    final_precision = _precision(final_matrix)
    primary_recall = _recall(primary_matrix)
    final_recall = _recall(final_matrix)
    candidates = [index for index, decision in enumerate(primary) if decision]
    verified = [index for index in candidates if verification[index] is not None]
    changed_to_false = sum(not final[index] for index in candidates)
    disagreements = sum(primary[index] != verification[index] for index in verified)
    failures = sum(
        verification[index] is None or statuses[index] in _FAILURE_STATUSES
        for index in candidates
    )
    candidate_count = len(candidates)

    resources.update({
        "incremental_tokens": resources["verification_tokens"],
        "incremental_cost": resources["verification_cost"],
        "incremental_elapsed_seconds": resources["verification_elapsed_seconds"],
    })
    report: dict[str, Any] = {
        "sample_size": len(materialized),
        "primary_confusion_matrix": primary_matrix,
        "final_confusion_matrix": final_matrix,
        "primary_precision": primary_precision,
        "final_precision": final_precision,
        "precision_delta": final_precision - primary_precision,
        "primary_recall": primary_recall,
        "final_recall": final_recall,
        "recall_delta": final_recall - primary_recall,
        "changed_to_false_percentage": (
            100.0 * changed_to_false / candidate_count if candidate_count else 0.0
        ),
        "disagreement_rate": disagreements / len(verified) if verified else 0.0,
        "verification_failure_rate": failures / candidate_count if candidate_count else 0.0,
        "resources": resources,
    }
    report["eligible_for_enforcement"] = is_eligible_for_enforcement(report)
    report["enforcement_notice"] = (
        "Enforce remains a tenant configuration decision and is never enabled by this tool."
    )
    return report
