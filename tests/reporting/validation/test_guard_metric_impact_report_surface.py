from __future__ import annotations

import copy
import hashlib
import json

import jsonschema
import pytest

from invarlock.reporting import (
    report_explanation,
    report_metric_impact,
    report_schema,
    report_summary,
)
from invarlock.reporting.rendering import guard_sections


def _diagnostic() -> dict[str, object]:
    return {
        "kind": "guard_metric_impact_info",
        "severity": "info",
        "message": "measured",
        "details": {},
    }


def _evaluated_payload(*, metric_kind: str = "ppl_causal") -> dict[str, object]:
    accuracy = metric_kind == "accuracy"
    facts: dict[str, object] = (
        {"correct": 90, "total": 100, "example_ids_digest": "b" * 64}
        if accuracy
        else {
            "weighted_logloss_sum": 100.0,
            "token_count": 50,
            "example_ids_digest": "b" * 64,
        }
    )
    bare_report = (
        {
            "primary_metric": {"kind": "accuracy", "final": 0.9},
            "final": {
                "correct_total": 90,
                "total": 100,
                "example_ids": list(range(100)),
            },
        }
        if accuracy
        else {
            "primary_metric": {"kind": metric_kind, "final": 7.0},
            "final": {
                "logloss": [2.0],
                (
                    "masked_token_counts"
                    if metric_kind == "ppl_mlm"
                    else "token_counts"
                ): [50],
                "window_ids": [1],
            },
        }
    )
    bare_report["status"] = "success"
    return {
        "metric_kind": metric_kind,
        "direction": "higher" if accuracy else "lower",
        "degradation_basis": "absolute_drop" if accuracy else "relative_increase",
        "bare_value": 0.9 if accuracy else 7.0,
        "guarded_value": 0.88 if accuracy else 7.07,
        "bare_facts": facts,
        "guarded_facts": copy.deepcopy(facts),
        "bare_report": bare_report,
        "degradation": 0.02 if accuracy else 0.01,
        "degradation_limit": 0.03,
        "display_value": 2.0 if accuracy else 1.0,
        "display_unit": "percentage_points" if accuracy else "percent",
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
        "diagnostics": [_diagnostic()],
        "mode": "bare",
        "source": "paired_final_windows",
        "schedule_digest": "a" * 32,
    }


def _impact_validator() -> jsonschema.Draft202012Validator:
    schema = report_schema.REPORT_JSON_SCHEMA["properties"]["guard_metric_impact"]
    return jsonschema.Draft202012Validator(schema)


@pytest.mark.parametrize(
    "metric_kind", ["ppl_causal", "ppl_mlm", "ppl_seq2seq", "accuracy"]
)
def test_guard_metric_impact_schema_accepts_closed_metric_specific_payloads(
    metric_kind: str,
) -> None:
    assert _impact_validator().is_valid(_evaluated_payload(metric_kind=metric_kind))


@pytest.mark.parametrize("status", ["failed", None])
def test_guard_metric_impact_schema_requires_successful_bare_status(
    status: str | None,
) -> None:
    payload = _evaluated_payload()
    bare_report = payload["bare_report"]
    assert isinstance(bare_report, dict)
    if status is None:
        bare_report.pop("status")
    else:
        bare_report["status"] = status

    assert not _impact_validator().is_valid(payload)


@pytest.mark.parametrize("mode", ["guarded", "skipped"])
def test_guard_metric_impact_schema_rejects_noncanonical_mode(
    mode: str,
) -> None:
    payload = _evaluated_payload()
    payload["mode"] = mode

    assert not _impact_validator().is_valid(payload)


def test_guard_metric_impact_schema_accepts_legacy_payload_without_mode() -> None:
    payload = _evaluated_payload()
    payload.pop("mode")

    assert _impact_validator().is_valid(payload)


@pytest.mark.parametrize(
    "legacy_field",
    [
        "bare_ppl",
        "guarded_ppl",
        "impact_ratio",
        "impact_percent",
        "impact_threshold",
        "metric_direction",
        "impact_basis",
        "impact_value",
    ],
)
def test_guard_metric_impact_schema_rejects_legacy_scalar_fields(
    legacy_field: str,
) -> None:
    payload = _evaluated_payload()
    payload[legacy_field] = 1.0
    assert not _impact_validator().is_valid(payload)


@pytest.mark.parametrize(
    "legacy_field",
    [
        "bare_ppl",
        "guarded_ppl",
        "impact_ratio",
        "impact_percent",
        "impact_threshold",
        "metric_direction",
        "impact_basis",
        "impact_value",
    ],
)
def test_guard_metric_impact_assembly_fails_closed_on_removed_fields(
    legacy_field: str,
) -> None:
    payload = _evaluated_payload()
    payload[legacy_field] = 1.0

    sanitized, passed = report_metric_impact.prepare_guard_metric_impact_section(
        payload
    )

    assert passed is False
    assert sanitized["evaluated"] is False
    assert sanitized["passed"] is False
    assert sanitized["diagnostics"][0]["kind"] == "guard_metric_impact_stale_contract"


@pytest.mark.parametrize(
    "field",
    [
        "bare_value",
        "guarded_value",
        "degradation",
        "degradation_limit",
        "display_value",
    ],
)
def test_guard_metric_impact_assembly_rejects_numeric_strings(field: str) -> None:
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ids_digest = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "metric_kind": "accuracy",
        "direction": "higher",
        "degradation_basis": "absolute_drop",
        "bare_value": 0.9,
        "guarded_value": 0.8,
        "bare_facts": {
            "correct": 9,
            "total": 10,
            "example_ids_digest": ids_digest,
        },
        "guarded_facts": {
            "correct": 8,
            "total": 10,
            "example_ids_digest": ids_digest,
        },
        "bare_report": {
            "primary_metric": {"kind": "accuracy", "final": 0.9},
            "final": {"correct_total": 9, "total": 10, "example_ids": ids},
        },
        "degradation": 0.1,
        "degradation_limit": 0.2,
        "display_value": 10.0,
        "display_unit": "percentage_points",
        "evaluated": True,
        "passed": True,
        "checks": {
            "metric_kind_matches": True,
            "measurements_valid": True,
            "guard_metric_impact": True,
            "arm_facts_replay": True,
        },
        "diagnostics": [],
        "source": "unit_test",
        "schedule_digest": "a" * 32,
    }
    payload[field] = str(payload[field])

    sanitized, passed = report_metric_impact.prepare_guard_metric_impact_section(
        payload
    )

    assert passed is False
    assert sanitized["evaluated"] is False


def test_guard_metric_impact_assembly_rejects_non_string_supplied_source() -> None:
    payload = _evaluated_payload()
    payload["source"] = 7

    sanitized, passed = report_metric_impact.prepare_guard_metric_impact_section(
        payload
    )

    assert passed is False
    assert sanitized["evaluated"] is False
    assert sanitized["diagnostics"][0]["kind"] == "guard_metric_impact_invalid_source"


def test_guard_metric_impact_schema_rejects_arbitrary_or_wrong_arm_facts() -> None:
    arbitrary = _evaluated_payload()
    assert isinstance(arbitrary["bare_facts"], dict)
    arbitrary["bare_facts"]["unbound_metric"] = 1
    assert not _impact_validator().is_valid(arbitrary)

    wrong_kind = _evaluated_payload(metric_kind="accuracy")
    wrong_kind["guarded_facts"] = {"weighted_logloss_sum": 1.0, "token_count": 1}
    assert not _impact_validator().is_valid(wrong_kind)

    unpaired_digest = _evaluated_payload()
    assert isinstance(unpaired_digest["guarded_facts"], dict)
    unpaired_digest["guarded_facts"].pop("example_ids_digest")
    assert not _impact_validator().is_valid(unpaired_digest)


def test_guard_metric_impact_schema_keeps_distinct_digest_widths() -> None:
    payload = _evaluated_payload()
    assert _impact_validator().is_valid(payload)
    payload["schedule_digest"] = "a" * 64
    assert not _impact_validator().is_valid(payload)

    payload = _evaluated_payload()
    assert isinstance(payload["bare_facts"], dict)
    payload["bare_facts"]["example_ids_digest"] = "b" * 32
    assert not _impact_validator().is_valid(payload)


def test_guard_metric_impact_schema_accepts_fail_closed_skip_only() -> None:
    skipped = {
        "degradation_limit": 0.01,
        "evaluated": False,
        "passed": False,
        "checks": {},
        "diagnostics": [_diagnostic()],
        "source": "config:context.run.skip_guard_metric_impact_check",
        "skipped": True,
        "skip_reason": "context.run.skip_guard_metric_impact_check",
        "mode": "skipped",
    }
    validator = _impact_validator()
    assert validator.is_valid(skipped)
    skipped["display_value"] = 0.0
    assert not validator.is_valid(skipped)


def test_metric_impact_presentations_use_canonical_display_semantics() -> None:
    payload = _evaluated_payload(metric_kind="accuracy")
    report = {
        "primary_metric": {
            "kind": "accuracy",
            "preview": 0.9,
            "final": 0.88,
            "delta_vs_baseline_pp": -2.0,
        },
        "validation": {"guard_metric_impact_acceptable": True},
        "guard_metric_impact": payload,
    }

    safety = report_summary.build_safety_dashboard_summary(report)
    impact_row = next(row for row in safety.rows if row.label == "Guard Metric Impact")
    assert "+2.00 pp" in impact_row.status
    assert impact_row.summary == "≤ +3.0 pp"

    quality = report_summary.build_quality_gates_summary(report)
    gate = next(
        row for row in quality.rows if row.label == "Guard Metric Impact Acceptable"
    )
    assert gate.measured == "+2.00 pp"
    assert gate.threshold == "≤ +3.0 pp"
    assert "overhead" not in gate.description.lower()

    lines: list[str] = []
    guard_sections._append_guard_metric_impact_observability(
        lines, evaluation_report=report
    )
    rendered = "\n".join(lines)
    assert "Degradation: +2.00 pp" in rendered
    assert "Overhead" not in rendered

    explanation = report_explanation.build_evaluation_report_explanation(report)
    explanation_text = "\n".join(explanation.lines)
    assert "observed: +2.00 pp" in explanation_text
    assert "threshold: ≤ +3.0 pp" in explanation_text
