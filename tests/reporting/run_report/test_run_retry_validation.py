from __future__ import annotations

import json
from pathlib import Path

from invarlock.reporting.report_builder_support import validate_retry_evaluation_report


def test_validate_retry_evaluation_report_passes_and_emits_telemetry(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"baseline": True}), encoding="utf-8")

    result = validate_retry_evaluation_report(
        report={"subject": True},
        baseline_report_data=None,
        baseline_path=baseline_path,
        build_retry_result_summary_fn=lambda validation: {
            "passed": True,
            "failures": [],
            "validation": validation,
        },
        make_report_fn=lambda report, baseline: {"validation": {"gate": "ok"}},
        telemetry_output_enabled_fn=lambda: True,
        telemetry_summary_line_fn=lambda evaluation_report: "telemetry-summary",
    )

    assert result.status == "passed"
    assert result.passed is True
    assert result.validation_gates == ()
    assert result.telemetry_summary == "telemetry-summary"
    assert result.attempt_summary["passed"] is True


def test_validate_retry_evaluation_report_failure_result() -> None:
    result = validate_retry_evaluation_report(
        report={"subject": True},
        baseline_report_data={"baseline": True},
        baseline_path=None,
        build_retry_result_summary_fn=lambda validation: {
            "passed": False,
            "failures": ["pm_ratio", "window_overlap"],
            "validation": validation,
        },
        make_report_fn=lambda report, baseline: {"validation": {"pm_ratio": False}},
        telemetry_output_enabled_fn=lambda: False,
        telemetry_summary_line_fn=lambda evaluation_report: None,
    )

    assert result.status == "failed"
    assert result.passed is False
    assert result.validation_gates == ("pm_ratio", "window_overlap")
    assert result.validation == {"pm_ratio": False}
    assert result.attempt_summary["failures"] == ["pm_ratio", "window_overlap"]


def test_validate_retry_evaluation_report_error_result() -> None:
    result = validate_retry_evaluation_report(
        report={"subject": True},
        baseline_report_data=None,
        baseline_path=None,
        build_retry_result_summary_fn=lambda validation: {"passed": True},
        make_report_fn=lambda report, baseline: {"validation": {}},
        telemetry_output_enabled_fn=lambda: False,
        telemetry_summary_line_fn=lambda evaluation_report: None,
    )

    assert result.status == "error"
    assert result.passed is False
    assert result.validation_gates == ("report_error",)
    assert result.attempt_summary == {
        "passed": False,
        "failures": ["report_error"],
        "validation": {},
    }
    assert result.diagnostic is not None
    assert "Baseline report unavailable" in result.diagnostic.message
