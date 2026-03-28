from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.reporting.report_make import make_report
from invarlock.reporting.report_telemetry import (
    telemetry_output_enabled,
    telemetry_summary_line,
)


@dataclass(frozen=True)
class RetryReportValidationResult:
    status: str
    passed: bool
    validation: dict[str, Any]
    failed_gates: tuple[str, ...]
    attempt_summary: dict[str, Any]
    evaluation_report: dict[str, Any] | None = None
    telemetry_summary: str | None = None
    error_message: str | None = None


def validate_retry_evaluation_report(
    *,
    report: dict[str, Any],
    baseline_report_data: dict[str, Any] | None,
    baseline_path: Path | None,
    build_retry_result_summary_fn: Any,
    make_report_fn: Any = make_report,
    telemetry_output_enabled_fn: Any = telemetry_output_enabled,
    telemetry_summary_line_fn: Any = telemetry_summary_line,
) -> RetryReportValidationResult:
    try:
        baseline_report = baseline_report_data
        if baseline_report is None and baseline_path is not None:
            with baseline_path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                baseline_report = loaded

        if baseline_report is None:
            raise FileNotFoundError("Baseline report unavailable")

        evaluation_report = make_report_fn(report, baseline_report)
        telemetry_summary = None
        if telemetry_output_enabled_fn():
            telemetry_summary = telemetry_summary_line_fn(evaluation_report)
        validation = (
            evaluation_report.get("validation", {})
            if isinstance(evaluation_report, dict)
            else {}
        )
        if not isinstance(validation, dict):
            validation = {}
        attempt_summary = build_retry_result_summary_fn(validation)
        failed_gates = tuple(attempt_summary.get("failures", []) or [])
        passed = bool(attempt_summary.get("passed"))
        return RetryReportValidationResult(
            status="passed" if passed else "failed",
            passed=passed,
            validation=validation,
            failed_gates=failed_gates,
            attempt_summary=attempt_summary,
            evaluation_report=evaluation_report,
            telemetry_summary=telemetry_summary,
            error_message=None,
        )
    except Exception as exc:
        return RetryReportValidationResult(
            status="error",
            passed=False,
            validation={},
            failed_gates=("report_error",),
            attempt_summary={
                "passed": False,
                "failures": ["report_error"],
                "validation": {},
            },
            evaluation_report=None,
            telemetry_summary=None,
            error_message=str(exc),
        )
