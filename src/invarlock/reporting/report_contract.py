from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from invarlock.core.report_inputs import load_report_input_json

from .report_bundle import save_evaluation_bundle
from .report_console import compute_console_validation_block
from .report_files import save_report
from .report_make import make_report
from .report_schema import validate_report
from .report_types import RunReport


@dataclass(frozen=True)
class ReportGenerationResult:
    output_dir: str
    formats: list[str]
    saved_files: dict[str, str]
    primary_report: RunReport
    compare_report: RunReport | None
    baseline_report: RunReport | None
    evaluation_report: dict[str, Any] | None
    validation_block: dict[str, Any] | None


def load_report_payload(path: str | Path) -> RunReport:
    _, payload = load_report_input_json(path)
    return cast(RunReport, payload)


def _extract_saved_provenance_env_flags(
    report: RunReport | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    provenance = report.get("provenance")
    if isinstance(provenance, dict):
        provenance_env_flags = provenance.get("env_flags")
        if isinstance(provenance_env_flags, dict) and provenance_env_flags:
            return dict(provenance_env_flags)
    meta = report.get("meta")
    if isinstance(meta, dict):
        meta_env_flags = meta.get("env_flags")
        if isinstance(meta_env_flags, dict) and meta_env_flags:
            return dict(meta_env_flags)
    return None


def _is_non_bool_finite_number(value: Any) -> bool:
    try:
        if isinstance(value, bool):
            return False
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _describe_run_report_health_error(
    report: RunReport | dict[str, Any] | None,
    *,
    role: str,
) -> str | None:
    if not isinstance(report, dict):
        return None

    status = report.get("status")
    if isinstance(status, str):
        normalized_status = status.strip().lower()
        if normalized_status in {"failed", "error"}:
            return (
                f"Cannot generate evaluation report from {role} run report with "
                f"status '{status}'."
            )

    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    primary_metric = metrics.get("primary_metric")
    if not isinstance(primary_metric, dict) or not primary_metric:
        return None
    degraded_reason = primary_metric.get("degraded_reason")
    reason_suffix = (
        f" ({degraded_reason})"
        if isinstance(degraded_reason, str) and degraded_reason.strip()
        else ""
    )
    if bool(primary_metric.get("invalid")) or bool(primary_metric.get("degraded")):
        return (
            f"Cannot generate evaluation report from {role} run report with "
            f"degraded primary metric{reason_suffix}."
        )

    for field_name in ("preview", "final", "ratio_vs_baseline"):
        if field_name not in primary_metric:
            continue
        field_value = primary_metric.get(field_name)
        if field_value is None:
            continue
        if field_name == "ratio_vs_baseline":
            continue
        if not _is_non_bool_finite_number(field_value):
            return (
                f"Cannot generate evaluation report from {role} run report with "
                f"non-finite primary metric field '{field_name}'."
            )

    return None


def _assert_report_can_generate_evaluation(
    report: RunReport | dict[str, Any] | None,
    *,
    role: str,
) -> None:
    health_error = _describe_run_report_health_error(report, role=role)
    if health_error is not None:
        raise ValueError(health_error)


def _assert_evaluation_report_is_finite(
    evaluation_report: dict[str, Any] | None,
) -> None:
    if not isinstance(evaluation_report, dict):
        raise ValueError("Generated evaluation report is missing or malformed.")

    primary_metric = evaluation_report.get("primary_metric")
    if not isinstance(primary_metric, dict) or not primary_metric:
        raise ValueError(
            "Generated evaluation report is missing a primary_metric block."
        )

    degraded_reason = primary_metric.get("degraded_reason")
    reason_suffix = (
        f" ({degraded_reason})"
        if isinstance(degraded_reason, str) and degraded_reason.strip()
        else ""
    )
    if bool(primary_metric.get("invalid")) or bool(primary_metric.get("degraded")):
        raise ValueError(
            "Generated evaluation report contains a degraded primary metric"
            f"{reason_suffix}."
        )

    for field_name in ("preview", "final", "ratio_vs_baseline"):
        if field_name not in primary_metric:
            continue
        field_value = primary_metric.get(field_name)
        if field_value is None:
            continue
        if not _is_non_bool_finite_number(field_value):
            raise ValueError(
                "Generated evaluation report contains non-finite primary metric "
                f"field '{field_name}'."
            )


def generate_reports(
    *,
    run: str,
    format: str = "json",
    compare: str | None = None,
    baseline: str | None = None,
    output: str | None = None,
) -> ReportGenerationResult:
    primary_report = load_report_payload(run)
    compare_report = load_report_payload(compare) if compare else None
    baseline_report = load_report_payload(baseline) if baseline else None

    output_dir = (
        output
        if output is not None
        else (
            f"reports_{Path(run).stem}"
            if Path(run).is_file()
            else f"reports_{Path(run).name}"
        )
    )

    allowed_formats = {"json", "md", "markdown", "html", "report", "all"}
    if format not in allowed_formats:
        raise ValueError(f"Unknown --format '{format}'")

    normalized_format = "markdown" if format == "md" else format
    formats = (
        ["json", "markdown", "html"]
        if normalized_format == "all"
        else [normalized_format]
    )

    if "report" in formats and baseline_report is None:
        raise ValueError("Evaluation report format requires --baseline")

    evaluation_report: dict[str, Any] | None = None
    validation_block: dict[str, Any] | None = None
    if "report" in formats and baseline_report is not None:
        _assert_report_can_generate_evaluation(primary_report, role="subject")
        _assert_report_can_generate_evaluation(baseline_report, role="baseline")
        evaluation_report = make_report(
            primary_report,
            baseline_report,
            provenance_env_flags=_extract_saved_provenance_env_flags(primary_report),
        )
        _assert_evaluation_report_is_finite(evaluation_report)
        validate_report(evaluation_report)
        validation_block = compute_console_validation_block(evaluation_report)

    save_formats = [fmt for fmt in formats if fmt != "report"]
    saved_files: dict[str, Path] = {}
    if save_formats:
        saved_files.update(
            save_report(
                primary_report,
                output_dir,
                formats=save_formats,
                compare=compare_report,
                filename_prefix="evaluation",
            )
        )
    if evaluation_report is not None:
        saved_files.update(
            save_evaluation_bundle(
                run_report=primary_report,
                output_dir=output_dir,
                evaluation_report=evaluation_report,
                source_run_path=run,
            )
        )

    return ReportGenerationResult(
        output_dir=output_dir,
        formats=formats,
        saved_files={key: str(path) for key, path in saved_files.items()},
        primary_report=primary_report,
        compare_report=compare_report,
        baseline_report=baseline_report,
        evaluation_report=evaluation_report,
        validation_block=validation_block,
    )
