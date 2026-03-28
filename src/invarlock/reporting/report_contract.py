from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarlock.core.report_inputs import load_report_input_json

from .report_bundle import save_evaluation_bundle
from .report_console import compute_console_validation_block
from .report_files import save_report
from .report_make import make_report
from .report_schema import validate_report


@dataclass(frozen=True)
class ReportGenerationResult:
    output_dir: str
    formats: list[str]
    saved_files: dict[str, str]
    primary_report: dict[str, Any]
    compare_report: dict[str, Any] | None
    baseline_report: dict[str, Any] | None
    evaluation_report: dict[str, Any] | None
    validation_block: dict[str, Any] | None


def load_report_payload(path: str | Path) -> dict[str, Any]:
    _, payload = load_report_input_json(path)
    return payload


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
        evaluation_report = make_report(primary_report, baseline_report)
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
