from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from invarlock.core.auto_tuning import get_tier_policies
from invarlock.reporting.report_builder_support import (
    telemetry_output_enabled,
    telemetry_summary_line,
)
from invarlock.reporting.report_explanation import (
    render_evaluation_report_explanation_lines,
)
from invarlock.reporting.report_make import make_report

console = Console()


def _load_json_payload(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def explain_evaluation_report(
    evaluation_report: dict[str, object],
    *,
    report_payload: object | None = None,
) -> None:
    """Explain gate decisions from an already-built evaluation report."""
    if telemetry_output_enabled():
        summary_line = telemetry_summary_line(evaluation_report)
        if summary_line:
            console.print(summary_line, markup=False)
    for line in render_evaluation_report_explanation_lines(
        evaluation_report,
        report_payload=report_payload,
        tier_policies_getter=get_tier_policies,
    ):
        console.print(line, markup=False)


def explain_gates_command(
    subject_report: str = typer.Option(
        ...,
        "--subject-report",
        help="Path to the subject run report.json",
    ),
    baseline_report: str = typer.Option(
        ...,
        "--baseline-report",
        help="Path to the baseline run report.json",
    ),
) -> None:
    """Explain evaluation report gates for a report vs baseline.

    Loads the reports, builds an evaluation report, and prints gate thresholds,
    observed statistics, and pass/fail reasons in a compact, readable form.
    """
    report_path = Path(subject_report)
    baseline_path = Path(baseline_report)
    if not report_path.exists() or not baseline_path.exists():
        console.print("[red]Missing --subject-report or --baseline-report file[/red]")
        raise typer.Exit(1)

    try:
        report_data = _load_json_payload(report_path)
        baseline_data = _load_json_payload(baseline_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        console.print(f"[red]Failed to load inputs: {exc}[/red]")
        raise typer.Exit(1) from exc

    evaluation_report = make_report(report_data, baseline_data)
    explain_evaluation_report(evaluation_report, report_payload=report_data)
