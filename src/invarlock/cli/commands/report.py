"""
Report operations group
=======================

Provides the `invarlock report` group with explicit subcommands for
generating, explaining, validating, and rendering report artifacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import typer
from rich.console import Console

from invarlock.cli import report_generation_output as report_output_mod
from invarlock.cli.output import print_event, resolve_output_style
from invarlock.core.report_inputs import (
    ReportInputError,
    load_evaluation_report_input_json,
    load_run_report_input_json,
    resolve_run_reports_from_evaluation_input,
)

if TYPE_CHECKING:
    from invarlock.reporting.report_contract import ReportGenerationResult

console = Console()
SECTION_WIDTH = report_output_mod.SECTION_WIDTH
perf_counter = report_output_mod.perf_counter
_ORIG_ARTIFACT_ENTRIES = report_output_mod._artifact_entries
_ORIG_FMT_CI_95 = report_output_mod._fmt_ci_95
_ORIG_FMT_METRIC_VALUE = report_output_mod._fmt_metric_value
_ORIG_FORMAT_SECTION_TITLE = report_output_mod._format_section_title
_ORIG_TELEMETRY_OUTPUT_ENABLED = report_output_mod._telemetry_output_enabled
_ORIG_TELEMETRY_SUMMARY_LINE = report_output_mod._telemetry_summary_line

_REPORT_RENDER_ERRORS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_REPORT_COMMAND_ERRORS = (OSError, RuntimeError, TypeError)


def _generate_reports(**kwargs: Any) -> ReportGenerationResult:
    from invarlock.reporting.report_contract import generate_reports

    return generate_reports(**kwargs)


def _normalize_option(value: Any) -> str | None:
    if isinstance(value, typer.models.OptionInfo) or value is None:
        return None
    text = str(value).strip()
    return text or None


def _raise_report_input_failure(message: str, *, no_color: bool = False) -> NoReturn:
    print_event(
        console,
        "FAIL",
        message,
        style=resolve_output_style(
            style="audit",
            profile="ci",
            progress=False,
            timing=False,
            no_color=no_color,
        ),
        emoji="❌",
    )
    raise typer.Exit(2)


def _format_section_title(title: str, *, suffix: str | None = None) -> str:
    return _ORIG_FORMAT_SECTION_TITLE(title, suffix=suffix)


def _fmt_metric_value(value: Any) -> str:
    return _ORIG_FMT_METRIC_VALUE(value)


def _fmt_ci_95(ci: Any) -> str | None:
    return _ORIG_FMT_CI_95(ci)


def _artifact_entries(
    saved_files: dict[str, str], output_dir: str
) -> list[tuple[str, str]]:
    return _ORIG_ARTIFACT_ENTRIES(saved_files, output_dir)


def _telemetry_output_enabled() -> bool:
    return _ORIG_TELEMETRY_OUTPUT_ENABLED()


def _telemetry_summary_line(evaluation_report: dict[str, Any]) -> str | None:
    return _ORIG_TELEMETRY_SUMMARY_LINE(evaluation_report)


def _render_generation_result(
    *,
    result: ReportGenerationResult,
    style: str = "audit",
    no_color: bool = False,
    summary_baseline_seconds: float | None = None,
    summary_subject_seconds: float | None = None,
    summary_report_start: float | None = None,
) -> None:
    report_output_mod.print_event = print_event
    report_output_mod.perf_counter = perf_counter
    report_output_mod._artifact_entries = _artifact_entries
    report_output_mod._fmt_metric_value = _fmt_metric_value
    report_output_mod._fmt_ci_95 = _fmt_ci_95
    report_output_mod._format_section_title = _format_section_title
    report_output_mod._telemetry_output_enabled = _telemetry_output_enabled
    report_output_mod._telemetry_summary_line = _telemetry_summary_line
    try:
        report_output_mod.render_generation_result(
            console,
            result=result,
            style=style,
            no_color=no_color,
            summary_baseline_seconds=summary_baseline_seconds,
            summary_subject_seconds=summary_subject_seconds,
            summary_report_start=summary_report_start,
        )
    except ReportInputError as exc:
        _raise_report_input_failure(str(exc), no_color=no_color)
    except ValueError as exc:
        _raise_report_input_failure(str(exc), no_color=no_color)


# Group with callback so `invarlock report` still generates reports
report_app = typer.Typer(
    help=(
        "Operations on evaluation bundles and run reports "
        "(generate, explain, html, validate)."
    ),
    invoke_without_command=False,
    no_args_is_help=False,
)


def report_callback(
    ctx: typer.Context | None = None,
    *,
    run: str | None = None,
    format: str = "json",
    compare: str | None = None,
    baseline: str | None = None,
    output: str | None = None,
    style: str = "audit",
    no_color: bool = False,
):
    """Generate reports from explicit run report artifacts."""
    if ctx is not None and (
        getattr(ctx, "resilient_parsing", False) or ctx.invoked_subcommand is not None
    ):
        return
    if not run:
        raise typer.BadParameter("--run is required", param_hint="--run")
    try:
        run_path, _ = load_run_report_input_json(run)
        compare_path = (
            load_run_report_input_json(compare)[0] if isinstance(compare, str) else None
        )
        baseline_path = (
            load_run_report_input_json(baseline)[0]
            if isinstance(baseline, str)
            else None
        )
    except ReportInputError as exc:
        _raise_report_input_failure(str(exc), no_color=no_color)
    try:
        result = _generate_reports(
            run=str(run_path),
            format=format,
            compare=str(compare_path) if compare_path is not None else None,
            baseline=str(baseline_path) if baseline_path is not None else None,
            output=output,
        )
    except ReportInputError as exc:
        _raise_report_input_failure(str(exc), no_color=no_color)
    except ValueError as exc:
        message = str(exc)
        if message == "Evaluation report format requires --baseline":
            print_event(
                console,
                "FAIL",
                message,
                style=resolve_output_style(
                    style=str(style),
                    profile="ci",
                    progress=False,
                    timing=False,
                    no_color=no_color,
                ),
                emoji="❌",
            )
            print_event(
                console,
                "INFO",
                "Use: invarlock report generate --run <subject_report.json> --format report --baseline-run-report <baseline_report.json>",
                style=resolve_output_style(
                    style=str(style),
                    profile="ci",
                    progress=False,
                    timing=False,
                    no_color=no_color,
                ),
            )
            raise typer.Exit(1) from exc
        _raise_report_input_failure(message, no_color=no_color)
    except _REPORT_COMMAND_ERRORS as exc:
        print_event(
            console,
            "FAIL",
            f"Report generation failed: {exc}",
            style=resolve_output_style(
                style=str(style),
                profile="ci",
                progress=False,
                timing=False,
                no_color=no_color,
            ),
            emoji="❌",
        )
        raise typer.Exit(1) from exc
    _render_generation_result(
        result=result,
        style=style,
        no_color=no_color,
    )
    return


@report_app.callback(invoke_without_command=True)
def report_root(ctx: typer.Context) -> None:
    """Report command namespace."""
    if getattr(ctx, "resilient_parsing", False) or ctx.invoked_subcommand is not None:
        return
    typer.echo(ctx.get_help())
    raise typer.Exit(0)


@report_app.command(
    name="generate",
    help="Generate reports from existing run report artifacts.",
)
def report_generate_command(
    run: str = typer.Option(
        ...,
        "--run",
        help=(
            "Path to subject run report JSON file or directory containing canonical "
            "report.json"
        ),
    ),
    format: str = typer.Option(
        "json", "--format", help="Output format (json|md|html|report|all)"
    ),
    compare: str | None = typer.Option(
        None,
        "--compare-run-report",
        help=(
            "Optional comparison run report JSON file or directory containing canonical "
            "report.json"
        ),
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline-run-report",
        help=(
            "Optional baseline run report JSON file or directory containing canonical "
            "report.json (required for report format)"
        ),
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
    style: str = typer.Option("audit", "--style", help="Output style (audit|friendly)"),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
) -> None:
    report_callback(
        run=run,
        format=format,
        compare=compare,
        baseline=baseline,
        output=output,
        style=style,
        no_color=no_color,
    )


def _load_run_report(path: str) -> dict:
    """Load a report from file or from a canonical report directory."""
    return load_run_report_input_json(path)[1]


@report_app.command(
    name="explain",
    help=(
        "Explain gate decisions from an evaluation bundle or from explicit "
        "subject/baseline run reports."
    ),
)
def report_explain(
    evaluation_report: str | None = typer.Option(
        None,
        "--evaluation-report",
        help=(
            "Path to evaluation report JSON file or directory containing "
            "canonical evaluation.report.json. Preferred reviewer input; "
            "auto-resolves linked subject and baseline run reports from provenance."
        ),
    ),
    subject_report: str | None = typer.Option(
        None,
        "--subject-report",
        help=(
            "Path to subject run report JSON file or directory containing "
            "canonical report.json. Use with --baseline-report when you want to "
            "explain directly from run artifacts."
        ),
    ),
    baseline_report: str | None = typer.Option(
        None,
        "--baseline-report",
        help=(
            "Path to baseline run report JSON file or directory containing "
            "canonical report.json. Optional when --evaluation-report is supplied."
        ),
    ),
):  # pragma: no cover - thin wrapper
    """Explain gate decisions for evaluation bundles or explicit run reports."""
    from .explain_gates import explain_gates_command as _explain

    output_style = resolve_output_style(
        style="audit",
        profile="ci",
        progress=False,
        timing=False,
        no_color=False,
    )
    evaluation_report = _normalize_option(evaluation_report)
    subject_report = _normalize_option(subject_report)
    baseline_report = _normalize_option(baseline_report)

    try:
        if evaluation_report:
            if subject_report or baseline_report:
                raise typer.BadParameter(
                    "Use either --evaluation-report or the --subject-report/--baseline-report pair, not both."
                )
            (
                evaluation_path,
                report_path,
                baseline_path,
            ) = resolve_run_reports_from_evaluation_input(evaluation_report)
            print_event(
                console,
                "INFO",
                (
                    "Resolved linked run reports from evaluation bundle provenance: "
                    f"{evaluation_path}"
                ),
                style=output_style,
            )
        else:
            if not subject_report or not baseline_report:
                raise typer.BadParameter(
                    "Pass --evaluation-report or both --subject-report and --baseline-report."
                )
            report_path, _report_payload = load_run_report_input_json(subject_report)
            baseline_path, _baseline_payload = load_run_report_input_json(
                baseline_report
            )
    except ReportInputError as exc:
        detail = str(exc)
        if exc.reason == "expected_run_payload":
            detail += (
                " Use --evaluation-report <evaluation.report.json> to "
                "auto-resolve the linked run reports."
            )
        _raise_report_input_failure(detail)
    except typer.BadParameter as exc:
        _raise_report_input_failure(str(exc))
    return _explain(
        subject_report=str(report_path),
        baseline_report=str(baseline_path),
    )


@report_app.command(name="html", help="Render an evaluation report JSON to HTML.")
def report_html(
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help=(
            "Path to evaluation report JSON file or directory containing "
            "canonical evaluation.report.json"
        ),
    ),
    output: str = typer.Option(..., "--output", "-o", help="Path to output HTML file"),
    embed_css: bool = typer.Option(
        True, "--embed-css/--no-embed-css", help="Inline a minimal static stylesheet"
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite output file if exists"
    ),
):  # pragma: no cover - thin wrapper
    from .export_html import export_html_command as _export

    try:
        input_path, _ = load_evaluation_report_input_json(input)
    except ReportInputError as exc:
        _raise_report_input_failure(str(exc))
    return _export(
        input=str(input_path),
        output=output,
        embed_css=embed_css,
        force=force,
    )


@report_app.command("validate")
def report_validate(
    report: str = typer.Argument(
        ...,
        help=(
            "Path to report JSON file or directory containing canonical "
            "evaluation.report.json to validate against "
            "schema v1"
        ),
    ),
):
    """Validate an evaluation report JSON against the current schema (v1)."""
    output_style = resolve_output_style(
        style="audit",
        profile="ci",
        progress=False,
        timing=False,
        no_color=False,
    )

    def _event(tag: str, message: str, *, emoji: str | None = None) -> None:
        print_event(console, tag, message, style=output_style, emoji=emoji)

    try:
        _, payload = load_evaluation_report_input_json(report)
    except ReportInputError as exc:
        _event("FAIL", str(exc), emoji="❌")
        raise typer.Exit(2) from exc

    try:
        from invarlock.reporting.report_schema import validate_report

        ok = validate_report(payload)
        if not ok:
            _event("FAIL", "Evaluation report schema validation failed", emoji="❌")
            raise typer.Exit(2)
        _event("PASS", "Evaluation report schema is valid", emoji="✅")
    except ValueError as exc:
        _event("FAIL", f"Evaluation report validation error: {exc}", emoji="❌")
        raise typer.Exit(2) from exc
    except typer.Exit:
        raise
    except _REPORT_RENDER_ERRORS as exc:
        _event("FAIL", f"Validation failed: {exc}", emoji="❌")
        raise typer.Exit(1) from exc


__all__ = ["report_app"]
