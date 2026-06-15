"""
Report operations group
=======================

Provides the `invarlock report` group with explicit subcommands for
generating, explaining, validating, and rendering report artifacts.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, NoReturn, cast

import typer
from rich.console import Console

from invarlock.cli.output import (
    make_command_event_emitter,
    print_command_detail,
    print_event,
    resolve_output_style,
)
from invarlock.core.report_inputs import (
    ReportInputError,
    load_evaluation_report_input_json,
    load_run_report_input_json,
)

from .report_export import register_report_export_command

if TYPE_CHECKING:
    from invarlock.reporting.report_contract import ReportGenerationResult

console = Console()
SECTION_WIDTH = 67
KV_LABEL_WIDTH = 16
GATE_LABEL_WIDTH = 32
ARTIFACT_LABEL_WIDTH = 18
_SUMMARY_FORMAT_ERRORS = (TypeError, ValueError)
_REPORT_RENDER_ERRORS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
)
_REPORT_INNER_ERRORS = _REPORT_RENDER_ERRORS + (ValueError,)
_REPORT_COMMAND_ERRORS = (OSError, RuntimeError, TypeError)
_JSON_INPUT_ERRORS = (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError)
_HTML_RENDER_ERRORS = (AttributeError, ImportError, RuntimeError, TypeError)
_HTML_OUTPUT_ERRORS = (OSError, UnicodeEncodeError)


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


def _section_as_mapping(section: Any) -> dict[str, Any]:
    return dict(section) if isinstance(section, dict) else {}


def _telemetry_output_enabled() -> bool:
    from invarlock.reporting.report_builder_support import telemetry_output_enabled

    return telemetry_output_enabled()


def _telemetry_summary_line(evaluation_report: dict[str, Any]) -> str | None:
    from invarlock.reporting.report_builder_support import telemetry_summary_line

    return telemetry_summary_line(evaluation_report)


def _format_section_title(title: str, *, suffix: str | None = None) -> str:
    if not suffix:
        return title
    combined = f"{title} {suffix}"
    if len(combined) > SECTION_WIDTH:
        return combined
    pad = max(1, SECTION_WIDTH - len(title) - len(suffix))
    return f"{title}{' ' * pad}{suffix}"


def _print_section_header(
    console: Console, title: str, *, suffix: str | None = None
) -> None:
    bar = "═" * SECTION_WIDTH
    console.print(bar)
    console.print(_format_section_title(title, suffix=suffix))
    console.print(bar)


def _format_kv_line(label: str, value: str, *, width: int = KV_LABEL_WIDTH) -> str:
    return f"  {label:<{width}}: {value}"


def _format_status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _fmt_metric_value(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(val):
        return "N/A"
    return f"{val:.3f}"


def _fmt_ci_95(ci: Any) -> str | None:
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        try:
            lo = float(ci[0])
            hi = float(ci[1])
        except (TypeError, ValueError):
            return None
        if math.isfinite(lo) and math.isfinite(hi):
            return f"[{lo:.3f}, {hi:.3f}]"
    return None


def _artifact_entries(
    saved_files: dict[str, str], output_dir: str
) -> list[tuple[str, str]]:
    order = [
        ("report", "Evaluation Report (JSON)"),
        ("report_md", "Evaluation Report (MD)"),
        ("json", "JSON"),
        ("markdown", "Markdown"),
        ("html", "HTML"),
    ]
    entries: list[tuple[str, str]] = [("Output", output_dir)]
    used: set[str] = set()
    for key, label in order:
        if key in saved_files:
            entries.append((label, str(saved_files[key])))
            used.add(key)
    for key in sorted(saved_files.keys()):
        if key in used:
            continue
        entries.append((key.upper(), str(saved_files[key])))
    return entries


def _render_generation_result(
    *,
    result: ReportGenerationResult,
    style: str = "audit",
    no_color: bool = False,
    summary_baseline_seconds: float | None = None,
    summary_subject_seconds: float | None = None,
    summary_report_start: float | None = None,
) -> None:
    try:
        output_style = resolve_output_style(
            style=str(style),
            profile="ci",
            progress=False,
            timing=False,
            no_color=no_color,
        )

        def _event(tag: str, message: str, *, emoji: str | None = None) -> None:
            print_event(console, tag, message, style=output_style, emoji=emoji)

        output_dir = result.output_dir
        saved_files = result.saved_files
        _event("PASS", "Reports generated successfully.", emoji="✅")

        if "report" not in result.formats or not result.evaluation_report:
            console.print(_format_kv_line("Output", str(output_dir)))
            for label, value in _artifact_entries(saved_files, str(output_dir))[1:]:
                console.print(
                    _format_kv_line(label, str(value), width=ARTIFACT_LABEL_WIDTH)
                )
            return

        try:
            evaluation_report = result.evaluation_report
            if _telemetry_output_enabled():
                summary_line = _telemetry_summary_line(evaluation_report)
                if summary_line:
                    console.print(summary_line, markup=False)
            block = result.validation_block or {"overall_pass": False, "rows": []}
            overall_pass = bool(block.get("overall_pass"))
            status_text = _format_status(overall_pass)

            console.print("")
            summary_suffix: str | None = None
            if summary_report_start is not None:
                try:
                    base = (
                        float(summary_baseline_seconds)
                        if summary_baseline_seconds is not None
                        else 0.0
                    )
                    subject = (
                        float(summary_subject_seconds)
                        if summary_subject_seconds is not None
                        else 0.0
                    )
                    report_elapsed = max(
                        0.0, float(perf_counter() - float(summary_report_start))
                    )
                    summary_suffix = f"[{(base + subject + report_elapsed):.2f}s]"
                except _SUMMARY_FORMAT_ERRORS:
                    summary_suffix = None
            _print_section_header(
                console,
                "EVALUATION REPORT SUMMARY",
                suffix=summary_suffix,
            )
            console.print(_format_kv_line("Status", status_text))

            schema_version = evaluation_report.get("schema_version")
            if schema_version:
                console.print(_format_kv_line("Schema Version", str(schema_version)))

            primary_meta = _section_as_mapping(result.primary_report.get("meta", {}))
            primary_edit = _section_as_mapping(result.primary_report.get("edit", {}))
            run_id = evaluation_report.get("run_id") or primary_meta.get("run_id")
            if run_id:
                console.print(_format_kv_line("Run ID", str(run_id)))

            model_id = primary_meta.get("model_id")
            edit_name = primary_edit.get("name")
            if model_id:
                console.print(_format_kv_line("Model", str(model_id)))
            if edit_name:
                console.print(_format_kv_line("Edit", str(edit_name)))

            pm = (
                (evaluation_report.get("primary_metric") or {})
                if isinstance(evaluation_report, dict)
                else {}
            )
            if not pm:
                pm = (result.primary_report.get("metrics", {}) or {}).get(
                    "primary_metric", {}
                )
            console.print("  PRIMARY METRIC")
            pm_entries: list[tuple[str, str]] = []
            if isinstance(pm, dict) and pm:
                kind = str(pm.get("kind") or "primary")
                pm_entries.append(("Kind", kind))
                preview = pm.get("preview")
                if preview is not None:
                    pm_entries.append(("Preview", _fmt_metric_value(preview)))
                final = pm.get("final")
                if final is not None:
                    pm_entries.append(("Final", _fmt_metric_value(final)))
                ratio = pm.get("ratio_vs_baseline")
                if ratio is not None:
                    pm_entries.append(("Ratio", _fmt_metric_value(ratio)))
                dci = pm.get("display_ci")
                ci_95 = _fmt_ci_95(dci)
                if ci_95 is not None:
                    pm_entries.append(("CI (95%)", ci_95))
            if not pm_entries:
                pm_entries.append(("Status", "Unavailable"))
            for idx, (label, value) in enumerate(pm_entries):
                branch = "└─" if idx == len(pm_entries) - 1 else "├─"
                console.print(f"  {branch} {label:<14} {value}")

            console.print("  VALIDATION GATES")
            rows = block.get("rows", [])
            if isinstance(rows, list) and rows:
                for idx, row in enumerate(rows):
                    label = str(row.get("label") or "Unknown")
                    ok = bool(row.get("ok"))
                    status = _format_status(ok)
                    mark = "✓" if ok else "✗"
                    branch = "└─" if idx == len(rows) - 1 else "├─"
                    console.print(
                        f"  {branch} {label:<{GATE_LABEL_WIDTH}} {mark} {status}"
                    )
            else:
                console.print(f"  └─ {'No validation rows':<{GATE_LABEL_WIDTH}} -")

            console.print("  ARTIFACTS")
            entries = _artifact_entries(saved_files, str(output_dir))
            artifact_label_width = max(
                ARTIFACT_LABEL_WIDTH,
                max((len(label) for label, _ in entries), default=0),
            )
            for idx, (label, value) in enumerate(entries):
                branch = "└─" if idx == len(entries) - 1 else "├─"
                console.print(f"  {branch} {label:<{artifact_label_width}} {value}")
            console.print("═" * SECTION_WIDTH)
        except _REPORT_INNER_ERRORS as exc:
            _event("WARN", f"Evaluation report validation error: {exc}", emoji="⚠️")
            raise typer.Exit(1) from exc
    except ReportInputError as exc:
        _raise_report_input_failure(str(exc), no_color=no_color)
    except ValueError as exc:
        _raise_report_input_failure(str(exc), no_color=no_color)
    except typer.Exit:
        raise
    except _REPORT_RENDER_ERRORS as exc:
        print_event(
            console,
            "FAIL",
            f"Report generation failed: {exc}",
            style=resolve_output_style(
                style="audit",
                profile="ci",
                progress=False,
                timing=False,
                no_color=False,
            ),
            emoji="❌",
        )
        raise typer.Exit(1) from exc


# Group with callback so `invarlock report` still generates reports
report_app = typer.Typer(
    help=(
        "Operations on evaluation bundles and run reports "
        "(generate, explain, html, export, validate)."
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
            "canonical evaluation.report.json. Preferred reviewer input; explains "
            "the evaluation bundle directly without requiring linked raw run reports."
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
    from .explain_gates import (
        explain_evaluation_report as _explain_evaluation_report,
    )
    from .explain_gates import (
        explain_gates_command as _explain,
    )

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
            evaluation_path, evaluation_payload = load_evaluation_report_input_json(
                evaluation_report
            )
            print_event(
                console,
                "INFO",
                (f"Explaining evaluation bundle directly: {evaluation_path}"),
                style=output_style,
            )
            return _explain_evaluation_report(evaluation_payload)
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


def _load_html_payload(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        raise ValueError("Expected report JSON object with string keys")
    return cast(dict[str, object], payload)


def _render_html_payload(payload: dict[str, object]) -> str:
    from invarlock.reporting.html import render_report_html

    return render_report_html(payload)


def _write_html_payload(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def export_html_command(
    input: str,
    output: str,
    embed_css: bool = True,
    force: bool = False,
) -> None:
    """Render an evaluation report JSON to HTML."""
    in_path = Path(str(input))
    out_path = Path(str(output))
    emit = make_command_event_emitter(console)

    if out_path.exists() and not force:
        emit("FAIL", "Output file already exists")
        print_command_detail(console, f"Use --force to overwrite: {out_path}")
        raise typer.Exit(1)

    try:
        payload = _load_html_payload(in_path)
    except _JSON_INPUT_ERRORS as exc:
        emit("FAIL", f"Failed to read input JSON: {exc}")
        raise typer.Exit(1) from exc

    try:
        html = _render_html_payload(payload)
    except ValueError as exc:
        emit("FAIL", f"Evaluation report validation failed: {exc}")
        raise typer.Exit(2) from exc
    except _HTML_RENDER_ERRORS as exc:
        emit("FAIL", f"Failed to render HTML: {exc}")
        raise typer.Exit(1) from exc

    if not embed_css:
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

    try:
        _write_html_payload(out_path, html)
    except _HTML_OUTPUT_ERRORS as exc:
        emit("FAIL", f"Failed to write output file: {exc}")
        raise typer.Exit(1) from exc

    emit("PASS", "Exported evaluation report HTML")
    print_command_detail(console, f"Input: {in_path}")
    print_command_detail(console, f"Output: {out_path}")


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
    try:
        input_path, _ = load_evaluation_report_input_json(input)
    except ReportInputError as exc:
        _raise_report_input_failure(str(exc))
    return export_html_command(
        input=str(input_path),
        output=output,
        embed_css=embed_css,
        force=force,
    )


register_report_export_command(report_app)


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
