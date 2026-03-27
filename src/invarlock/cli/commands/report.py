"""
Report operations group
=======================

Provides the `invarlock report` group with:
  - default callback to generate reports from runs
  - subcommands: verify, explain, html, validate
"""

import math
from time import perf_counter
from typing import Any, NoReturn

import typer
from rich.console import Console

from invarlock.cli.output import print_event, resolve_output_style
from invarlock.core.report_inputs import (
    ReportInputError,
    load_report_input_json,
    resolve_report_input_path,
)
from invarlock.reporting import report_builder as report_builder
from invarlock.reporting.report_contract import (
    ReportGenerationResult,
    generate_reports,
    load_report_payload,
)
from invarlock.reporting.report_telemetry import (
    telemetry_output_enabled,
    telemetry_summary_line,
)

console = Console()

SECTION_WIDTH = 67
KV_LABEL_WIDTH = 16
GATE_LABEL_WIDTH = 32
ARTIFACT_LABEL_WIDTH = 18


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


# Group with callback so `invarlock report` still generates reports
report_app = typer.Typer(
    help="Operations on run reports and evaluation reports (verify, explain, html, validate).",
    invoke_without_command=True,
)


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

        # Show results
        _event("PASS", "Reports generated successfully.", emoji="✅")

        if "report" in result.formats and result.evaluation_report:
            try:
                evaluation_report = result.evaluation_report
                if telemetry_output_enabled():
                    summary_line = telemetry_summary_line(evaluation_report)
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
                    except Exception:
                        summary_suffix = None
                _print_section_header(
                    console,
                    "EVALUATION REPORT SUMMARY",
                    suffix=summary_suffix,
                )
                console.print(_format_kv_line("Status", status_text))

                schema_version = evaluation_report.get("schema_version")
                if schema_version:
                    console.print(
                        _format_kv_line("Schema Version", str(schema_version))
                    )

                run_id = evaluation_report.get("run_id") or (
                    (result.primary_report.get("meta", {}) or {}).get("run_id")
                )
                if run_id:
                    console.print(_format_kv_line("Run ID", str(run_id)))

                model_id = (result.primary_report.get("meta", {}) or {}).get("model_id")
                edit_name = (result.primary_report.get("edit", {}) or {}).get("name")
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

                # In CLI report flow, do not hard-exit on validation failure; just display status.
                # CI gating should be handled by dedicated verify commands.

            except Exception as e:
                _event("WARN", f"Evaluation report validation error: {e}", emoji="⚠️")
                # Exit non-zero on evaluation report generation error
                raise typer.Exit(1) from e
        else:
            console.print(_format_kv_line("Output", str(output_dir)))
            for label, value in _artifact_entries(saved_files, str(output_dir))[1:]:
                console.print(
                    _format_kv_line(label, str(value), width=ARTIFACT_LABEL_WIDTH)
                )

    except ReportInputError as e:
        _raise_report_input_failure(str(e), no_color=no_color)
    except ValueError as e:
        _raise_report_input_failure(str(e), no_color=no_color)
    except typer.Exit:
        raise
    except Exception as e:
        print_event(
            console,
            "FAIL",
            f"Report generation failed: {e}",
            style=resolve_output_style(
                style="audit",
                profile="ci",
                progress=False,
                timing=False,
                no_color=False,
            ),
            emoji="❌",
        )
        raise typer.Exit(1) from e


@report_app.callback(invoke_without_command=True)
def report_callback(
    ctx: typer.Context,
    run: str | None = typer.Option(
        None,
        "--run",
        help=(
            "Path to run report JSON file or directory containing canonical "
            "report.json or evaluation.report.json"
        ),
    ),
    format: str = typer.Option(
        "json", "--format", help="Output format (json|md|html|report|all)"
    ),
    compare: str | None = typer.Option(
        None,
        "--compare",
        help=(
            "Path to comparison report JSON file or directory containing "
            "canonical report.json or evaluation.report.json"
        ),
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help=(
            "Path to baseline report JSON file or directory containing "
            "canonical report.json or evaluation.report.json "
            "(required for report format)"
        ),
    ),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory"),
    style: str = typer.Option("audit", "--style", help="Output style (audit|friendly)"),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colors (respects NO_COLOR=1)"
    ),
):
    """Generate a report from a run (default callback)."""
    if getattr(ctx, "resilient_parsing", False) or ctx.invoked_subcommand is not None:
        return
    if not run:
        print_event(
            console,
            "FAIL",
            "--run is required when no subcommand is provided",
            style=resolve_output_style(
                style=str(style),
                profile="ci",
                progress=False,
                timing=False,
                no_color=no_color,
            ),
            emoji="❌",
        )
        raise typer.Exit(2)
    try:
        result = generate_reports(
            run=run,
            format=format,
            compare=compare,
            baseline=baseline,
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
                "Use: invarlock report --run <subject_report.json> --format report --baseline <baseline_report.json>",
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
    except Exception as exc:
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


def _load_run_report(path: str) -> dict:
    """Load a report from file or from a canonical report directory."""
    return load_report_payload(path)


# Subcommands wired from existing modules
@report_app.command(
    name="verify", help="Recompute and verify metrics for evaluation reports."
)
def report_verify_command(
    reports: list[str] = typer.Argument(
        ...,
        help=(
            "One or more evaluation report JSON files or directories "
            "containing canonical report.json or evaluation.report.json to "
            "verify."
        ),
    ),
    baseline: str | None = typer.Option(
        None,
        "--baseline",
        help=(
            "Optional baseline evaluation report JSON file or directory "
            "containing canonical report.json or evaluation.report.json to "
            "enforce provider parity."
        ),
    ),
    tolerance: float = typer.Option(
        1e-9, "--tolerance", help="Tolerance for analysis-basis comparisons."
    ),
    profile: str | None = typer.Option(
        "dev",
        "--profile",
        help="Execution profile affecting parity enforcement and exit codes (dev|ci|release).",
    ),
    allow_unattested_artifacts: bool = typer.Option(
        False,
        "--allow-unattested-artifacts",
        help="Allow verification of reports without runtime attestation metadata.",
    ),
):  # pragma: no cover - thin wrapper around verify_command
    from pathlib import Path as _Path

    from .verify import verify_command as _verify_command

    try:
        report_paths = [resolve_report_input_path(_Path(p)) for p in reports]
        baseline_path = (
            resolve_report_input_path(_Path(baseline))
            if isinstance(baseline, str)
            else None
        )
    except ReportInputError as exc:
        _raise_report_input_failure(str(exc))
    return _verify_command(
        reports=report_paths,
        baseline=baseline_path,
        tolerance=tolerance,
        profile=profile,
        allow_unattested_artifacts=allow_unattested_artifacts,
    )


@report_app.command(
    name="explain",
    help="Explain gate decisions for subject and baseline run reports.",
)
def report_explain(
    report: str = typer.Option(
        ...,
        "--report",
        help=(
            "Path to primary run report JSON file or directory containing "
            "canonical report.json"
        ),
    ),
    baseline: str = typer.Option(
        ...,
        "--baseline",
        help=(
            "Path to baseline run report JSON file or directory containing "
            "canonical report.json"
        ),
    ),
):  # pragma: no cover - thin wrapper
    """Explain gate decisions for a subject run report vs baseline run report."""
    from .explain_gates import explain_gates_command as _explain

    try:
        report_path, report_payload = load_report_input_json(report)
        baseline_path, baseline_payload = load_report_input_json(baseline)
    except ReportInputError as exc:
        _raise_report_input_failure(str(exc))
    if isinstance(report_payload.get("validation"), dict):
        _raise_report_input_failure(
            "report explain expects a subject run report.json; pass the run "
            "report emitted by invarlock evaluate/run, not an evaluation.report.json bundle."
        )
    if isinstance(baseline_payload.get("validation"), dict):
        _raise_report_input_failure(
            "report explain expects a baseline run report.json; pass the baseline "
            "run report emitted by invarlock evaluate/run, not an evaluation.report.json bundle."
        )
    return _explain(report=str(report_path), baseline=str(baseline_path))


@report_app.command(name="html", help="Render an evaluation report JSON to HTML.")
def report_html(
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help=(
            "Path to evaluation report JSON file or directory containing "
            "canonical report.json or evaluation.report.json"
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
        input_path, _ = load_report_input_json(input)
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
            "report.json or evaluation.report.json to validate against "
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
        _, payload = load_report_input_json(report)
    except ReportInputError as exc:
        _event("FAIL", str(exc), emoji="❌")
        raise typer.Exit(2) from exc

    try:
        from invarlock.reporting.report_builder import validate_report

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
    except Exception as exc:  # noqa: BLE001
        _event("FAIL", f"Validation failed: {exc}", emoji="❌")
        raise typer.Exit(1) from exc


__all__ = ["report_app"]
