from __future__ import annotations

import math
from time import perf_counter
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console

from invarlock.cli.output import print_event, resolve_output_style
from invarlock.core.report_inputs import ReportInputError

if TYPE_CHECKING:
    from invarlock.reporting.report_contract import ReportGenerationResult

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


def _section_as_mapping(section: Any) -> dict[str, Any]:
    return dict(section) if isinstance(section, dict) else {}


def _telemetry_output_enabled() -> bool:
    from invarlock.reporting.report_telemetry import telemetry_output_enabled

    return telemetry_output_enabled()


def _telemetry_summary_line(evaluation_report: dict[str, Any]) -> str | None:
    from invarlock.reporting.report_telemetry import telemetry_summary_line

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


def render_generation_result(
    console: Console,
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
    except ReportInputError:
        raise
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


__all__ = ["render_generation_result"]
