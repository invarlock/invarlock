from __future__ import annotations

from typing import Any

from .render_helpers import _fmt_by_kind

_NON_FATAL_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _is_estimated_metric(primary_metric: dict[str, Any]) -> bool:
    try:
        if bool(primary_metric.get("estimated")):
            return True
        return str(primary_metric.get("counts_source", "")).lower() == "pseudo_config"
    except _NON_FATAL_EXCEPTIONS:
        return False


def _format_secondary_metric_ratio(metric: dict[str, Any], kind: str) -> str:
    ratio = metric.get("ratio_vs_baseline")
    try:
        if kind.startswith("ppl"):
            return f"{float(ratio):.3f}"
        return _fmt_by_kind(ratio, kind)
    except _NON_FATAL_EXCEPTIONS:
        return "N/A"


def append_primary_metric_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    primary_metric = evaluation_report.get("primary_metric")
    if not isinstance(primary_metric, dict) or not primary_metric:
        return

    kind = primary_metric.get("kind", "unknown")
    lines.append("## Primary Metric")
    lines.append("")
    unit = primary_metric.get("unit", "-")
    paired = primary_metric.get("paired", False)
    estimated_flag = _is_estimated_metric(primary_metric)
    estimated_suffix = " (estimated)" if estimated_flag else ""

    lines.append(f"- Kind: {kind} (unit: {unit}){estimated_suffix}")
    gating_basis = primary_metric.get("gating_basis") or primary_metric.get("basis")
    if gating_basis:
        lines.append(f"- Basis: {gating_basis}")
    if isinstance(paired, bool):
        lines.append(f"- Paired: {paired}")
    reps = primary_metric.get("reps")
    if isinstance(reps, int | float):
        lines.append(f"- Bootstrap Reps: {int(reps)}")
    ci = primary_metric.get("ci") or primary_metric.get("display_ci")
    if (
        isinstance(ci, list | tuple)
        and len(ci) == 2
        and all(isinstance(value, int | float) for value in ci)
    ):
        lines.append(f"- CI: {ci[0]:.3f}–{ci[1]:.3f}")

    preview = primary_metric.get("preview")
    final = primary_metric.get("final")
    ratio = primary_metric.get("ratio_vs_baseline")

    lines.append("")
    kind_name = str(kind).lower()
    if estimated_flag and kind_name in {"accuracy", "vqa_accuracy"}:
        lines.append(
            "- Note: Accuracy derived from pseudo counts (quick dev preset); use a labeled preset for measured accuracy."
        )
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Preview | {_fmt_by_kind(preview, str(kind))} |")
    lines.append(f"| Final | {_fmt_by_kind(final, str(kind))} |")

    if kind in {"accuracy", "vqa_accuracy"}:
        lines.append(f"| Δ vs Baseline | {_fmt_by_kind(ratio, str(kind))} |")
        try:
            baseline_point = primary_metric.get("baseline_point")
        except _NON_FATAL_EXCEPTIONS:
            baseline_point = None
        if isinstance(baseline_point, int | float) and baseline_point < 0.05:
            lines.append("- Note: baseline < 5%; ratio suppressed; showing Δpp")
    else:
        try:
            lines.append(f"| Ratio vs Baseline | {float(ratio):.3f} |")
        except _NON_FATAL_EXCEPTIONS:
            lines.append("| Ratio vs Baseline | N/A |")
    lines.append("")

    secondary_metrics = evaluation_report.get("secondary_metrics")
    if not isinstance(secondary_metrics, list) or not secondary_metrics:
        return

    lines.append("## Secondary Metrics (informational)")
    lines.append("")
    lines.append("| Kind | Preview | Final | vs Baseline | CI |")
    lines.append("|------|---------|-------|-------------|----|")
    for metric in secondary_metrics:
        if not isinstance(metric, dict):
            continue
        metric_kind = str(metric.get("kind", "?"))
        preview_value = _fmt_by_kind(metric.get("preview"), metric_kind)
        final_value = _fmt_by_kind(metric.get("final"), metric_kind)
        ratio_value = _format_secondary_metric_ratio(metric, metric_kind)
        ci = metric.get("display_ci") or metric.get("ci")
        if isinstance(ci, tuple | list) and len(ci) == 2:
            ci_value = f"{float(ci[0]):.3f}-{float(ci[1]):.3f}"
        else:
            ci_value = "–"
        lines.append(
            f"| {metric_kind} | {preview_value} | {final_value} | {ratio_value} | {ci_value} |"
        )
    lines.append("")
