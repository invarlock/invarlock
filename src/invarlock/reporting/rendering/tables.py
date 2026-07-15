from __future__ import annotations

import math
from typing import Any

from ..report_outline import build_evaluation_report_outline
from ..utils import _fmtv, _p

_TABLE_PARSE_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    OSError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def append_system_overhead_section(lines: list[str], sys_over: dict[str, Any]) -> None:
    """Append the System Overhead markdown section to lines given a payload."""
    if not (isinstance(sys_over, dict) and sys_over):
        return
    lines.append("## System Overhead")
    lines.append("")
    lines.append("| Metric | Baseline | Edited | Δ | Ratio |")
    lines.append("|--------|----------|--------|---|-------|")

    mapping = {
        "latency_ms_p50": "Latency p50 (ms)",
        "latency_ms_p95": "Latency p95 (ms)",
        "throughput_sps": "Throughput (samples/s)",
    }
    for key, label in mapping.items():
        ent = sys_over.get(key)
        if not isinstance(ent, dict):
            continue
        b_raw = ent.get("baseline")
        e_raw = ent.get("edited")
        if isinstance(b_raw, int | float):
            b_val = float(b_raw)
        else:
            b_val = float("nan")
        if isinstance(e_raw, int | float):
            e_val = float(e_raw)
        else:
            e_val = float("nan")
        if (not math.isfinite(b_val) or b_val == 0.0) and (
            not math.isfinite(e_val) or e_val == 0.0
        ):
            b_str = e_str = d_str = r_str = "N/A"
        else:
            b_str = _fmtv(key, b_val)
            e_str = _fmtv(key, e_val)
            d = ent.get("delta")
            r = ent.get("ratio")
            d_str = _fmtv(key, d) if isinstance(d, int | float) else "-"
            r_str = _fmtv(key, r) if isinstance(r, int | float) else "-"
        lines.append(f"| {label} | {b_str} | {e_str} | {d_str} | {r_str} |")
    lines.append("")


def append_accuracy_subgroups(lines: list[str], subgroups: dict[str, Any]) -> None:
    """Append the Accuracy Subgroups markdown table given a subgroups payload."""
    if not (isinstance(subgroups, dict) and subgroups):
        return
    lines.append("## Accuracy Subgroups (informational)")
    lines.append("")
    lines.append("| Group | n(prev) | n(final) | Acc(prev) | Acc(final) | Δpp |")
    lines.append("|-------|---------|----------|-----------|------------|-----|")
    for group, record in subgroups.items():
        try:
            preview_n = int(record.get("n_preview", 0))
        except _TABLE_PARSE_EXCEPTIONS:
            preview_n = 0
        try:
            final_n = int(record.get("n_final", 0))
        except _TABLE_PARSE_EXCEPTIONS:
            final_n = 0
        delta_pp = record.get("delta_pp")
        try:
            delta_text = f"{float(delta_pp):+.1f} pp"
        except _TABLE_PARSE_EXCEPTIONS:
            delta_text = "N/A"
        lines.append(
            f"| {group} | {preview_n} | {final_n} | "
            f"{_p(record.get('preview'))} | {_p(record.get('final'))} | {delta_text} |"
        )
    lines.append("")


def _markdown_table_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def append_outline_fact_summary_section(
    lines: list[str], evaluation_report: dict[str, Any]
) -> None:
    """Append shared report-outline facts to the markdown report."""
    outline = build_evaluation_report_outline(evaluation_report)
    summary_sections = [
        section
        for section in outline.sections
        if section.priority in {"summary", "review", "audit"}
    ]
    if not summary_sections:
        return

    lines.append("## Report Outline")
    lines.append("")
    lines.append(
        "Renderer-neutral summary facts shared by HTML, Markdown, and report explain surfaces."
    )
    lines.append("")
    lines.append("| Section | Fact | Value | Status | Source |")
    lines.append("|---------|------|-------|--------|--------|")
    for section in summary_sections:
        for fact in section.facts:
            lines.append(
                "| "
                f"{_markdown_table_cell(section.title)} | "
                f"{_markdown_table_cell(fact.label)} | "
                f"{_markdown_table_cell(fact.value)} | "
                f"{_markdown_table_cell(fact.status)} | "
                f"`{_markdown_table_cell(fact.source or '-')}` |"
            )
    lines.append("")


__all__ = [
    "append_accuracy_subgroups",
    "append_outline_fact_summary_section",
    "append_system_overhead_section",
]
