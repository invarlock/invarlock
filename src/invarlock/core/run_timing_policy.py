from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimingSummaryPayload:
    timings: dict[str, float]
    order: tuple[tuple[str, str], ...]
    extra_lines: tuple[str, ...]


def build_timing_summary_payload(
    *,
    timings: Mapping[str, Any] | None,
    total_duration: float | None,
    report: Mapping[str, Any] | None,
) -> TimingSummaryPayload | None:
    """Normalize timing output into a deterministic summary payload."""
    timings_for_summary: dict[str, float] = {}
    for key, value in dict(timings or {}).items():
        if isinstance(value, int | float):
            timings_for_summary[str(key)] = float(value)
    if total_duration is not None:
        timings_for_summary["total"] = max(0.0, float(total_duration))

    has_breakdown = any(
        key in timings_for_summary
        for key in (
            "prepare",
            "prepare_guards",
            "edit",
            "guards",
            "eval",
            "finalize",
        )
    )

    order: list[tuple[str, str]] = []
    for label, key in (
        ("Load model", "load_model"),
        ("Load data", "load_dataset"),
        ("Prepare", "prepare"),
        ("Prep guards", "prepare_guards"),
        ("Edit", "edit"),
        ("Guards", "guards"),
        ("Eval", "eval"),
        ("Finalize", "finalize"),
        ("Execute", "execute"),
        ("Total", "total"),
    ):
        if key == "execute" and has_breakdown:
            continue
        if key in {"prepare", "prepare_guards", "edit", "guards", "eval", "finalize"}:
            if not has_breakdown:
                continue
        if key in timings_for_summary:
            order.append((label, key))

    extra_lines: list[str] = []
    metrics_section = report.get("metrics", {}) if isinstance(report, Mapping) else {}
    if isinstance(metrics_section, Mapping):
        mem_peak = metrics_section.get("memory_mb_peak")
        gpu_peak = metrics_section.get("gpu_memory_mb_peak")
        if isinstance(mem_peak, int | float):
            extra_lines.append(f"  Peak Memory : {float(mem_peak):.2f} MB")
        if isinstance(gpu_peak, int | float):
            extra_lines.append(f"  Peak GPU Mem: {float(gpu_peak):.2f} MB")

    if not timings_for_summary or not order:
        return None
    return TimingSummaryPayload(
        timings=timings_for_summary,
        order=tuple(order),
        extra_lines=tuple(extra_lines),
    )
