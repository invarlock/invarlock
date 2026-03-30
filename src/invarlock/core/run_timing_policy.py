from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimingSummaryPayload:
    timings: dict[str, float]
    ordered_keys: tuple[str, ...]
    memory_mb_peak: float | None
    gpu_memory_mb_peak: float | None


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

    ordered_keys: list[str] = []
    for key in (
        "load_model",
        "load_dataset",
        "prepare",
        "prepare_guards",
        "edit",
        "guards",
        "eval",
        "finalize",
        "execute",
        "total",
    ):
        if key == "execute" and has_breakdown:
            continue
        if key in {"prepare", "prepare_guards", "edit", "guards", "eval", "finalize"}:
            if not has_breakdown:
                continue
        if key in timings_for_summary:
            ordered_keys.append(key)

    memory_mb_peak: float | None = None
    gpu_memory_mb_peak: float | None = None
    metrics_section = report.get("metrics", {}) if isinstance(report, Mapping) else {}
    if isinstance(metrics_section, Mapping):
        mem_peak = metrics_section.get("memory_mb_peak")
        gpu_peak = metrics_section.get("gpu_memory_mb_peak")
        if isinstance(mem_peak, int | float):
            memory_mb_peak = float(mem_peak)
        if isinstance(gpu_peak, int | float):
            gpu_memory_mb_peak = float(gpu_peak)

    if not timings_for_summary or not ordered_keys:
        return None
    return TimingSummaryPayload(
        timings=timings_for_summary,
        ordered_keys=tuple(ordered_keys),
        memory_mb_peak=memory_mb_peak,
        gpu_memory_mb_peak=gpu_memory_mb_peak,
    )
