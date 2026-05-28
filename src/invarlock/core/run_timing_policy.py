from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimingSummaryPayload:
    timings: dict[str, float]
    ordered_keys: tuple[str, ...]
    memory_mb_peak: float | None
    gpu_memory_mb_peak: float | None


def _coerce_non_bool_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def build_timing_summary_payload(
    *,
    timings: Mapping[str, Any] | None,
    total_duration: float | None,
    report: Mapping[str, Any] | None,
) -> TimingSummaryPayload | None:
    """Normalize timing output into a deterministic summary payload."""
    timings_for_summary: dict[str, float] = {}
    for key, value in dict(timings or {}).items():
        resolved = _coerce_non_bool_float(value)
        if resolved is not None:
            timings_for_summary[str(key)] = resolved
    total_duration_value = _coerce_non_bool_float(total_duration)
    if total_duration_value is not None:
        timings_for_summary["total"] = max(0.0, total_duration_value)

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
        resolved_mem_peak = _coerce_non_bool_float(mem_peak)
        resolved_gpu_peak = _coerce_non_bool_float(gpu_peak)
        if resolved_mem_peak is not None:
            memory_mb_peak = resolved_mem_peak
        if resolved_gpu_peak is not None:
            gpu_memory_mb_peak = resolved_gpu_peak

    if not timings_for_summary or not ordered_keys:
        return None
    return TimingSummaryPayload(
        timings=timings_for_summary,
        ordered_keys=tuple(ordered_keys),
        memory_mb_peak=memory_mb_peak,
        gpu_memory_mb_peak=gpu_memory_mb_peak,
    )
