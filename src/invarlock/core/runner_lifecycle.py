from __future__ import annotations

import time
from typing import Any

from .api import RunConfig, RunReport


def initialize_run_report(
    *,
    config: RunConfig,
    serialized_config: dict[str, Any],
    cuda_flags: dict[str, Any],
    auto_config: dict[str, Any] | None = None,
    report_factory: type[RunReport] = RunReport,
    start_time: float | None = None,
) -> RunReport:
    report = report_factory()
    context = config.context
    report.meta["cuda_flags"] = cuda_flags
    report.meta["start_time"] = (
        float(start_time) if start_time is not None else float(time.time())
    )
    report.meta["config"] = serialized_config

    if context:
        normalized_context = dict(context)
        try:
            report.context.update(normalized_context)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            report.context = normalized_context

    run_id = context.get("run_id") if context is not None else None
    if run_id:
        report.meta["run_id"] = run_id
    plugins_meta = context.get("plugins") if context is not None else None
    if plugins_meta:
        report.meta["plugins"] = plugins_meta

    if auto_config:
        report.meta["auto"] = auto_config
        existing_auto = context.get("auto") if context is not None else None
        if isinstance(context, dict) and isinstance(existing_auto, dict):
            merged_auto = dict(existing_auto)
            merged_auto.update(auto_config)
            context["auto"] = merged_auto
            report.context["auto"] = context["auto"]
        elif isinstance(context, dict):
            context["auto"] = dict(auto_config)
            report.context["auto"] = context["auto"]

    return report


def finalize_run_report(
    report: RunReport,
    *,
    final_status: str,
    end_time: float | None = None,
) -> None:
    end_ts = float(end_time) if end_time is not None else float(time.time())
    report.status = final_status
    report.meta["end_time"] = end_ts
    start_time = report.meta.get("start_time")
    if isinstance(start_time, int | float):
        report.meta["duration"] = end_ts - float(start_time)


def merge_execution_metrics(
    report: RunReport,
    *,
    timings: dict[str, float],
    guard_timings: dict[str, float],
    memory_snapshots: list[dict[str, Any]],
    memory_summary: dict[str, Any],
) -> None:
    metrics_obj: object = report.metrics
    if isinstance(metrics_obj, dict):
        metrics = metrics_obj
    else:
        report.metrics = {}
        metrics = report.metrics

    if timings:
        metrics.setdefault("timings", {}).update(timings)

    if guard_timings:
        metrics["guard_timings"] = guard_timings

    if not memory_snapshots:
        return

    metrics["memory_snapshots"] = memory_snapshots
    summary = dict(memory_summary)
    mem_peak = summary.get("memory_mb_peak")
    if isinstance(mem_peak, int | float):
        existing_peak = metrics.get("memory_mb_peak")
        if isinstance(existing_peak, int | float):
            summary["memory_mb_peak"] = max(float(existing_peak), float(mem_peak))
    metrics.update(summary)


__all__ = [
    "finalize_run_report",
    "initialize_run_report",
    "merge_execution_metrics",
]
