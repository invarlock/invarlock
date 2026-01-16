"""
Telemetry report utilities.

Produces a compact JSON summary for performance analysis.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_telemetry_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Build a structured telemetry payload from a run report."""
    meta_in = report.get("meta", {}) if isinstance(report, dict) else {}
    metrics_in = report.get("metrics", {}) if isinstance(report, dict) else {}

    payload: dict[str, Any] = {"generated_at": datetime.now().isoformat()}

    if isinstance(meta_in, dict):
        payload["meta"] = {
            "model_id": meta_in.get("model_id"),
            "adapter": meta_in.get("adapter"),
            "device": meta_in.get("device"),
            "run_id": meta_in.get("run_id"),
            "profile": meta_in.get("profile"),
        }

    if isinstance(metrics_in, dict):
        timings = metrics_in.get("timings")
        if isinstance(timings, dict):
            payload["timings"] = timings

        guard_timings = metrics_in.get("guard_timings")
        if isinstance(guard_timings, dict):
            payload["guard_timings"] = guard_timings

        memory_snapshots = metrics_in.get("memory_snapshots")
        if isinstance(memory_snapshots, list):
            payload["memory_snapshots"] = memory_snapshots

        memory_summary: dict[str, Any] = {}
        for key in (
            "memory_mb_peak",
            "gpu_memory_mb_peak",
            "gpu_memory_reserved_mb_peak",
        ):
            value = metrics_in.get(key)
            if isinstance(value, int | float):
                memory_summary[key] = float(value)
        if memory_summary:
            payload["memory"] = memory_summary

        perf_metrics: dict[str, Any] = {}
        for key in (
            "latency_ms_per_tok",
            "throughput_tok_per_s",
            "eval_samples",
            "total_tokens",
        ):
            value = metrics_in.get(key)
            if isinstance(value, int | float):
                perf_metrics[key] = float(value)
        if perf_metrics:
            payload["performance"] = perf_metrics

    return payload


def save_telemetry_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    filename: str = "telemetry.json",
) -> Path:
    """Write telemetry JSON payload to the output directory."""
    payload = build_telemetry_payload(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = ["build_telemetry_payload", "save_telemetry_report"]
