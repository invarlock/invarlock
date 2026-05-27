#!/usr/bin/env python3
"""Summarize evidence-pack evaluation-loop scheduling telemetry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

_RUN_TIMING_KEYS = (
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
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _sum_timing(payloads: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for payload in payloads:
        timings = payload.get("timings_seconds")
        if not isinstance(timings, dict):
            continue
        value = timings.get(key)
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def _sum_run_timing(payloads: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for payload in payloads:
        aggregate = payload.get("aggregate_run_timings_seconds")
        if isinstance(aggregate, dict):
            try:
                total += float(aggregate.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                pass
            continue
        run_timings = payload.get("run_timings_seconds")
        if not isinstance(run_timings, dict):
            continue
        for side_payload in run_timings.values():
            if not isinstance(side_payload, dict):
                continue
            try:
                total += float(side_payload.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    return total


def _collect_evaluate_timings(run_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("**/evaluate_timing.json")):
        if "/results/analysis/" in str(path):
            continue
        payload = _read_json(path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _collect_group_summaries(run_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("**/evaluation_groups/*/summary.json")):
        payload = _read_json(path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def build_summary(run_dir: Path) -> dict[str, Any]:
    timings = _collect_evaluate_timings(run_dir)
    group_summaries = _collect_group_summaries(run_dir)
    grouped_task_sizes = [
        int(item.get("completed_entries") or 0) for item in group_summaries
    ]
    grouped_entries = sum(grouped_task_sizes)
    avoided_processes = sum(
        int(item.get("avoided_cli_process_invocations") or 0)
        for item in group_summaries
    )
    return {
        "schema": "invarlock/evidence-pack-evaluation-optimization-summary-v1",
        "run_dir": str(run_dir),
        "controls": {
            "PACK_GROUP_EVALUATIONS": os.environ.get("PACK_GROUP_EVALUATIONS")
            or os.environ.get("PACK_EVALUATE_GROUPS")
            or "0",
            "PACK_DEFER_REPORT_RENDERING": os.environ.get("PACK_DEFER_REPORT_RENDERING")
            or os.environ.get("PACK_DEFER_OPTIONAL_REPORT_RENDERING")
            or "0",
        },
        "evaluation_reports_timed": len(timings),
        "baseline_report_reuse_count": sum(
            1 for item in timings if bool(item.get("baseline_report_reused"))
        ),
        "deferred_rendering_count": sum(
            1 for item in timings if bool(item.get("defer_report_rendering"))
        ),
        "grouped_evaluation_tasks": len(group_summaries),
        "grouped_evaluation_entries": grouped_entries,
        "grouped_evaluation_task_sizes": grouped_task_sizes,
        "grouped_evaluation_max_entries_per_task": max(grouped_task_sizes, default=0),
        "avoided_cli_process_invocations": avoided_processes,
        "timing_totals_seconds": {
            "plan": _sum_timing(timings, "plan"),
            "baseline": _sum_timing(timings, "baseline"),
            "subject": _sum_timing(timings, "subject"),
            "evaluation_report": _sum_timing(timings, "evaluation_report"),
            "total": _sum_timing(timings, "total"),
        },
        "run_timing_totals_seconds": {
            key: _sum_run_timing(timings, key) for key in _RUN_TIMING_KEYS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = build_summary(run_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
