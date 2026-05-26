#!/usr/bin/env python3
"""Summarize evidence-pack evaluation-loop optimization telemetry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


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
    grouped_entries = sum(
        int(item.get("completed_entries") or 0) for item in group_summaries
    )
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
        "avoided_cli_process_invocations": avoided_processes,
        "timing_totals_seconds": {
            "baseline": _sum_timing(timings, "baseline"),
            "subject": _sum_timing(timings, "subject"),
            "evaluation_report": _sum_timing(timings, "evaluation_report"),
            "total": _sum_timing(timings, "total"),
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
