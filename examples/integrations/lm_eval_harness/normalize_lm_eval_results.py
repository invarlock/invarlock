#!/usr/bin/env python3
"""Normalize LM Evaluation Harness JSON into a compact sidecar summary."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMAT_VERSION = "invarlock-lm-eval-sidecar-v1"
TASK_METADATA_KEYS = {"alias", "name", "sample_len"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize LM Eval baseline/subject JSON into sidecar summary."
    )
    parser.add_argument("--baseline-json", required=True, type=Path)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--subject-json", type=Path)
    parser.add_argument("--subject-model")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--limit", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lane-label", required=True)
    parser.add_argument("--command-log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"LM Eval result JSON does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"LM Eval result JSON is invalid: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise SystemExit(f"LM Eval result JSON must contain an object: {path}")
    if not isinstance(payload.get("results"), dict):
        raise SystemExit(f"LM Eval result JSON lacks a results object: {path}")
    return payload


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def metric_name(raw_name: str) -> str:
    return raw_name.split(",", maxsplit=1)[0]


def stderr_name(raw_name: str) -> str | None:
    suffix = "_stderr"
    base_name, separator, remainder = raw_name.partition(",")
    if base_name.endswith(suffix):
        metric_key = base_name[: -len(suffix)]
        if separator:
            metric_key = f"{metric_key}{separator}{remainder}"
        return metric_key
    return None


def normalize_results(payload: dict[str, Any], source_json: Path) -> dict[str, Any]:
    versions = payload.get("versions", {})
    sample_counts = payload.get("n-samples", {})
    higher_is_better = payload.get("higher_is_better", {})

    tasks: dict[str, Any] = {}
    for task_name, task_result in sorted(payload["results"].items()):
        if not isinstance(task_result, dict):
            continue

        metrics: dict[str, float] = {}
        metric_aliases: dict[str, str] = {}
        stderrs: dict[str, float | str] = {}
        for raw_key, value in sorted(task_result.items()):
            if raw_key in TASK_METADATA_KEYS:
                continue
            stderr_for = stderr_name(raw_key)
            if stderr_for is not None:
                stderrs[stderr_for] = value
                continue
            if finite_number(value):
                metrics[raw_key] = float(value)
                metric_aliases[raw_key] = metric_name(raw_key)

        tasks[task_name] = {
            "alias": task_result.get("alias"),
            "sample_len": task_result.get("sample_len"),
            "version": versions.get(task_name),
            "samples": sample_counts.get(task_name),
            "higher_is_better": higher_is_better.get(task_name),
            "metrics": metrics,
            "metric_aliases": metric_aliases,
            "stderrs": stderrs,
        }

    config = payload.get("config", {})
    model_config = config if isinstance(config, dict) else {}

    return {
        "source_json": str(source_json),
        "lm_eval_version": payload.get("lm_eval_version"),
        "git_hash": payload.get("git_hash"),
        "model_source": payload.get("model_source"),
        "model_name": payload.get("model_name"),
        "model_sha": model_config.get("model_sha"),
        "device": model_config.get("device"),
        "limit": model_config.get("limit"),
        "batch_size": model_config.get("batch_size"),
        "seeds": {
            "random": model_config.get("random_seed"),
            "numpy": model_config.get("numpy_seed"),
            "torch": model_config.get("torch_seed"),
            "fewshot": model_config.get("fewshot_seed"),
        },
        "tasks": tasks,
    }


def compare_reports(
    baseline: dict[str, Any], subject: dict[str, Any] | None
) -> dict[str, Any] | None:
    if subject is None:
        return None

    comparisons: dict[str, Any] = {}
    baseline_tasks = baseline["tasks"]
    subject_tasks = subject["tasks"]

    for task_name in sorted(set(baseline_tasks) & set(subject_tasks)):
        baseline_metrics = baseline_tasks[task_name]["metrics"]
        subject_metrics = subject_tasks[task_name]["metrics"]
        metric_comparisons: dict[str, Any] = {}

        for name in sorted(set(baseline_metrics) & set(subject_metrics)):
            baseline_value = baseline_metrics[name]
            subject_value = subject_metrics[name]
            entry: dict[str, float] = {
                "baseline": baseline_value,
                "subject": subject_value,
                "subject_minus_baseline": subject_value - baseline_value,
            }
            if baseline_value != 0.0:
                entry["subject_over_baseline"] = subject_value / baseline_value
            metric_comparisons[name] = entry

        if metric_comparisons:
            comparisons[task_name] = metric_comparisons

    return comparisons


def main() -> int:
    args = parse_args()

    baseline_payload = load_report(args.baseline_json)
    subject_payload = load_report(args.subject_json) if args.subject_json else None

    baseline = normalize_results(baseline_payload, args.baseline_json)
    subject = (
        normalize_results(subject_payload, args.subject_json)
        if subject_payload is not None and args.subject_json is not None
        else None
    )

    summary = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "toolchain": "lm-evaluation-harness",
        "tasks_requested": args.tasks,
        "limit_requested": args.limit,
        "device_requested": args.device,
        "lane_artifact_label": args.lane_label,
        "command_log": str(args.command_log) if args.command_log else None,
        "baseline_model_requested": args.baseline_model,
        "subject_model_requested": args.subject_model,
        "baseline": baseline,
        "subject": subject,
        "comparison": compare_reports(baseline, subject),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote sidecar summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
