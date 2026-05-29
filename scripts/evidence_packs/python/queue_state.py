#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, indent=4, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _as_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed >= 0 else default


def write_progress(args: argparse.Namespace) -> int:
    pending = _as_non_negative_int(args.pending)
    ready = _as_non_negative_int(args.ready)
    running = _as_non_negative_int(args.running)
    completed = _as_non_negative_int(args.completed)
    failed = _as_non_negative_int(args.failed)
    total = _as_non_negative_int(args.total)
    progress_pct = int(completed * 100 / total) if total > 0 else 0
    payload = {
        "updated_at": str(args.updated_at),
        "total_tasks": total,
        "pending_tasks": pending,
        "ready_tasks": ready,
        "running_tasks": running,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "progress_pct": progress_pct,
        "status": str(args.status),
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=4, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return 0


def retry_task(args: argparse.Namespace) -> int:
    task_file = Path(args.task_file)
    payload = _read_json(task_file)
    payload["retries"] = _as_non_negative_int(payload.get("retries")) + 1
    payload["status"] = str(args.status)
    payload["assigned_gpus"] = None
    payload["started_at"] = None
    payload["completed_at"] = None
    payload["error_msg"] = None
    _write_json(task_file, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Structured state helpers for evidence-pack queues."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    progress_parser = subparsers.add_parser("progress")
    progress_parser.add_argument("--output", required=True)
    progress_parser.add_argument("--updated-at", required=True)
    progress_parser.add_argument("--pending", required=True)
    progress_parser.add_argument("--ready", required=True)
    progress_parser.add_argument("--running", required=True)
    progress_parser.add_argument("--completed", required=True)
    progress_parser.add_argument("--failed", required=True)
    progress_parser.add_argument("--total", required=True)
    progress_parser.add_argument("--status", required=True)
    progress_parser.set_defaults(func=write_progress)

    retry_parser = subparsers.add_parser("retry-task")
    retry_parser.add_argument("--task-file", required=True)
    retry_parser.add_argument("--status", required=True)
    retry_parser.set_defaults(func=retry_task)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
