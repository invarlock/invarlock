#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


def parse_batch(val: str | None, default: int) -> int:
    if not val:
        return default
    val = str(val).strip()
    if val.startswith("auto:"):
        try:
            return int(val.split(":", 1)[1])
        except (TypeError, ValueError):
            return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def estimate_task_memory(_args: argparse.Namespace) -> int:
    profile_path = Path(os.environ["PROFILE_PATH"])
    task_type = os.environ.get("TASK_TYPE", "")
    model_id = (os.environ.get("MODEL_ID") or "").lower()

    try:
        profile = json.loads(profile_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0

    if not isinstance(profile, dict):
        return 0

    if not model_id:
        model_id = str(profile.get("model_id", "")).lower()

    weights_gb = profile.get("weights_gb") or 0.0
    if not weights_gb:
        weights_bytes = profile.get("weights_bytes") or 0
        weights_gb = weights_bytes / (1024**3) if weights_bytes else 0.0

    hidden_size = profile.get("hidden_size")
    num_layers = profile.get("num_layers")
    num_heads = profile.get("num_heads")
    num_kv_heads = profile.get("num_kv_heads") or num_heads
    dtype_bytes = profile.get("dtype_bytes") or 2

    def size_category() -> str:
        if any(t in model_id for t in ("mixtral", "moe", "8x7b")):
            return "moe"
        if weights_gb >= 120:
            return "70"
        if weights_gb >= 80:
            return "40"
        if weights_gb >= 60:
            return "30"
        if weights_gb >= 24:
            return "13"
        return "7"

    category = size_category()

    invarlock_cfg: dict[str, tuple[int, int]] = {
        "7": (512, 96),
        "13": (512, 64),
        "30": (1024, 48),
        "40": (1024, 32),
        "moe": (1024, 8),
        "70": (128, 2),
    }

    default_seq_len, default_batch = invarlock_cfg.get(category, (1024, 32))
    seq_len_invarlock = parse_batch(
        os.environ.get("INVARLOCK_SEQ_LEN"), default_seq_len
    )
    batch_invarlock = parse_batch(os.environ.get("INVARLOCK_EVAL_BATCH"), default_batch)

    def kv_cache_gb(batch: int, seq_len: int) -> float:
        if not all(
            isinstance(x, int) and x > 0 for x in (hidden_size, num_layers, num_heads)
        ):
            return 0.0
        kv_heads = (
            num_kv_heads
            if isinstance(num_kv_heads, int) and num_kv_heads > 0
            else num_heads
        )
        head_dim = hidden_size // num_heads if num_heads else 0
        if head_dim == 0:
            return 0.0
        elems = 2 * num_layers * batch * seq_len * kv_heads * head_dim
        return elems * dtype_bytes / (1024**3)

    load_overhead = float(os.environ.get("MODEL_LOAD_OVERHEAD_GB", "4"))
    edit_overhead = float(os.environ.get("EDIT_OVERHEAD_GB", "8"))
    batch_overhead = float(os.environ.get("BATCH_EDIT_OVERHEAD_GB", "8"))
    inv_overhead = float(os.environ.get("INVARLOCK_OVERHEAD_GB", "6"))
    per_device = int(os.environ.get("GPU_MEMORY_PER_DEVICE", "180"))
    max_gpus = int(os.environ.get("NUM_GPUS", "8"))

    if task_type == "GENERATE_PRESET":
        required = 5.0
    elif task_type == "SETUP_BASELINE":
        required = float(weights_gb) + load_overhead
    elif task_type in ("CLEANUP_EDIT", "CLEANUP_ERROR"):
        required = 1.0
    elif task_type == "CREATE_EDITS_BATCH":
        required = (float(weights_gb) * 2.0) + batch_overhead
    elif task_type in ("CREATE_EDIT", "CREATE_ERROR"):
        required = float(weights_gb) + edit_overhead
    elif task_type in ("CALIBRATION_RUN", "evaluate_EDIT", "evaluate_ERROR"):
        required = (
            float(weights_gb)
            + kv_cache_gb(int(batch_invarlock), int(seq_len_invarlock))
            + inv_overhead
        )
    else:
        required = float(weights_gb) + inv_overhead

    if category == "moe" and task_type in {
        "CALIBRATION_RUN",
        "CREATE_EDIT",
        "CREATE_EDITS_BATCH",
        "CREATE_ERROR",
        "evaluate_EDIT",
        "evaluate_ERROR",
    }:
        required = max(required, float(((max_gpus - 1) * per_device) + 1))

    required_mem = int(math.ceil(required))
    required_gpus = max(1, int(math.ceil(required_mem / per_device)))
    if max_gpus > 0:
        required_gpus = min(required_gpus, max_gpus)

    print(f"{required_mem} {required_gpus}")
    return 0


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

    estimate_parser = subparsers.add_parser("estimate-task-memory")
    estimate_parser.set_defaults(func=estimate_task_memory)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
