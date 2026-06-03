from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "evidence_packs" / "python" / "queue_state.py"


def _run(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_queue_state_retry_task_updates_structured_fields(tmp_path: Path) -> None:
    task_file = tmp_path / "failed.task"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "status": "failed",
                "retries": None,
                "assigned_gpus": "0",
                "started_at": "x",
                "completed_at": "y",
                "error_msg": "boom",
            }
        ),
        encoding="utf-8",
    )

    result = _run("retry-task", "--task-file", str(task_file), "--status", "ready")

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(task_file.read_text(encoding="utf-8"))
    assert payload["retries"] == 1
    assert payload["status"] == "ready"
    assert payload["assigned_gpus"] is None
    assert payload["started_at"] is None
    assert payload["completed_at"] is None
    assert payload["error_msg"] is None


def test_queue_state_progress_writes_canonical_summary(tmp_path: Path) -> None:
    output = tmp_path / "progress.json"

    result = _run(
        "progress",
        "--output",
        str(output),
        "--updated-at",
        "2026-01-01T00:00:00Z",
        "--pending",
        "1",
        "--ready",
        "1",
        "--running",
        "0",
        "--completed",
        "2",
        "--failed",
        "0",
        "--total",
        "4",
        "--status",
        "running",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "updated_at": "2026-01-01T00:00:00Z",
        "total_tasks": 4,
        "pending_tasks": 1,
        "ready_tasks": 1,
        "running_tasks": 0,
        "completed_tasks": 2,
        "failed_tasks": 0,
        "progress_pct": 50,
        "status": "running",
    }


def test_queue_state_estimate_task_memory_preserves_stdout_contract(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "model_id": "allenai/OLMo-2-1124-7B",
                "weights_gb": 14,
                "hidden_size": 4096,
                "num_layers": 32,
                "num_heads": 32,
                "num_kv_heads": 32,
                "dtype_bytes": 2,
            }
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "TASK_TYPE": "CALIBRATION_RUN",
        "MODEL_ID": "allenai/OLMo-2-1124-7B",
        "PROFILE_PATH": str(profile),
        "GPU_MEMORY_PER_DEVICE": "80",
        "NUM_GPUS": "1",
    }

    result = _run("estimate-task-memory", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "44 1"
