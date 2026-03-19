from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_model_evidence_remote_dry_run_emits_tmux_launch_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--gpus",
            "0,1",
            "--slug",
            "qwen3_8b",
            "--stamp",
            "20260319T120000Z",
            "--remote-output-root",
            "/root/evidence",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["host"] == "root@example.test"
    assert payload["gpus"] == ["0", "1"]
    assert payload["remote_python"] == "auto"
    assert "/root/venvs/invarlock/bin/python" in payload["remote_python_candidates"]
    assert "git checkout staging/next" in payload["sync_command"]
    assert (
        "$PYTHON_BIN scripts/sync_packaged_contracts.py --check"
        in payload["sync_command"]
    )
    assert len(payload["launches"]) == 2
    assert payload["launches"][0]["session"] == "model-evidence-20260319T120000Z-g0"
    assert (
        "$PYTHON_BIN scripts/model_evidence_sweep.py"
        in payload["launches"][0]["remote_command"]
    )
    assert "CUDA_VISIBLE_DEVICES=0" in payload["launches"][0]["remote_command"]
    assert "--shard-index 0" in payload["launches"][0]["remote_command"]
    assert "--shard-count 2" in payload["launches"][0]["remote_command"]
    assert payload["launches"][1]["session"] == "model-evidence-20260319T120000Z-g1"
    assert "tmux list-sessions" in " ".join(payload["monitor"]["tmux_list"])


def test_run_model_evidence_remote_dry_run_respects_skip_sync() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--gpus",
            "2",
            "--skip-sync",
            "--stamp",
            "20260319T120000Z",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["sync_command"] is None
    assert len(payload["launches"]) == 1
    assert payload["launches"][0]["gpu"] == "2"
