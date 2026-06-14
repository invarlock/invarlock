from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_model_evidence_remote_dry_run_emits_tmux_launch_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--gpus",
            "0,1",
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gemma4_e2b_public",
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
    assert payload["remote_repo"] == "/root/invarlock-public"
    assert payload["remote_repo_candidates"] == [
        "/root/invarlock-public",
        "/root/invarlock-public-a100",
    ]
    assert payload["remote_python"] == "auto"
    assert payload["execution_mode"] == "container"
    assert (
        "/root/invarlock-public-a100/.venv/bin/python"
        in payload["remote_python_candidates"]
    )
    assert "/root/venvs/invarlock/bin/python" in payload["remote_python_candidates"]
    assert 'REPO_DIR=""' in payload["sync_command"]
    assert (
        '[ -d "$candidate/.git" ] || [ -f "$candidate/.git" ]'
        in payload["sync_command"]
    )
    assert "for candidate in $REPO_DIR/.venv/bin/python" in payload["sync_command"]
    assert "'$REPO_DIR/.venv/bin/python'" not in payload["sync_command"]
    assert (
        "git fetch origin "
        "refs/heads/staging/next:refs/remotes/origin/staging/next"
        in payload["sync_command"]
    )
    assert "git checkout staging/next" in payload["sync_command"]
    assert "git merge --ff-only origin/staging/next" in payload["sync_command"]
    assert "git pull --ff-only" not in payload["sync_command"]
    assert (
        "$PYTHON_BIN scripts/checks/sync_packaged_contracts.py --check"
        in payload["sync_command"]
    )
    assert len(payload["launches"]) == 2
    assert payload["launches"][0]["session"] == "model-evidence-20260319T120000Z-g0"
    assert (
        "$PYTHON_BIN scripts/model_evidence/model_evidence_sweep.py"
        in payload["launches"][0]["remote_command"]
    )
    assert "cd $REPO_DIR" in payload["launches"][0]["remote_command"]
    assert "--profile" not in payload["launches"][0]["remote_command"]
    assert "--execution-mode container" in payload["launches"][0]["remote_command"]
    assert "CUDA_VISIBLE_DEVICES=0" in payload["launches"][0]["remote_command"]
    assert "--shard-index 0" in payload["launches"][0]["remote_command"]
    assert "--shard-count 2" in payload["launches"][0]["remote_command"]
    assert payload["launches"][1]["session"] == "model-evidence-20260319T120000Z-g1"
    assert "tmux list-sessions" in " ".join(payload["monitor"]["tmux_list"])


def test_run_model_evidence_remote_sync_handles_work_branch_names() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--branch",
            "work/purge-non-apache-qwen35-4b-sanity",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert (
        "git fetch origin "
        "refs/heads/work/purge-non-apache-qwen35-4b-sanity:"
        "refs/remotes/origin/work/purge-non-apache-qwen35-4b-sanity"
        in payload["sync_command"]
    )
    assert "git checkout work/purge-non-apache-qwen35-4b-sanity" in payload[
        "sync_command"
    ]
    assert (
        "git merge --ff-only origin/work/purge-non-apache-qwen35-4b-sanity"
        in payload["sync_command"]
    )
    assert "git pull --ff-only" not in payload["sync_command"]


def test_run_model_evidence_remote_dry_run_forwards_preset_overrides() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--gpus",
            "0",
            "--preset-override",
            "huggingfacetb_smollm3_3b=tmp/smollm3_release.yaml",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["preset_overrides"] == [
        "huggingfacetb_smollm3_3b=tmp/smollm3_release.yaml"
    ]
    remote_command = payload["launches"][0]["remote_command"]
    assert "--preset-override" in remote_command
    assert "huggingfacetb_smollm3_3b=tmp/smollm3_release.yaml" in remote_command


def test_run_model_evidence_remote_dry_run_supports_multi_gpu_groups() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--gpus",
            "0,1,2,3",
            "--gpu-group",
            "0,1",
            "--gpu-group",
            "2,3",
            "--suite",
            "support-matrix-backlog-gpu",
            "--slug",
            "mistralai_mixtral_8x7b_v0_1",
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
    assert payload["gpus"] == ["0", "1", "2", "3"]
    assert payload["gpu_groups"] == [["0", "1"], ["2", "3"]]
    assert len(payload["launches"]) == 2
    first = payload["launches"][0]
    second = payload["launches"][1]
    assert first["gpu"] == "0,1"
    assert first["gpu_group"] == ["0", "1"]
    assert first["session"] == "model-evidence-20260319T120000Z-g0-1"
    assert first["output_root"].endswith("/shard-00-gpu-0-1")
    assert "CUDA_VISIBLE_DEVICES=0,1" in first["remote_command"]
    assert "--shard-count 2" in first["remote_command"]
    assert second["gpu"] == "2,3"
    assert second["session"] == "model-evidence-20260319T120000Z-g2-3"
    assert "CUDA_VISIBLE_DEVICES=2,3" in second["remote_command"]


def test_run_model_evidence_remote_gpu_group_controls_payload_gpus() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--gpu-group",
            "4,5",
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
    assert payload["gpus"] == ["4", "5"]
    assert payload["gpu_groups"] == [["4", "5"]]
    assert payload["launches"][0]["gpu"] == "4,5"


def test_run_model_evidence_remote_dry_run_respects_skip_sync() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

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


def test_run_model_evidence_remote_host_mode_is_forwarded() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--gpus",
            "0",
            "--execution-mode",
            "host",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_mode"] == "host"
    assert "--execution-mode host" in payload["launches"][0]["remote_command"]
    assert "--profile" not in payload["launches"][0]["remote_command"]


def test_run_model_evidence_remote_dry_run_exports_remote_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--gpus",
            "0",
            "--remote-env",
            "HF_HUB_DISABLE_XET=1",
            "--remote-env",
            "HF_HOME=/root/.cache/huggingface",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["remote_env"] == [
        {"name": "HF_HUB_DISABLE_XET", "value": "1"},
        {"name": "HF_HOME", "value": "/root/.cache/huggingface"},
    ]
    remote_command = payload["launches"][0]["remote_command"]
    assert "HF_HUB_DISABLE_XET=1" in remote_command
    assert "HF_HOME=/root/.cache/huggingface" in remote_command
    assert "CUDA_VISIBLE_DEVICES=0" in remote_command


def test_run_model_evidence_remote_rejects_invalid_remote_env_name() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--remote-env",
            "BAD-NAME=1",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 2
    assert "valid shell name" in proc.stderr


def test_run_model_evidence_remote_dry_run_respects_explicit_remote_repo() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "run_model_evidence_remote.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--host",
            "root@example.test",
            "--gpus",
            "0",
            "--remote-repo",
            "/srv/invarlock-custom",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["remote_repo_candidates"] == [
        "/srv/invarlock-custom",
        "/root/invarlock-public-a100",
    ]
    assert (
        "/srv/invarlock-custom/.venv/bin/python" in payload["remote_python_candidates"]
    )
