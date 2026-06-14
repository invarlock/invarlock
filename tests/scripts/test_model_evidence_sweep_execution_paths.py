from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.scripts._support_model_evidence_sweep import (
    load_script_module,
    write_fake_python,
    write_flaky_fake_python,
)


def test_model_evidence_sweep_host_mode_emits_explicit_runtime_flags(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "evidence-host"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gemma4_e2b_public",
            "--execution-mode",
            "host",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload) == 1
    assert payload[0]["execution_mode"] == "host"
    assert payload[0]["prefetch"][-1] == "google/gemma-4-E2B-it"
    assert payload[0]["prefetch"][1] == "-c"
    assert "--execution-mode" in payload[0]["evaluate"]
    assert (
        payload[0]["evaluate"][payload[0]["evaluate"].index("--execution-mode") + 1]
        == "host"
    )
    assert payload[0]["evaluate"][payload[0]["evaluate"].index("--assurance") + 1] == (
        "off"
    )
    assert "--allow-host-execution" not in payload[0]["evaluate"]
    assert "--runtime-provenance" in payload[0]["verify"]
    assert (
        payload[0]["verify"][payload[0]["verify"].index("--runtime-provenance") + 1]
        == "host"
    )
    assert payload[0]["verify"][payload[0]["verify"].index("--profile") + 1] == "dev"
    preset_idx = payload[0]["evaluate"].index("--preset") + 1
    assert payload[0]["evaluate"][preset_idx] == str(
        repo_root / "configs/presets/causal_lm/gemma4_e2b_512.yaml"
    )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "host"


def test_model_evidence_sweep_host_mode_prefetches_before_evaluate(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "fake-python"
    write_fake_python(fake_python)
    output_root = tmp_path / "evidence-host-prefetch"
    log_path = tmp_path / "fake-python.log"

    env = dict(os.environ)
    env["FAKE_PYTHON_LOG"] = str(log_path)

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gemma4_e2b_public",
            "--execution-mode",
            "host",
            "--output-root",
            str(output_root),
            "--python",
            str(fake_python),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env=env,
    )

    assert proc.returncode == 1, proc.stderr
    invocations = log_path.read_text(encoding="utf-8").splitlines()
    assert "google/gemma-4-E2B-it" in invocations[0]
    assert "-m invarlock evaluate" in invocations[1]


def test_model_evidence_sweep_retries_evaluate_once_after_sigterm(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "flaky-fake-python"
    write_flaky_fake_python(fake_python)
    output_root = tmp_path / "evidence-host-retry"
    log_path = tmp_path / "fake-python-retry.log"
    state_path = tmp_path / "retry-state"

    env = dict(os.environ)
    env["FAKE_PYTHON_LOG"] = str(log_path)
    env["FAKE_PYTHON_STATE"] = str(state_path)

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gemma4_e2b_public",
            "--execution-mode",
            "host",
            "--output-root",
            str(output_root),
            "--python",
            str(fake_python),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    invocations = log_path.read_text(encoding="utf-8").splitlines()
    evaluate_invocations = [
        line for line in invocations if "-m invarlock evaluate" in line
    ]
    verify_invocations = [line for line in invocations if "-m invarlock verify" in line]
    assert len(evaluate_invocations) == 2
    assert len(verify_invocations) == 1
    lane_log = (output_root / "logs" / "gemma4_e2b_public.log").read_text(
        encoding="utf-8"
    )
    assert "evaluate exited with -15; retrying once." in lane_log


def test_model_evidence_sweep_marks_gated_prefetch_as_skipped(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "gated-fake-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-c" ]]; then
  echo "GatedRepoError: Cannot access gated repo" >&2
  exit 1
fi
echo "unexpected invocation" >&2
exit 99
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    output_root = tmp_path / "evidence-host-gated"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "model-catalog-gpu",
            "--slug",
            "google_gemma_3_4b_it",
            "--execution-mode",
            "host",
            "--output-root",
            str(output_root),
            "--python",
            str(fake_python),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 0, proc.stderr
    summary_tsv = (output_root / "summary.tsv").read_text(encoding="utf-8")
    assert "status\tdetail" in summary_tsv
    assert "skipped\tgated_repo" in summary_tsv
    status_log = (output_root / "status.log").read_text(encoding="utf-8")
    assert "status=skipped detail=gated_repo" in status_log


def test_build_evaluate_command_uses_container_safe_repo_relative_paths(
    tmp_path: Path,
) -> None:
    mod = load_script_module("model_evidence_sweep")
    spec = next(
        lane
        for lane in mod.CURRENT_PUBLISHED_BASIS_LANES
        if lane.slug == "gemma4_e2b_public"
    )
    external_output_root = tmp_path / "external-container-evidence"
    execution_root = mod._execution_root(
        external_output_root, execution_mode="container"
    )
    lane_root = execution_root / "eval" / spec.slug

    command = mod.build_evaluate_command(
        spec,
        python_exe=sys.executable,
        profile="ci",
        device="cuda",
        execution_mode="container",
        lane_root=lane_root,
    )

    out_idx = command.index("--out") + 1
    report_idx = command.index("--report-out") + 1
    base = f"tmp/model_evidence_container/{execution_root.name}"
    assert command[out_idx] == f"{base}/eval/gemma4_e2b_public/runs"
    assert command[report_idx] == f"{base}/eval/gemma4_e2b_public/report"


def test_runtime_env_preserves_container_default_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = load_script_module("model_evidence_sweep")
    config_root = tmp_path / "config-root"
    config_root.mkdir()
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("INVARLOCK_ALLOW_NETWORK", raising=False)
    monkeypatch.setenv("INVARLOCK_CONFIG_ROOT", str(config_root))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("INVARLOCK_EXPORT_DIR", str(export_dir))
    monkeypatch.setenv("INVARLOCK_STORE_EVAL_WINDOWS", "1")
    monkeypatch.setenv("INVARLOCK_SNAPSHOT_MODE", "auto")

    env = mod.runtime_env()

    assert env["PYTHONPATH"] == str(mod.REPO_ROOT / "src")
    assert env["INVARLOCK_ALLOW_NETWORK"] == "1"
    assert env["INVARLOCK_CONFIG_ROOT"] == str(config_root)
    assert env["HF_HOME"] == str(hf_home)
    assert env["TMPDIR"] == str(tmpdir)
    assert env["INVARLOCK_EXPORT_DIR"] == str(export_dir)
    assert env["INVARLOCK_STORE_EVAL_WINDOWS"] == "1"
    assert env["INVARLOCK_SNAPSHOT_MODE"] == "auto"


def test_runtime_env_defaults_hf_cache_to_repo_visible_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = load_script_module("model_evidence_sweep")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(mod, "REPO_ROOT", repo_root)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)

    env = mod.runtime_env()

    hf_home = repo_root / "tmp" / "model_evidence_hf_home"
    assert env["HF_HOME"] == str(hf_home)
    assert env["HF_HUB_CACHE"] == str(hf_home / "hub")
    assert env["HF_DATASETS_CACHE"] == str(hf_home / "datasets")
    assert hf_home.is_dir()
    assert (hf_home / "hub").is_dir()
    assert (hf_home / "datasets").is_dir()


def test_model_evidence_sweep_container_mode_publishes_external_output_root(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "fake-python"
    write_fake_python(fake_python)
    output_root = tmp_path / "external-container-evidence"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gemma4_e2b_public",
            "--execution-mode",
            "container",
            "--output-root",
            str(output_root),
            "--python",
            str(fake_python),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )

    assert proc.returncode == 1, proc.stderr
    published_lane = output_root / "eval" / "gemma4_e2b_public"
    assert (published_lane / "report" / "evaluation.report.json").is_file()
    assert (published_lane / "verify.json").is_file()
    scratch_hash = hashlib.sha256(
        output_root.resolve().as_posix().encode("utf-8")
    ).hexdigest()[:16]
    scratch_root = repo_root / "tmp" / "model_evidence_container" / scratch_hash
    assert not scratch_root.exists()
