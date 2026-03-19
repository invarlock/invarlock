from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path


def _load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_python(path: Path) -> None:
    script = """#!/bin/bash
set -euo pipefail

if [[ "${1:-}" == "-m" && "${2:-}" == "invarlock" && "${3:-}" == "evaluate" ]]; then
  report_dir=""
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--report-out" ]]; then
      report_dir="$2"
      shift 2
      continue
    fi
    shift
  done
  mkdir -p "$report_dir"
  printf '{"ok": true}\\n' > "$report_dir/evaluation.report.json"
  exit 0
fi

if [[ "${1:-}" == "-m" && "${2:-}" == "invarlock" && "${3:-}" == "verify" ]]; then
  printf '{"status":"fail"}\\n'
  exit 1
fi

printf 'unexpected invocation: %s\\n' "$*" >&2
exit 99
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_manifest_lane_ids_match_supported_experimental_support_matrix() -> None:
    mod = _load_script_module("model_evidence_sweep")

    expected = mod.supported_experimental_lane_ids()
    actual = mod.manifest_lane_ids(mod.CURRENT_SUPPORTED_EXPERIMENTAL_LANES)

    assert actual == expected
    for lane in mod.CURRENT_SUPPORTED_EXPERIMENTAL_LANES:
        assert lane.preset_path.is_file(), lane.preset_relpath


def test_select_specs_sharding_is_stable() -> None:
    mod = _load_script_module("model_evidence_sweep")

    shard = mod.select_specs(
        mod.DEFAULT_SUITE,
        slugs=[],
        lane_ids=[],
        shard_index=1,
        shard_count=3,
    )

    assert [lane.slug for lane in shard] == [
        "qwen2_7b",
        "deepseek_r1_distill_qwen_7b",
        "olmo2_7b",
    ]


def test_model_evidence_sweep_dry_run_emits_commands_and_manifest(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence_sweep.py"
    output_root = tmp_path / "evidence"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--slug",
            "qwen3_8b",
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
    assert payload[0]["slug"] == "qwen3_8b"
    assert "invarlock" in " ".join(payload[0]["evaluate"])
    assert "evaluation.report.json" in " ".join(payload[0]["verify"])

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite"] == "current-supported-experimental"
    assert manifest["lanes"][0]["slug"] == "qwen3_8b"


def test_model_evidence_sweep_returns_failure_when_verify_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence_sweep.py"
    fake_python = tmp_path / "fake-python"
    _write_fake_python(fake_python)
    output_root = tmp_path / "evidence"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--slug",
            "tinyllama_1_1b",
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
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is False
    assert len(summary["results"]) == 1
    result = summary["results"][0]
    assert result["slug"] == "tinyllama_1_1b"
    assert result["evaluate_exit"] == 0
    assert result["verify_exit"] == 1
    assert result["ok"] is False
    assert (output_root / "eval" / "tinyllama_1_1b" / "verify.json").is_file()
