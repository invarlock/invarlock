from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path


def test_model_evidence_sweep_captures_artifact_manifest_and_failure_reason(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    fake_python = tmp_path / "docker-denied-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-m" && "${2:-}" == "invarlock" && "${3:-}" == "evaluate" ]]; then
  echo "Unable to find image 'ghcr.io/invarlock/invarlock-runtime:latest' locally" >&2
  echo "docker: Error response from daemon: error from registry: denied" >&2
  exit 125
fi
echo "unexpected invocation: $*" >&2
exit 99
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    output_root = tmp_path / "evidence-container-denied"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite",
            "repo-mentioned-gpu",
            "--slug",
            "gemma4_e2b_public",
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
    result = summary["results"][0]
    assert result["detail"] == "container_image_pull_denied"
    assert result["evaluate_exit"] == 125

    revisions = json.loads(
        (output_root / "model_revisions.json").read_text(encoding="utf-8")
    )
    assert revisions["schema"] == "invarlock/model-evidence-model-revisions-v1"
    assert revisions["models"][0]["model_id"] == "google/gemma-4-E2B-it"

    artifact_manifest = json.loads(
        (output_root / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert artifact_manifest["schema"] == (
        "invarlock/model-evidence-artifact-manifest-v1"
    )
    assert artifact_manifest["ok"] is False
    paths = {entry["path"] for entry in artifact_manifest["files"]}
    assert {
        "manifest.json",
        "summary.json",
        "summary.tsv",
        "status.log",
        "model_revisions.json",
        "logs/gemma4_e2b_public.log",
    }.issubset(paths)
    assert artifact_manifest["lane_results"][0]["detail"] == (
        "container_image_pull_denied"
    )
