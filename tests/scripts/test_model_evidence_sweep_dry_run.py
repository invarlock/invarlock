from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_model_evidence_sweep_dry_run_emits_commands_and_manifest(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "model_evidence" / "model_evidence_sweep.py"
    output_root = tmp_path / "evidence"

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
    assert payload[0]["slug"] == "gemma4_e2b_public"
    assert payload[0]["execution_mode"] == "container"
    assert "invarlock" in " ".join(payload[0]["evaluate"])
    assert "evaluation.report.json" in " ".join(payload[0]["verify"])
    preset_idx = payload[0]["evaluate"].index("--preset") + 1
    assert (
        payload[0]["evaluate"][preset_idx]
        == "configs/presets/causal_lm/gemma4_e2b_512.yaml"
    )
    assert "--allow-host-execution" not in payload[0]["evaluate"]
    assert "--runtime-provenance" not in payload[0]["verify"]

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite"] == "repo-mentioned-gpu"
    assert manifest["execution_mode"] == "container"
    assert manifest["lanes"][0]["slug"] == "gemma4_e2b_public"
