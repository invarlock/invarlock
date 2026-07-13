from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_guard_validation_smoke_writes_deterministic_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "guard-validation"

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "smoke" / "guard_validation_smoke.py"),
            "--output-dir",
            str(out_dir),
            "--replicates",
            "25",
            "--seed",
            "11",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(
        (out_dir / "guard-validation-smoke.json").read_text(encoding="utf-8")
    )
    markdown = (out_dir / "guard-validation-smoke.md").read_text(encoding="utf-8")
    assert payload["schema"] == "invarlock/guard-validation-smoke-v1"
    assert payload["seed"] == 11
    assert payload["replicates"] == 25
    assert {row["guard"] for row in payload["rate_rows"]} == {
        "spectral",
        "rmt",
        "variance",
    }
    assert all(0.0 <= row["null_trigger_rate"] <= 1.0 for row in payload["rate_rows"])
    assert all(
        0.0 <= row["shifted_trigger_rate"] <= 1.0 for row in payload["rate_rows"]
    )
    assert all(len(row["null_outcomes"]) == 25 for row in payload["rate_rows"])
    assert all(len(row["shifted_outcomes"]) == 25 for row in payload["rate_rows"])
    assert all(
        row["null_trigger_count"] == sum(row["null_outcomes"])
        for row in payload["rate_rows"]
    )
    assert all(
        row["shifted_trigger_count"] == sum(row["shifted_outcomes"])
        for row in payload["rate_rows"]
    )
    assert "synthetic production-primitive smoke" in payload["scope"]
    assert payload["production_primitives"] == {
        "spectral": {
            "entrypoint": (
                "invarlock.guards.spectral_detection.summarize_family_z_scores"
            ),
            "role": "violation_summary",
        },
        "rmt": {
            "entrypoint": "invarlock.guards.rmt_policy.compute_epsilon_violations",
            "role": "violation_detection",
        },
        "variance": {
            "entrypoint": "invarlock.guards.variance_policy.predictive_gate_outcome",
            "role": "gate_outcome",
        },
    }
    assert {
        row["guard"]: {
            "entrypoint": row["production_entrypoint"],
            "role": row["primitive_role"],
        }
        for row in payload["rate_rows"]
    } == payload["production_primitives"]
    assert (
        "| Guard | Windows | Synthetic Null Trigger | Synthetic Shifted Trigger |"
        in markdown
    )
    assert payload["evidence_sha256"] in markdown
    assert payload["markdown_sha256"] == (
        "sha256:" + hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    )


def test_guard_validation_smoke_rejects_invalid_replicates(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "smoke" / "guard_validation_smoke.py"),
            "--output-dir",
            str(tmp_path / "guard-validation"),
            "--replicates",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert "replicates must be in [1, 10000]" in proc.stderr
