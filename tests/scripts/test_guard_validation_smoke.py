from __future__ import annotations

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
    assert all(0.0 <= row["type_i_error"] <= 1.0 for row in payload["rate_rows"])
    assert all(0.0 <= row["power"] <= 1.0 for row in payload["rate_rows"])
    assert "synthetic smoke" in payload["scope"]
    assert "| Guard | Windows | Type-I Error | Power |" in markdown
