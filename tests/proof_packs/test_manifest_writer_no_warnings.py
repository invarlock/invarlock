from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_manifest_writer_runs_with_warnings_as_errors(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "proof_packs" / "python" / "manifest_writer.py"
    assert script.is_file()

    pack_dir = tmp_path / "pack"
    run_dir = tmp_path / "run_dir"
    (pack_dir / "results").mkdir(parents=True, exist_ok=True)
    (pack_dir / "state").mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal artifact to ensure artifacts list is non-empty.
    (pack_dir / "results" / "final_verdict.json").write_text("{}", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "error"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--pack-dir",
            str(pack_dir),
            "--run-dir",
            str(run_dir),
            "--suite",
            "subset",
            "--net",
            "1",
            "--determinism",
            "throughput",
            "--repeats",
            "0",
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    manifest_path = pack_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("format") == "proof-pack-v1"
    assert manifest.get("suite") == "subset"
    assert str(manifest.get("generated_at", "")).endswith("Z")
