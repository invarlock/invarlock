from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_manifest_writer_runs_with_warnings_as_errors(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "evidence_packs" / "python" / "manifest_writer.py"
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
    assert manifest.get("format") == "evidence-pack-v1"
    assert manifest.get("suite") == "subset"
    assert str(manifest.get("generated_at", "")).endswith("Z")
    assert manifest.get("builder", {}).get("id") == "invarlock/evidence-pack@v1"
    assert manifest.get("subject", {}).get("path") == "results/final_verdict.json"
    assert isinstance(manifest.get("materials"), list)
    assert "config_source" in (manifest.get("invocation") or {})


def test_write_source_repo_metadata_fails_closed_without_git(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "scripts"
        / "evidence_packs"
        / "python"
        / "write_source_repo_metadata.py"
    )
    out_path = tmp_path / "source_repo.json"
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()

    proc = subprocess.run(
        [sys.executable, str(script), "--out", str(out_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env={**os.environ, "PATH": str(empty_path)},
    )

    assert proc.returncode == 1
    assert "git is required to collect evidence-pack source provenance" in proc.stderr
    assert not out_path.exists()


def test_manifest_writer_uses_existing_source_repo_metadata_without_git(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "evidence_packs" / "python" / "manifest_writer.py"
    pack_dir = tmp_path / "pack"
    run_dir = tmp_path / "run_dir"
    (pack_dir / "results").mkdir(parents=True, exist_ok=True)
    (pack_dir / "metadata").mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "results" / "final_verdict.json").write_text("{}", encoding="utf-8")
    (pack_dir / "metadata" / "source_repo.json").write_text(
        json.dumps(
            {
                "uri": "git+https://example.invalid/invarlock.git",
                "commit": "abc123",
                "branch": "staging/next",
                "describe": "v0.5.0-12-gabc123",
                "dirty": False,
            }
        ),
        encoding="utf-8",
    )
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()

    proc = subprocess.run(
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
            "0",
            "--determinism",
            "strict",
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env={**os.environ, "PATH": str(empty_path)},
    )

    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    config_source = manifest["invocation"]["config_source"]
    assert config_source["commit"] == "abc123"
    assert config_source["branch"] == "staging/next"
    assert config_source["describe"] == "v0.5.0-12-gabc123"
    assert config_source["uri"] == "git+https://example.invalid/invarlock.git"
    assert config_source["dirty"] is False


def test_manifest_writer_fails_closed_without_source_repo_metadata_or_git(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "evidence_packs" / "python" / "manifest_writer.py"
    pack_dir = tmp_path / "pack"
    run_dir = tmp_path / "run_dir"
    (pack_dir / "results").mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "results" / "final_verdict.json").write_text("{}", encoding="utf-8")
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()

    proc = subprocess.run(
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
            "0",
            "--determinism",
            "strict",
            "--repeats",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env={**os.environ, "PATH": str(empty_path)},
    )

    assert proc.returncode == 1
    assert "git is required to collect evidence-pack source provenance" in proc.stderr
    assert not (pack_dir / "manifest.json").exists()
