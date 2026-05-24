from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_release_checklist_exists_outside_published_docs_tree() -> None:
    repo_root = _repo_root()
    checklist = repo_root / ".github" / "release-checklist.md"
    assert checklist.is_file()
    assert not (repo_root / "docs" / "release").exists()
    text = checklist.read_text(encoding="utf-8")
    for required in (
        "make verify",
        "make coverage-enforce",
        "make dist-check",
        "make security",
        "make container-front-door-smoke",
        "make release-evidence-check",
        "wheel-sdist-hashes.txt",
        "runtime-image-digest.txt",
        "strict/evaluation.report.json",
        "strict/verify.json",
    ):
        assert required in text


def test_release_evidence_check_passes_with_required_artifacts(tmp_path: Path) -> None:
    repo_root = _repo_root()
    dist = tmp_path / "dist"
    release = tmp_path / "release"
    sbom = tmp_path / "sbom.json"
    dist.mkdir()
    (dist / "invarlock-0.9.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")
    (dist / "invarlock-0.9.0.tar.gz").write_text("sdist", encoding="utf-8")
    (release / "strict").mkdir(parents=True)
    (release / "wheel-sdist-hashes.txt").write_text("hashes", encoding="utf-8")
    (release / "runtime-image-digest.txt").write_text(
        "sha256:" + "1" * 64,
        encoding="utf-8",
    )
    (release / "strict" / "evaluation.report.json").write_text("{}", encoding="utf-8")
    (release / "strict" / "verify.json").write_text("{}", encoding="utf-8")
    sbom.write_text("{}", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "release" / "check_release_evidence.py"),
            "--root",
            str(release),
            "--dist",
            str(dist),
            "--sbom",
            str(sbom),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert '"ok": true' in proc.stdout


def test_release_evidence_check_reports_missing_artifacts(tmp_path: Path) -> None:
    repo_root = _repo_root()

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "release" / "check_release_evidence.py"),
            "--root",
            str(tmp_path / "release"),
            "--dist",
            str(tmp_path / "dist"),
            "--sbom",
            str(tmp_path / "sbom.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "wheel artifact missing" in proc.stderr
    assert "strict example report missing" in proc.stderr
