from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "qualification_source.py"


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    repository.joinpath("source.txt").write_text("authenticated\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "source.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, commit


def test_create_emits_an_exact_no_clobber_git_archive(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    output = tmp_path / "source.tar"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "create",
            "--repository",
            str(repository),
            "--commit",
            commit,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["source_commit"] == commit
    assert result["source_bundle"] == str(output)
    assert result["source_bundle_sha256"].startswith("sha256:")
    with tarfile.open(output, mode="r:") as archive:
        assert archive.pax_headers["comment"] == commit
        assert archive.extractfile("source.txt").read() == b"authenticated\n"

    repeated = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "create",
            "--repository",
            str(repository),
            "--commit",
            commit,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert output.read_bytes()


def test_create_ignores_git_replacement_objects(tmp_path: Path) -> None:
    repository, clean = _repository(tmp_path)
    source = repository / "source.txt"
    source.write_text("malicious\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qam", "malicious"], check=True
    )
    malicious = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repository), "replace", clean, malicious], check=True
    )
    output = tmp_path / "clean-source.tar"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "create",
            "--repository",
            str(repository),
            "--commit",
            clean,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    with tarfile.open(output, mode="r:") as archive:
        assert archive.extractfile("source.txt").read() == b"authenticated\n"


def test_verify_authenticates_the_exact_git_archive(tmp_path: Path) -> None:
    repository, commit = _repository(tmp_path)
    output = tmp_path / "source.tar"
    created = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "create",
            "--repository",
            str(repository),
            "--commit",
            commit,
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = json.loads(created.stdout)["source_bundle_sha256"]

    verified = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            "--repository",
            str(repository),
            "--commit",
            commit,
            "--bundle",
            str(output),
            "--bundle-sha256",
            digest,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert verified.returncode == 0, verified.stderr
    result = json.loads(verified.stdout)
    assert result["ok"] is True
    assert result["source_commit"] == commit
    assert result["source_bundle_sha256"] == digest


@pytest.mark.parametrize(
    ("commit", "digest", "message"),
    (
        ("not-a-commit", "sha256:" + "a" * 64, "source commit"),
        ("a" * 40, "unbound", "source bundle digest"),
    ),
)
def test_verify_rejects_unbound_source_identity(
    commit: str, digest: str, message: str
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            "--repository",
            str(ROOT),
            "--commit",
            commit,
            "--bundle",
            str(SCRIPT),
            "--bundle-sha256",
            digest,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
