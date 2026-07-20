from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qualification_source

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


def _tar_payload(
    *, comment: str | None, member: tarfile.TarInfo | None = None
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output, mode="w", pax_headers={"comment": comment} if comment else None
    ) as archive:
        if member is not None:
            archive.addfile(member, io.BytesIO(b"x") if member.isfile() else None)
    return output.getvalue()


def test_source_helpers_fail_closed_on_missing_tools_and_unsafe_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    def which(name: str, path: str | None = None) -> str | None:
        calls.append(path)
        return None if path is not None else "/usr/bin/git"

    monkeypatch.setattr(qualification_source.shutil, "which", which)
    assert qualification_source._git().endswith("/git")
    assert calls == [qualification_source.os.defpath, None]
    monkeypatch.setattr(
        qualification_source.shutil, "which", lambda *_args, **_kwargs: None
    )
    with pytest.raises(SystemExit, match="git is required"):
        qualification_source._git()

    with pytest.raises(SystemExit, match="unavailable"):
        qualification_source._regular_file_bytes(tmp_path / "missing.tar")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(SystemExit, match="regular file"):
        qualification_source._regular_file_bytes(directory)
    oversized = tmp_path / "oversized.tar"
    oversized.write_bytes(b"x")
    monkeypatch.setattr(qualification_source, "_MAX_BUNDLE_BYTES", 0)
    with pytest.raises(SystemExit, match="size limit"):
        qualification_source._regular_file_bytes(oversized)


def test_archive_validation_rejects_unbound_invalid_and_unsafe_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    with pytest.raises(SystemExit, match="exact Git tar archive"):
        qualification_source._validate_archive(b"not a tar", commit=commit)
    with pytest.raises(SystemExit, match="does not bind"):
        qualification_source._validate_archive(
            _tar_payload(comment=None), commit=commit
        )

    monkeypatch.setattr(qualification_source, "_MAX_ARCHIVE_MEMBERS", 0)
    member = tarfile.TarInfo("file.txt")
    member.size = 1
    with pytest.raises(SystemExit, match="too many entries"):
        qualification_source._validate_archive(
            _tar_payload(comment=commit, member=member), commit=commit
        )
    monkeypatch.setattr(qualification_source, "_MAX_ARCHIVE_MEMBERS", 10)
    unsafe = tarfile.TarInfo("link")
    unsafe.type = tarfile.SYMTYPE
    unsafe.linkname = "target"
    with pytest.raises(SystemExit, match="unsafe entry"):
        qualification_source._validate_archive(
            _tar_payload(comment=commit, member=unsafe), commit=commit
        )


def test_authenticate_and_create_reject_identity_and_archive_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    commit = "a" * 40
    bundle = tmp_path / "bundle.tar"
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(
        qualification_source, "_commit_identity", lambda *_args: "b" * 40
    )
    with pytest.raises(SystemExit, match="selected Git object"):
        qualification_source.authenticate_bundle(
            repository=repository,
            commit=commit,
            bundle=bundle,
            bundle_sha256="sha256:" + "c" * 64,
        )

    monkeypatch.setattr(qualification_source, "_commit_identity", lambda *_args: commit)
    monkeypatch.setattr(
        qualification_source.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(SystemExit, match="archive creation failed"):
        qualification_source._create(repository, commit, tmp_path / "created.tar")

    existing_parent = tmp_path / "parent-file"
    existing_parent.write_text("not a directory", encoding="utf-8")
    with pytest.raises((SystemExit, NotADirectoryError), match="parent"):
        qualification_source._create(
            repository, commit, existing_parent / "created.tar"
        )
