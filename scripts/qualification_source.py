#!/usr/bin/env python3
"""Create or authenticate the exact Git source identity used by qualification."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path

_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_BUNDLE_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 50_000


def _git() -> str:
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        executable = shutil.which("git")
    if executable is None:
        raise SystemExit("git is required")
    return str(Path(executable).resolve(strict=True))


def _git_environment() -> dict[str, str]:
    return {
        "GIT_NO_REPLACE_OBJECTS": "1",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }


def _commit_identity(repository: Path, reference: str) -> str:
    completed = subprocess.run(
        [
            _git(),
            "--no-replace-objects",
            "-C",
            str(repository),
            "rev-parse",
            "--verify",
            f"{reference}^{{commit}}",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_git_environment(),
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or _COMMIT.fullmatch(commit) is None:
        raise SystemExit("source reference does not identify one Git commit")
    return commit


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def _regular_file_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SystemExit("source bundle is unavailable") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise SystemExit("source bundle must be a regular file")
        if opened.st_size > _MAX_BUNDLE_BYTES:
            raise SystemExit("source bundle exceeds the size limit")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise SystemExit("source bundle changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SystemExit("source bundle changed while being read")
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SystemExit("source bundle changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_archive(payload: bytes, *, commit: str) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            if (archive.pax_headers or {}).get("comment") != commit:
                raise SystemExit("source bundle does not bind the selected commit")
            members = archive.getmembers()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise SystemExit("source bundle contains too many entries")
            for member in members:
                relative = Path(member.name)
                if (
                    not member.name
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise SystemExit("source bundle contains an unsafe entry")
    except (OSError, tarfile.TarError) as exc:
        raise SystemExit("source bundle is not an exact Git tar archive") from exc


def authenticate_bundle(
    *,
    repository: Path,
    commit: str,
    bundle: Path,
    bundle_sha256: str,
) -> tuple[dict[str, object], bytes]:
    """Authenticate exact Git archive bytes and return them for build-context use."""

    if _COMMIT.fullmatch(commit) is None:
        raise SystemExit("source commit must be 40-64 lowercase hexadecimal characters")
    if _DIGEST.fullmatch(bundle_sha256) is None:
        raise SystemExit("source bundle digest must be a lowercase sha256 digest")
    repository = repository.resolve(strict=True)
    observed_commit = _commit_identity(repository, commit)
    if observed_commit != commit:
        raise SystemExit("source commit does not match the selected Git object")
    payload = _regular_file_bytes(bundle)
    observed_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if observed_digest != bundle_sha256:
        raise SystemExit("source bundle does not match its declared digest")
    _validate_archive(payload, commit=commit)

    with tempfile.TemporaryDirectory(prefix="invarlock-source-verify-") as temporary:
        expected = Path(temporary) / "expected.tar"
        completed = subprocess.run(
            [
                _git(),
                "--no-replace-objects",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                f"--output={expected}",
                commit,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=_git_environment(),
        )
        if completed.returncode != 0:
            raise SystemExit("source commit could not be archived for authentication")
        expected_payload = _regular_file_bytes(expected)
    if payload != expected_payload:
        raise SystemExit("source bundle bytes do not match the selected Git commit")
    return (
        {
            "format_version": "invarlock/qualification-source-v1",
            "ok": True,
            "source_bundle": str(Path(bundle).resolve()),
            "source_bundle_sha256": observed_digest,
            "source_commit": observed_commit,
        },
        payload,
    )


def _create(repository: Path, reference: str, output: Path) -> dict[str, object]:
    repository = repository.resolve(strict=True)
    commit = _commit_identity(repository, reference)
    destination = Path(os.path.abspath(os.fspath(output)))
    if destination.exists() or destination.is_symlink():
        raise SystemExit("source bundle destination already exists")
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or not parent.is_dir() or parent.is_symlink():
        raise SystemExit("source bundle parent must be one real directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        completed = subprocess.run(
            [
                _git(),
                "--no-replace-objects",
                "-C",
                str(repository),
                "archive",
                "--format=tar",
                f"--output={temporary}",
                commit,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            env=_git_environment(),
        )
        if completed.returncode != 0:
            raise SystemExit("Git source archive creation failed")
        with tarfile.open(temporary, mode="r:") as archive:
            if (archive.pax_headers or {}).get("comment") != commit:
                raise SystemExit("Git source archive does not bind the selected commit")
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise SystemExit("source bundle destination already exists") from exc
        digest = _sha256(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "format_version": "invarlock/qualification-source-v1",
        "ok": True,
        "source_bundle": str(destination),
        "source_bundle_sha256": digest,
        "source_commit": commit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--repository", type=Path, default=Path.cwd())
    create.add_argument("--commit", default="HEAD")
    create.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--repository", type=Path, default=Path.cwd())
    verify.add_argument("--commit", required=True)
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--bundle-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "create":
        result = _create(arguments.repository, arguments.commit, arguments.output)
    else:
        result, _payload = authenticate_bundle(
            repository=arguments.repository,
            commit=arguments.commit,
            bundle=arguments.bundle,
            bundle_sha256=arguments.bundle_sha256,
        )
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
