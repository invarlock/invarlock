#!/usr/bin/env python3
"""Create a canonical candidate-wheel manifest for runtime qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

FORMAT_VERSION = "invarlock/qualification-candidate-wheels-v1"
_MAX_WHEEL_BYTES = 512 * 1024 * 1024
_MAX_WHEELS = 8


class CandidateWheelManifestError(RuntimeError):
    """One fail-closed manifest creation error."""


@dataclass(frozen=True)
class CandidateWheel:
    """Stable identity captured from one caller-selected wheel."""

    path: Path
    sha256: str
    file_identity: tuple[int, int]


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _capture_wheel(value: Path) -> CandidateWheel:
    lexical = Path(os.path.abspath(os.fspath(value)))
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise CandidateWheelManifestError("candidate wheel is unavailable") from exc
    if resolved != lexical or lexical.suffix != ".whl":
        raise CandidateWheelManifestError(
            "candidate wheel must be one real .whl path without symbolic links"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(lexical, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_WHEEL_BYTES:
            raise CandidateWheelManifestError(
                "candidate wheel must be one bounded regular file"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise CandidateWheelManifestError("candidate wheel could not be read") from exc
    finally:
        if descriptor >= 0:  # pragma: no branch - a successful open always sets it
            os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise CandidateWheelManifestError("candidate wheel changed while it was read")
    return CandidateWheel(
        path=lexical,
        sha256=f"sha256:{digest.hexdigest()}",
        file_identity=(before.st_dev, before.st_ino),
    )


def _destination(value: Path) -> Path:
    destination = Path(os.path.abspath(os.fspath(value)))
    if destination.exists() or destination.is_symlink():
        raise CandidateWheelManifestError("manifest destination already exists")
    try:
        parent = destination.parent.resolve(strict=True)
    except OSError as exc:
        raise CandidateWheelManifestError(
            "manifest parent directory is unavailable"
        ) from exc
    if parent != destination.parent or not parent.is_dir():
        raise CandidateWheelManifestError(
            "manifest parent must be one real directory without symbolic links"
        )
    return destination


def _publish_no_clobber(destination: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise CandidateWheelManifestError(
                "manifest destination already exists"
            ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def create_manifest(wheels: list[Path], *, output: Path) -> dict[str, object]:
    """Capture wheel identities and publish one canonical no-clobber manifest."""

    if not wheels or len(wheels) > _MAX_WHEELS:
        raise CandidateWheelManifestError(
            f"candidate manifest requires 1-{_MAX_WHEELS} wheels"
        )
    destination = _destination(output)
    captured = [_capture_wheel(wheel) for wheel in wheels]
    paths = [item.path for item in captured]
    identities = [item.file_identity for item in captured]
    if len(paths) != len(set(paths)) or len(identities) != len(set(identities)):
        raise CandidateWheelManifestError("candidate wheel is repeated")
    manifest: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "wheels": [
            {"path": str(item.path), "sha256": item.sha256}
            for item in sorted(captured, key=lambda item: str(item.path))
        ],
    }
    _publish_no_clobber(destination, _canonical_json(manifest))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = create_manifest(arguments.wheel, output=arguments.output)
    except CandidateWheelManifestError as exc:
        raise SystemExit(str(exc)) from exc
    print(_canonical_json(manifest).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI gate
    raise SystemExit(main())
