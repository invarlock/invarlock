"""Bounded, descriptor-based copies for immutable client handoffs."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

MAX_ENTRIES = 4096
MAX_BYTES = 2 * 1024**3
MAX_DEPTH = 16
_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _walk(
    root: int, target: Path | None
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    entries: dict[str, tuple[int, ...]] = {}
    hashes: dict[str, str] = {}
    total = 0

    def visit(descriptor: int, relative: str, depth: int) -> None:
        nonlocal total
        if depth > MAX_DEPTH:
            raise ValueError("evidence directory nesting exceeds handoff limit")
        before = _identity(os.fstat(descriptor))
        entries[relative] = before
        for name in sorted(os.listdir(descriptor)):
            if len(entries) >= MAX_ENTRIES:
                raise ValueError("evidence exceeds handoff entry limit")
            if "\\" in name or any(ord(c) < 32 for c in name):
                raise ValueError("evidence filename is not portable in a handoff")
            member = f"{relative}/{name}" if relative else name
            expected = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not (stat.S_ISREG(expected.st_mode) or stat.S_ISDIR(expected.st_mode)):
                raise ValueError(
                    "evidence must contain only regular files and directories"
                )
            flags = _FLAGS | (os.O_DIRECTORY if stat.S_ISDIR(expected.st_mode) else 0)
            child = os.open(name, flags, dir_fd=descriptor)
            try:
                if _identity(os.fstat(child)) != _identity(expected):
                    raise ValueError("evidence changed before copying")
                if stat.S_ISDIR(expected.st_mode):
                    if target is not None:
                        (target / member).mkdir()
                    visit(child, member, depth + 1)
                else:
                    entries[member] = _identity(expected)
                    total += expected.st_size
                    if total > MAX_BYTES:
                        raise ValueError(
                            "evidence exceeds the 2 GiB handoff copy limit"
                        )
                    if target is not None:
                        digest = hashlib.sha256()
                        remaining = expected.st_size
                        with (target / member).open("xb") as output:
                            while chunk := os.read(child, min(65536, remaining + 1)):
                                remaining -= len(chunk)
                                if remaining < 0:
                                    raise ValueError("evidence grew while copying")
                                output.write(chunk)
                                digest.update(chunk)
                        if remaining:
                            raise ValueError("evidence shrank while copying")
                        hashes[member] = digest.hexdigest()
                if _identity(os.fstat(child)) != _identity(expected) or _identity(
                    os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                ) != _identity(expected):
                    raise ValueError("evidence changed while copying")
            finally:
                os.close(child)
        if _identity(os.fstat(descriptor)) != before:
            raise ValueError("evidence directory changed while copying")

    visit(root, "", 0)
    return entries, hashes


def copy_evidence(
    root: int, destination: Path
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Copy every file once, without following links or loading the pack into RAM."""
    destination.mkdir()
    return _walk(root, destination)


def require_unchanged(root: int, expected: dict[str, tuple[int, ...]]) -> None:
    if _walk(root, None)[0] != expected:
        raise ValueError("evidence changed during handoff preparation")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
