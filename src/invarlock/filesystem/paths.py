"""Descriptor-pinned directory traversal shared by readers and publishers."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class UnsafePathError(OSError):
    """The supplied path does not meet the directory contract."""


class PathChangedError(UnsafePathError):
    """A directory entry no longer names the directory opened by this operation."""


def entry_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode


@contextmanager
def _directory_descriptor(
    path: str, flags: int, *, dir_fd: int | None = None
) -> Iterator[int]:
    descriptor = os.open(path, flags, dir_fd=dir_fd)
    try:
        yield descriptor
    finally:
        # Keep allocation and release in this context manager so static analysis
        # and readers can see the complete ownership lifetime.
        try:
            os.close(descriptor)
        except OSError:
            # A failed close may already have released the descriptor. Retrying
            # could close an unrelated descriptor that reused the same number.
            pass


@contextmanager
def _pinned_descendants(
    parent: int,
    names: tuple[str, ...],
    *,
    flags: int,
    traversal_flags: int,
    create: bool,
    bindings: list[tuple[int, str, tuple[int, int, int]]],
    index: int = 0,
) -> Iterator[int]:
    if index == len(names):
        yield parent
        return

    name = names[index]
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise UnsafePathError("path must use non-symlink directories")
    try:
        descriptor = os.open(
            name,
            flags if index == len(names) - 1 else traversal_flags,
            dir_fd=parent,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENOENT}:
            raise PathChangedError("directory changed while opening") from exc
        raise
    try:
        identity = entry_identity(before)
        if identity != entry_identity(os.fstat(descriptor)):
            raise PathChangedError("directory changed while opening")
        bindings.append((parent, name, identity))
        with _pinned_descendants(
            descriptor,
            names,
            flags=flags,
            traversal_flags=traversal_flags,
            create=create,
            bindings=bindings,
            index=index + 1,
        ) as leaf:
            yield leaf
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


@contextmanager
def pinned_directory(path: Path, *, create: bool = False) -> Iterator[int]:
    """Reject symlinks and retain/check every ancestor through the operation.

    These checks detect changed bindings; they do not lock caller-owned paths
    against changes after the operation returns. Preserve a primary operation
    failure instead of masking it with a subsequent binding check. On Linux,
    ancestors need only traversal permission; the final descriptor remains
    readable for directory listing and synchronization.
    """
    path = Path(path).absolute()
    if ".." in path.parts:
        raise UnsafePathError("directory must not contain parent traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    traversal_flags = flags | getattr(os, "O_PATH", 0)
    names = path.parts[1:]
    bindings: list[tuple[int, str, tuple[int, int, int]]] = []
    with _directory_descriptor(
        path.anchor, traversal_flags if names else flags
    ) as root:
        with _pinned_descendants(
            root,
            names,
            flags=flags,
            traversal_flags=traversal_flags,
            create=create,
            bindings=bindings,
        ) as current:
            yield current
            for parent, name, identity in bindings:
                try:
                    named = os.stat(name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError as exc:
                    raise PathChangedError("directory source was replaced") from exc
                if entry_identity(named) != identity:
                    raise PathChangedError("directory source was replaced")
