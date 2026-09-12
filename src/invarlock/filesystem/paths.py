"""Descriptor-pinned directory traversal shared by readers and publishers."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
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
    with ExitStack() as descriptors:
        current = descriptors.enter_context(
            _directory_descriptor(path.anchor, traversal_flags if names else flags)
        )
        for index, name in enumerate(names):
            parent = current
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=parent)
                except FileExistsError:
                    pass
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise UnsafePathError("path must use non-symlink directories")
            try:
                current = descriptors.enter_context(
                    _directory_descriptor(
                        name,
                        flags if index == len(names) - 1 else traversal_flags,
                        dir_fd=parent,
                    )
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENOENT}:
                    raise PathChangedError("directory changed while opening") from exc
                raise
            identity = entry_identity(before)
            if identity != entry_identity(os.fstat(current)):
                raise PathChangedError("directory changed while opening")
            bindings.append((parent, name, identity))
        yield current
        for parent, name, identity in bindings:
            try:
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise PathChangedError("directory source was replaced") from exc
            if entry_identity(named) != identity:
                raise PathChangedError("directory source was replaced")
