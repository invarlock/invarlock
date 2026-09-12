"""No-replace publication of complete files through retained descriptors."""

from __future__ import annotations

import ctypes
import os
import secrets
import stat
from pathlib import Path

from .atomic_directory import _rename_no_replace
from .paths import (
    PathChangedError,
    UnsafePathError,
    entry_identity,
    pinned_directory,
)


def _contents_identity(value: os.stat_result) -> tuple[int, ...]:
    # Rename may change ctime, but must not change the file's contents or mode.
    return (*entry_identity(value), value.st_size, value.st_mtime_ns)


def _cleanup_stage(parent: int, name: str, stage: int, file_fd: int | None) -> None:
    """Remove only the known leaf through the private directory descriptor.

    Never traverse an old staging pathname or recursively remove its contents.
    Cleanup is best effort and cannot turn a committed file into a failure.
    A concurrent parent writer can substitute an empty root between stat/rmdir;
    this bounded cleanup never removes a nonempty replacement directory.
    """
    try:
        if file_fd is not None:
            named = os.stat("payload", dir_fd=stage, follow_symlinks=False)
            if entry_identity(named) == entry_identity(os.fstat(file_fd)):
                os.unlink("payload", dir_fd=stage)
    except OSError:
        pass
    try:
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if entry_identity(named) == entry_identity(os.fstat(stage)):
            os.rmdir(name, dir_fd=parent)
    except OSError:
        pass


def write_file_no_replace(
    path: Path, payload: bytes, *, mode: int = 0o600, create_parents: bool = True
) -> None:
    """Install the completed inode at an absent destination, then check binding.

    Private staging and descriptor-relative operations protect against unrelated
    writers in the parent directory. The caller must still control processes
    with the same filesystem identity. No existing destination is overwritten.
    If a post-publication check fails, a completed file may remain; it is never
    deleted by pathname rollback. These checks cannot prevent later modification.
    """
    path = Path(path)
    if path.name in {"", ".", ".."}:
        raise UnsafePathError("destination must name a file")
    with pinned_directory(path.parent, create=create_parents) as parent:
        for _attempt in range(10):
            name = ".invarlock-write-" + secrets.token_hex(16)
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent)
            except FileExistsError:
                continue
            break
        else:
            raise OSError("could not allocate a private staging directory")
        try:
            stage = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent,
            )
        except OSError:
            # mkdir succeeded, but no descriptor was obtained. Remove at most
            # the empty staging root; never traverse a possibly replaced name.
            try:
                os.rmdir(name, dir_fd=parent)
            except OSError:
                pass
            raise
        descriptor: int | None = None
        try:
            stage_stat = os.fstat(stage)
            if (
                stage_stat.st_uid != os.geteuid()
                or stat.S_IMODE(stage_stat.st_mode) != 0o700
            ):
                raise PathChangedError("private staging directory identity changed")
            descriptor = os.open(
                "payload",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=stage,
            )
            # Retain ownership even if stream construction/closure fails.
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fchmod(descriptor, mode)
                os.fsync(descriptor)
            expected = _contents_identity(os.fstat(descriptor))
            if (
                _contents_identity(
                    os.stat("payload", dir_fd=stage, follow_symlinks=False)
                )
                != expected
            ):
                raise PathChangedError(
                    "staged file identity changed before publication"
                )
            os.fsync(stage)
            os.fsync(parent)
            ctypes.set_errno(0)
            result = _rename_no_replace(
                source_parent_fd=stage,
                source_name=b"payload",
                destination_parent_fd=parent,
                destination_name=os.fsencode(path.name),
            )
            if result != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(path))
            # Persist the new destination entry. Failure after rename is
            # explicitly ambiguous and must never unlink the final pathname.
            os.fsync(parent)
            published = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (
                _contents_identity(published) != expected
                or _contents_identity(os.fstat(descriptor)) != expected
            ):
                raise PathChangedError(
                    "published file identity changed; destination retained"
                )
        finally:
            try:
                _cleanup_stage(parent, name, stage, descriptor)
            finally:
                try:
                    if descriptor is not None:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
                finally:
                    try:
                        os.close(stage)
                    except OSError:
                        pass
