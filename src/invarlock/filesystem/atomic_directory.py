"""Fail-closed atomic publication for completed directory trees."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import stat
from pathlib import Path

_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


class AtomicDirectoryPublicationError(OSError):
    """Raised when a directory cannot be safely published."""


class AtomicDirectoryExistsError(AtomicDirectoryPublicationError):
    """Raised when the destination already exists at the publication instant."""


def _directory_open_flags() -> int:
    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(directory, int) or not isinstance(nofollow, int):
        raise AtomicDirectoryPublicationError(
            "secure descriptor-relative directory opening is unavailable"
        )
    return os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)


def _open_directory(path: Path, *, label: str) -> int:
    """Open a directory without following a symlink in any path component."""

    if not path.is_absolute():
        raise AtomicDirectoryPublicationError(f"{label} must use an absolute path")
    components = path.parts[1:]
    if ".." in components:
        raise AtomicDirectoryPublicationError(
            f"{label} must not contain parent traversal"
        )
    flags = _directory_open_flags()
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise AtomicDirectoryPublicationError(
            f"{label} must be an existing non-symlink directory"
        ) from exc
    try:
        for component in components:
            if component in {"", "."}:
                continue
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise AtomicDirectoryPublicationError(
                    f"{label} must be an existing non-symlink directory"
                ) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise AtomicDirectoryPublicationError(
                f"{label} must be an existing non-symlink directory"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_child_directory(parent_fd: int, name: str) -> int:
    flags = _directory_open_flags()
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise AtomicDirectoryPublicationError(
            "staging must be an existing non-symlink directory"
        ) from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise AtomicDirectoryPublicationError(
                "staging must be an existing non-symlink directory"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _rename_no_replace(
    *,
    source_parent_fd: int,
    source_name: bytes,
    destination_parent_fd: int,
    destination_name: bytes,
) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    system = platform.system()
    if system == "Linux":
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise AtomicDirectoryPublicationError(
                "atomic no-replace directory publication is unavailable on Linux"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        return int(
            renameat2(
                source_parent_fd,
                source_name,
                destination_parent_fd,
                destination_name,
                _RENAME_NOREPLACE,
            )
        )
    if system == "Darwin":
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise AtomicDirectoryPublicationError(
                "atomic descriptor-relative no-replace directory publication is "
                "unavailable on macOS"
            )
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        return int(
            renameatx_np(
                source_parent_fd,
                source_name,
                destination_parent_fd,
                destination_name,
                _RENAME_EXCL,
            )
        )
    raise AtomicDirectoryPublicationError(
        "atomic no-replace directory publication is unsupported on this platform"
    )


def _validate_basename(path: Path, *, label: str) -> str:
    name = path.name
    if not name or name in {".", ".."} or os.sep in name:
        raise AtomicDirectoryPublicationError(f"{label} must name one directory entry")
    if os.altsep is not None and os.altsep in name:
        raise AtomicDirectoryPublicationError(f"{label} must name one directory entry")
    return name


def _raise_rename_error(error_number: int) -> None:
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise AtomicDirectoryExistsError(
            error_number,
            "refusing to replace an existing publication destination",
        )
    raise AtomicDirectoryPublicationError(
        error_number,
        "atomic no-replace directory publication failed: " + os.strerror(error_number),
    )


def _entry_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode


def _require_directory_binding(path: Path, descriptor: int, *, label: str) -> None:
    expected = os.fstat(descriptor)
    try:
        named = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise AtomicDirectoryPublicationError(f"{label} identity changed") from exc
    if not stat.S_ISDIR(named.st_mode) or _entry_identity(named) != _entry_identity(
        expected
    ):
        raise AtomicDirectoryPublicationError(f"{label} identity changed")


def _restore_rejected_publication(
    *,
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    rejected_identity: tuple[int, int, int],
) -> None:
    """Move a post-rename identity mismatch out of the destination name."""

    ctypes.set_errno(0)
    result = _rename_no_replace(
        source_parent_fd=destination_parent_fd,
        source_name=os.fsencode(destination_name),
        destination_parent_fd=source_parent_fd,
        destination_name=os.fsencode(source_name),
    )
    if result != 0:
        raise AtomicDirectoryPublicationError(
            ctypes.get_errno(),
            "rejected publication could not be restored to the staging name",
        )
    try:
        os.stat(
            destination_name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise AtomicDirectoryPublicationError(
            "rejected publication destination could not be verified absent"
        ) from exc
    else:
        raise AtomicDirectoryPublicationError(
            "rejected publication remained at the destination name"
        )
    try:
        restored = os.stat(
            source_name,
            dir_fd=source_parent_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise AtomicDirectoryPublicationError(
            "rejected publication could not be verified at the staging name"
        ) from exc
    if _entry_identity(restored) != rejected_identity:
        raise AtomicDirectoryPublicationError(
            "restored rejected publication identity does not match"
        )


def publish_directory_no_replace(staging: Path, destination: Path) -> None:
    """Atomically rename one staged directory into an absent destination.

    The operation is descriptor-relative and delegates exclusivity to one kernel
    rename operation. It never performs an existence check followed by a rename.
    """

    source = Path(staging)
    target = Path(destination)
    base = Path.cwd()
    if not source.is_absolute():
        source = base / source
    if not target.is_absolute():
        target = base / target
    source_name = _validate_basename(source, label="staging")
    target_name = _validate_basename(target, label="destination")
    if source == target:
        raise AtomicDirectoryPublicationError(
            "staging and destination must name different directory entries"
        )

    source_parent_fd = _open_directory(source.parent, label="staging parent")
    try:
        destination_parent_fd = _open_directory(
            target.parent, label="destination parent"
        )
        try:
            source_fd = _open_child_directory(source_parent_fd, source_name)
            try:
                source_identity = os.fstat(source_fd)
                try:
                    named_source = os.stat(
                        source_name,
                        dir_fd=source_parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise AtomicDirectoryPublicationError(
                        "staging identity changed before publication"
                    ) from exc
                if _entry_identity(named_source) != _entry_identity(source_identity):
                    raise AtomicDirectoryPublicationError(
                        "staging identity changed before publication"
                    )
                _require_directory_binding(
                    source.parent,
                    source_parent_fd,
                    label="staging parent",
                )
                _require_directory_binding(
                    target.parent,
                    destination_parent_fd,
                    label="destination parent",
                )
                ctypes.set_errno(0)
                result = _rename_no_replace(
                    source_parent_fd=source_parent_fd,
                    source_name=os.fsencode(source_name),
                    destination_parent_fd=destination_parent_fd,
                    destination_name=os.fsencode(target_name),
                )
                if result != 0:
                    _raise_rename_error(ctypes.get_errno())

                try:
                    published = os.stat(
                        target_name,
                        dir_fd=destination_parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise AtomicDirectoryPublicationError(
                        "published directory identity could not be verified"
                    ) from exc
                if not stat.S_ISDIR(published.st_mode) or (
                    published.st_dev,
                    published.st_ino,
                ) != (source_identity.st_dev, source_identity.st_ino):
                    _restore_rejected_publication(
                        source_parent_fd=source_parent_fd,
                        source_name=source_name,
                        destination_parent_fd=destination_parent_fd,
                        destination_name=target_name,
                        rejected_identity=_entry_identity(published),
                    )
                    raise AtomicDirectoryPublicationError(
                        "published directory identity does not match staging"
                    )
                try:
                    _require_directory_binding(
                        source.parent,
                        source_parent_fd,
                        label="staging parent",
                    )
                    _require_directory_binding(
                        target.parent,
                        destination_parent_fd,
                        label="destination parent",
                    )
                except AtomicDirectoryPublicationError:
                    _restore_rejected_publication(
                        source_parent_fd=source_parent_fd,
                        source_name=source_name,
                        destination_parent_fd=destination_parent_fd,
                        destination_name=target_name,
                        rejected_identity=_entry_identity(published),
                    )
                    raise
            finally:
                os.close(source_fd)
        finally:
            os.close(destination_parent_fd)
    finally:
        os.close(source_parent_fd)


__all__ = [
    "AtomicDirectoryExistsError",
    "AtomicDirectoryPublicationError",
    "publish_directory_no_replace",
]
