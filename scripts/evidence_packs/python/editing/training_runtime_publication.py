"""Atomic publication primitives for training evidence artifacts."""

from __future__ import annotations

import errno
import os
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Any


def publish_directory_no_replace(
    staging: Path,
    output_dir: Path,
    *,
    ctypes_module: Any,
    platform_system: Callable[[], str],
    error_type: type[Exception],
) -> None:
    """Atomically publish ``staging`` without replacing an existing path."""

    source = os.fsencode(staging)
    destination = os.fsencode(output_dir)
    libc = ctypes_module.CDLL(None, use_errno=True)
    system = platform_system()
    if system == "Linux":
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise error_type(
                "atomic no-replace publication is unavailable on this Linux runtime"
            )
        renameat2.argtypes = [
            ctypes_module.c_int,
            ctypes_module.c_char_p,
            ctypes_module.c_int,
            ctypes_module.c_char_p,
            ctypes_module.c_uint,
        ]
        renameat2.restype = ctypes_module.c_int
        result = renameat2(-100, source, -100, destination, 1)
    elif system == "Darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise error_type(
                "atomic no-replace publication is unavailable on this macOS runtime"
            )
        renamex_np.argtypes = [
            ctypes_module.c_char_p,
            ctypes_module.c_char_p,
            ctypes_module.c_uint,
        ]
        renamex_np.restype = ctypes_module.c_int
        result = renamex_np(source, destination, 0x00000004)
    elif os.name == "nt":  # pragma: no cover - Windows runner dependent
        try:
            staging.rename(output_dir)
        except FileExistsError as exc:
            raise error_type(
                f"refusing to replace existing output: {output_dir}"
            ) from exc
        return
    else:  # pragma: no cover - fail closed on unsupported kernels
        raise error_type(
            "atomic no-replace publication is unsupported on this platform"
        )

    if result == 0:
        return
    error_number = ctypes_module.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise error_type(f"refusing to replace existing output: {output_dir}")
    raise error_type(
        "atomic no-replace publication failed: " + os.strerror(error_number)
    )


def fsync_directory(path: Path, *, error_type: type[Exception]) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise error_type(
            f"publication parent directory is unavailable: {path}"
        ) from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise error_type(
            f"publication parent directory could not be synchronized: {path}"
        ) from exc
    finally:
        os.close(descriptor)


def discard_failed_publication(
    output_dir: Path,
    *,
    expected_identity: tuple[int, int],
    publish: Callable[[Path, Path], None],
    fsync: Callable[[Path], None],
    directory_identity: Callable[..., tuple[int, int]],
    error_type: type[Exception],
) -> None:
    """Move a rejected publication away from its consumable output path."""

    quarantine = output_dir.with_name(
        f".{output_dir.name}.rejected-{secrets.token_hex(16)}"
    )
    publish(output_dir, quarantine)
    fsync(output_dir.parent)
    if (
        directory_identity(quarantine, label="rejected training subject")
        != expected_identity
    ):
        raise error_type(
            "rejected training subject identity changed while being quarantined"
        )
