"""Bounded snapshot and no-clobber file helpers for runtime behavior."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from invarlock.evidence_pack_json import (
    StrictJsonError,
    parse_json_bytes,
    read_regular_file_bytes,
)
from invarlock.policy_pack import read_policy_pack_snapshot

from .contracts import MAX_RUNTIME_BEHAVIORAL_SIDE_FILE_BYTES, RuntimeBehaviorError


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeBehaviorError(
            "runtime behavioral evidence must be finite JSON"
        ) from exc


def _open_real_parent(path: Path) -> tuple[Path, int]:
    """Create and retain the exact parent reached without following symlinks."""

    parent = Path(os.path.abspath(os.fspath(path.parent)))
    directory = getattr(os, "O_DIRECTORY", None)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(directory, int) or not isinstance(nofollow, int):
        raise RuntimeBehaviorError("secure output parent traversal is unavailable")
    flags = os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(parent.anchor, flags)
    except OSError as exc:
        raise RuntimeBehaviorError("output parent must be a real directory") from exc
    try:
        for component in parent.parts[1:]:
            if component in {"", "."}:
                continue
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o777, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise RuntimeBehaviorError(
                        "output parent must be a real directory"
                    ) from exc
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise RuntimeBehaviorError(
                        "output parent must be a real directory"
                    ) from exc
            except OSError as exc:
                raise RuntimeBehaviorError(
                    "output parent must be a real directory"
                ) from exc
            os.close(descriptor)
            descriptor = child
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RuntimeBehaviorError("output parent must be a real directory")
    except BaseException:
        os.close(descriptor)
        raise
    return parent, descriptor


def require_real_parent(path: Path) -> Path:
    """Create and validate every parent component without following symlinks."""

    parent, descriptor = _open_real_parent(path)
    os.close(descriptor)
    return parent


def _entry_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode


def _validate_output_basename(path: Path) -> str:
    name = path.name
    if (
        not name
        or name in {".", ".."}
        or "\0" in name
        or os.sep in name
        or os.altsep is not None
        and os.altsep in name
    ):
        raise RuntimeBehaviorError("output must name one file entry")
    return name


def _require_parent_binding(parent: Path, descriptor: int) -> None:
    expected = os.fstat(descriptor)
    try:
        named = os.stat(parent, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeBehaviorError("output parent identity changed") from exc
    if not stat.S_ISDIR(named.st_mode) or _entry_identity(named) != _entry_identity(
        expected
    ):
        raise RuntimeBehaviorError("output parent identity changed")


def _create_temporary(parent_descriptor: int, target_name: str) -> tuple[int, str]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _attempt in range(128):
        name = f".{target_name}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        return descriptor, name
    raise RuntimeBehaviorError("could not allocate a unique temporary output")


def _unlink_if_identity(
    parent_descriptor: int,
    name: str,
    expected_identity: tuple[int, int, int],
) -> bool:
    try:
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    if _entry_identity(current) != expected_identity:
        return False
    os.unlink(name, dir_fd=parent_descriptor)
    return True


def atomic_write_new(path: Path, payload: bytes) -> None:
    output = Path(path)
    target_name = _validate_output_basename(output)
    parent, parent_descriptor = _open_real_parent(output)
    temporary_name: str | None = None
    temporary_identity: tuple[int, int, int] | None = None
    descriptor: int | None = None
    published = False
    try:
        _require_parent_binding(parent, parent_descriptor)
        descriptor, temporary_name = _create_temporary(
            parent_descriptor,
            target_name,
        )
        temporary_identity = _entry_identity(os.fstat(descriptor))
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o600)
        expected = os.fstat(descriptor)
        temporary_identity = _entry_identity(expected)
        named = os.stat(
            temporary_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _entry_identity(named) != temporary_identity:
            raise RuntimeBehaviorError(
                "temporary output identity changed before publication"
            )
        _require_parent_binding(parent, parent_descriptor)
        os.link(
            temporary_name,
            target_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        published = True
        published_stat = os.stat(
            target_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _entry_identity(published_stat) != temporary_identity:
            if not _unlink_if_identity(
                parent_descriptor,
                target_name,
                _entry_identity(published_stat),
            ):
                raise RuntimeBehaviorError(
                    "published output identity changed before rollback"
                )
            published = False
            raise RuntimeBehaviorError(
                "published output identity does not match the temporary file"
            )
        try:
            _require_parent_binding(parent, parent_descriptor)
        except RuntimeBehaviorError:
            if _unlink_if_identity(
                parent_descriptor,
                target_name,
                temporary_identity,
            ):
                published = False
            raise
        if not _unlink_if_identity(
            parent_descriptor,
            temporary_name,
            temporary_identity,
        ):
            if _unlink_if_identity(
                parent_descriptor,
                target_name,
                temporary_identity,
            ):
                published = False
            raise RuntimeBehaviorError(
                "temporary output identity changed after publication"
            )
        temporary_name = None
        published = False
    except FileExistsError as exc:
        raise RuntimeBehaviorError(f"output already exists: {target_name}") from exc
    except OSError as exc:
        raise RuntimeBehaviorError(
            f"could not atomically publish output: {target_name}"
        ) from exc
    finally:
        if published and temporary_identity is not None:
            _unlink_if_identity(
                parent_descriptor,
                target_name,
                temporary_identity,
            )
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None and temporary_identity is not None:
            _unlink_if_identity(
                parent_descriptor,
                temporary_name,
                temporary_identity,
            )
        os.close(parent_descriptor)


def read_json_object(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    try:
        payload = read_regular_file_bytes(
            path,
            label=label,
            max_bytes=MAX_RUNTIME_BEHAVIORAL_SIDE_FILE_BYTES,
        )
        decoded = parse_json_bytes(payload, label=label)
    except StrictJsonError as exc:
        raise RuntimeBehaviorError(str(exc)) from exc
    if not isinstance(decoded, dict):
        raise RuntimeBehaviorError(f"{label} must be a JSON object")
    return payload, cast(dict[str, object], decoded)


def read_policy_pack_bounded(path: Path) -> dict[str, object]:
    try:
        _, decoded = read_policy_pack_snapshot(
            path,
            max_bytes=MAX_RUNTIME_BEHAVIORAL_SIDE_FILE_BYTES,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeBehaviorError(f"policy pack could not be loaded: {exc}") from exc
    return cast(dict[str, object], decoded)


__all__ = ["atomic_write_new", "read_json_object", "read_policy_pack_bounded"]
