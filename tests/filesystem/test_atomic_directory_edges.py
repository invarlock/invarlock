from __future__ import annotations

import ctypes
import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.filesystem import atomic_directory
from invarlock.filesystem.atomic_directory import (
    AtomicDirectoryPublicationError,
    publish_directory_no_replace,
)


def test_relative_directory_publication_uses_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    staging = Path("staging")
    staging.mkdir()
    (staging / "payload").write_text("complete", encoding="utf-8")

    publish_directory_no_replace(staging, Path("published"))

    assert not staging.exists()
    assert Path("published/payload").read_text(encoding="utf-8") == "complete"


def test_publication_rejects_same_source_and_destination(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(AtomicDirectoryPublicationError, match="must name different"):
        publish_directory_no_replace(staging, staging)


def test_publication_rejects_missing_or_non_directory_staging(tmp_path: Path) -> None:
    with pytest.raises(AtomicDirectoryPublicationError, match="staging must be"):
        publish_directory_no_replace(tmp_path / "missing", tmp_path / "published")

    regular_file = tmp_path / "file"
    regular_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(AtomicDirectoryPublicationError, match="staging must be"):
        publish_directory_no_replace(regular_file, tmp_path / "published")


def test_unexpected_kernel_rename_error_is_not_misclassified_as_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    def denied(**_kwargs: object) -> int:
        ctypes.set_errno(errno.EACCES)
        return -1

    monkeypatch.setattr(atomic_directory, "_rename_no_replace", denied)

    with pytest.raises(AtomicDirectoryPublicationError, match="Permission denied"):
        publish_directory_no_replace(staging, tmp_path / "published")
    assert staging.is_dir()


def test_unsupported_platform_fails_without_nonexclusive_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(atomic_directory.platform, "system", lambda: "OtherOS")

    with pytest.raises(AtomicDirectoryPublicationError, match="unsupported"):
        publish_directory_no_replace(staging, tmp_path / "published")
    assert staging.is_dir()


def test_missing_secure_directory_flags_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.delattr(atomic_directory.os, "O_DIRECTORY", raising=False)

    with pytest.raises(AtomicDirectoryPublicationError, match="secure descriptor"):
        publish_directory_no_replace(staging, tmp_path / "published")


def test_low_level_directory_open_rejects_relative_and_parent_traversal() -> None:
    with pytest.raises(AtomicDirectoryPublicationError, match="absolute path"):
        atomic_directory._open_directory(Path("relative"), label="test directory")

    with pytest.raises(AtomicDirectoryPublicationError, match="parent traversal"):
        atomic_directory._open_directory(
            Path("/tmp/parent/../child"), label="test directory"
        )


def test_linux_directory_open_requires_path_only_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_directory.platform, "system", lambda: "Linux")
    monkeypatch.delattr(atomic_directory.os, "O_PATH", raising=False)

    with pytest.raises(AtomicDirectoryPublicationError, match="path-only"):
        atomic_directory._directory_open_flags()


def test_directory_binding_fails_closed_when_the_name_disappears(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "opened"
    directory.mkdir()
    descriptor = atomic_directory._open_directory(directory, label="test directory")
    directory.rmdir()
    try:
        with pytest.raises(AtomicDirectoryPublicationError, match="identity changed"):
            atomic_directory._require_directory_binding(
                directory, descriptor, label="test directory"
            )
    finally:
        atomic_directory.os.close(descriptor)


def test_low_level_directory_open_wraps_anchor_and_component_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = os.open

    def fail_anchor(path: object, _flags: int, **_kwargs: object) -> int:
        raise OSError(errno.ENOENT, f"missing {path}")

    monkeypatch.setattr(atomic_directory.os, "open", fail_anchor)
    with pytest.raises(AtomicDirectoryPublicationError, match="existing non-symlink"):
        atomic_directory._open_directory(tmp_path, label="directory")

    calls = 0

    def fail_component(path: object, flags: int, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError(errno.ELOOP, f"unsafe {path}")
        return real_open(path, flags, **kwargs)

    monkeypatch.setattr(atomic_directory.os, "open", fail_component)
    with pytest.raises(AtomicDirectoryPublicationError, match="existing non-symlink"):
        atomic_directory._open_directory(tmp_path, label="directory")


def test_low_level_directory_open_and_child_reject_non_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_directory, "_directory_open_flags", lambda: 0)
    monkeypatch.setattr(atomic_directory.os, "open", lambda *_args, **_kwargs: 91)
    monkeypatch.setattr(
        atomic_directory.os,
        "fstat",
        lambda _fd: SimpleNamespace(st_mode=stat.S_IFREG | 0o600),
    )
    closed: list[int] = []
    monkeypatch.setattr(atomic_directory.os, "close", closed.append)

    with pytest.raises(AtomicDirectoryPublicationError, match="existing non-symlink"):
        atomic_directory._open_directory(Path("/entry"), label="directory")
    with pytest.raises(AtomicDirectoryPublicationError, match="staging must be"):
        atomic_directory._open_child_directory(5, "entry")
    assert closed == [91, 91, 91]


def test_low_level_child_open_wraps_kernel_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_directory, "_directory_open_flags", lambda: 0)

    def denied(*_args: object, **_kwargs: object) -> int:
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr(atomic_directory.os, "open", denied)
    with pytest.raises(AtomicDirectoryPublicationError, match="staging must be"):
        atomic_directory._open_child_directory(5, "entry")


class _FakeRename:
    def __init__(self, result: int) -> None:
        self.result = result
        self.argtypes: object = None
        self.restype: object = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        return self.result


def test_platform_rename_bindings_and_missing_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = object()
    monkeypatch.setattr(atomic_directory.ctypes, "CDLL", lambda *_a, **_k: missing)
    monkeypatch.setattr(atomic_directory.platform, "system", lambda: "Linux")
    with pytest.raises(AtomicDirectoryPublicationError, match="unavailable on Linux"):
        atomic_directory._rename_no_replace(
            source_parent_fd=1,
            source_name=b"source",
            destination_parent_fd=2,
            destination_name=b"target",
        )

    linux = _FakeRename(7)
    monkeypatch.setattr(
        atomic_directory.ctypes,
        "CDLL",
        lambda *_a, **_k: SimpleNamespace(renameat2=linux),
    )
    assert (
        atomic_directory._rename_no_replace(
            source_parent_fd=1,
            source_name=b"source",
            destination_parent_fd=2,
            destination_name=b"target",
        )
        == 7
    )
    assert linux.calls

    monkeypatch.setattr(atomic_directory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(atomic_directory.ctypes, "CDLL", lambda *_a, **_k: missing)
    with pytest.raises(AtomicDirectoryPublicationError, match="unavailable on macOS"):
        atomic_directory._rename_no_replace(
            source_parent_fd=1,
            source_name=b"source",
            destination_parent_fd=2,
            destination_name=b"target",
        )

    darwin = _FakeRename(0)
    monkeypatch.setattr(
        atomic_directory.ctypes,
        "CDLL",
        lambda *_a, **_k: SimpleNamespace(renameatx_np=darwin),
    )
    assert (
        atomic_directory._rename_no_replace(
            source_parent_fd=1,
            source_name=b"source",
            destination_parent_fd=2,
            destination_name=b"target",
        )
        == 0
    )
    assert darwin.calls


def test_basename_and_directory_binding_identity_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(atomic_directory.os, "altsep", "\\")
    with pytest.raises(AtomicDirectoryPublicationError, match="one directory entry"):
        atomic_directory._validate_basename(Path("bad\\name"), label="destination")

    descriptor = os.open(tmp_path, os.O_RDONLY)
    try:
        monkeypatch.setattr(
            atomic_directory.os,
            "stat",
            lambda *_a, **_k: SimpleNamespace(
                st_mode=stat.S_IFREG | 0o600, st_dev=1, st_ino=2
            ),
        )
        with pytest.raises(AtomicDirectoryPublicationError, match="identity changed"):
            atomic_directory._require_directory_binding(
                tmp_path, descriptor, label="directory"
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("error_number", [errno.EEXIST, errno.ENOTEMPTY])
def test_rename_existing_errors_are_classified(error_number: int) -> None:
    with pytest.raises(atomic_directory.AtomicDirectoryExistsError):
        atomic_directory._raise_rename_error(error_number)


def test_restore_rejected_publication_error_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = (1, 2, stat.S_IFDIR | 0o700)
    kwargs = {
        "source_parent_fd": 1,
        "source_name": "source",
        "destination_parent_fd": 2,
        "destination_name": "target",
        "rejected_identity": identity,
    }
    monkeypatch.setattr(atomic_directory, "_rename_no_replace", lambda **_k: -1)
    monkeypatch.setattr(atomic_directory.ctypes, "get_errno", lambda: errno.EIO)
    with pytest.raises(AtomicDirectoryPublicationError, match="could not be restored"):
        atomic_directory._restore_rejected_publication(**kwargs)

    monkeypatch.setattr(atomic_directory, "_rename_no_replace", lambda **_k: 0)

    def destination_error(name: object, **_kwargs: object) -> object:
        if name == "target":
            raise OSError(errno.EIO, "failure")
        return SimpleNamespace(st_dev=1, st_ino=2, st_mode=identity[2])

    monkeypatch.setattr(atomic_directory.os, "stat", destination_error)
    with pytest.raises(AtomicDirectoryPublicationError, match="verified absent"):
        atomic_directory._restore_rejected_publication(**kwargs)

    monkeypatch.setattr(
        atomic_directory.os,
        "stat",
        lambda *_a, **_k: SimpleNamespace(st_dev=1, st_ino=2, st_mode=identity[2]),
    )
    with pytest.raises(AtomicDirectoryPublicationError, match="remained"):
        atomic_directory._restore_rejected_publication(**kwargs)

    def missing_source(name: object, **_kwargs: object) -> object:
        if name == "target":
            raise FileNotFoundError
        raise OSError(errno.ENOENT, "missing")

    monkeypatch.setattr(atomic_directory.os, "stat", missing_source)
    with pytest.raises(AtomicDirectoryPublicationError, match="staging name"):
        atomic_directory._restore_rejected_publication(**kwargs)

    def wrong_source(name: object, **_kwargs: object) -> object:
        if name == "target":
            raise FileNotFoundError
        return SimpleNamespace(st_dev=9, st_ino=9, st_mode=identity[2])

    monkeypatch.setattr(atomic_directory.os, "stat", wrong_source)
    with pytest.raises(
        AtomicDirectoryPublicationError, match="identity does not match"
    ):
        atomic_directory._restore_rejected_publication(**kwargs)
