from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from invarlock.core import checkpoint_identity as identity


def _write_checkpoint(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")


def _stat_value(*, mode: int, inode: int = 1, size: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        st_dev=1,
        st_ino=inode,
        st_mode=mode,
        st_size=size,
        st_mtime_ns=1,
        st_ctime_ns=1,
    )


class _ScanContext:
    def __init__(self, entries: list[object]) -> None:
        self._entries = entries

    def __enter__(self):  # noqa: ANN204
        return iter(self._entries)

    def __exit__(self, *args: object) -> None:
        del args


class _Entry:
    def __init__(self, name: str, stat_value: object | Exception) -> None:
        self.name = name
        self._stat_value = stat_value

    def stat(self, *, follow_symlinks: bool):  # noqa: ANN201
        assert follow_symlinks is False
        if isinstance(self._stat_value, Exception):
            raise self._stat_value
        return self._stat_value


def test_canonicalizers_and_checkpoint_file_classification() -> None:
    revision = "a" * 40
    digest = "sha256:" + "b" * 64

    assert identity.canonical_remote_revision(None) is None
    assert identity.canonical_remote_revision(f" {revision} ") == revision
    assert identity.canonical_remote_revision("main") is None
    assert identity.canonical_checkpoint_tree_digest(None) is None
    assert identity.canonical_checkpoint_tree_digest(f" {digest} ") == digest
    assert identity.canonical_checkpoint_tree_digest("sha256:bad") is None

    assert identity._checkpoint_file(Path("training_receipt.json")) is False
    assert identity._checkpoint_file(Path(".tmp-model.safetensors")) is False
    assert identity._checkpoint_file(Path("config.json")) is True
    assert identity._checkpoint_file(Path("weights.gguf")) is True
    assert identity._checkpoint_file(Path("model.index.json")) is True
    assert identity._checkpoint_file(Path("random.index.json")) is True
    assert identity._checkpoint_file(Path("backend.lookup")) is True


def test_scan_wraps_scandir_and_entry_stat_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root_fd = os.open(tmp_path, identity._directory_open_flags())
    try:
        monkeypatch.setattr(
            identity.os,
            "scandir",
            lambda _fd: (_ for _ in ()).throw(OSError("scan failed")),
        )
        with pytest.raises(
            identity.CheckpointIdentityError, match="changed while scanning"
        ):
            identity._scan_checkpoint_tree(root_fd)

        monkeypatch.setattr(
            identity.os,
            "scandir",
            lambda _fd: _ScanContext([_Entry("config.json", OSError("gone"))]),
        )
        with pytest.raises(identity.CheckpointIdentityError, match="config.json"):
            identity._scan_checkpoint_tree(root_fd)
    finally:
        os.close(root_fd)


def test_scan_rejects_directory_open_race_identity_change_and_special_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    root_fd = os.open(tmp_path, identity._directory_open_flags())
    original_open = os.open
    try:
        monkeypatch.setattr(
            identity.os,
            "open",
            lambda path, flags, *args, **kwargs: (
                (_ for _ in ()).throw(OSError("blocked"))
                if path == "child"
                else original_open(path, flags, *args, **kwargs)
            ),
        )
        with pytest.raises(identity.CheckpointIdentityError, match="securely opened"):
            identity._scan_checkpoint_tree(root_fd)

        monkeypatch.setattr(identity.os, "open", original_open)
        original_fstat = os.fstat

        def changed_child(fd: int):  # noqa: ANN202
            current = original_fstat(fd)
            if fd != root_fd and stat.S_ISDIR(current.st_mode):
                return _stat_value(mode=current.st_mode, inode=current.st_ino + 1)
            return current

        monkeypatch.setattr(identity.os, "fstat", changed_child)
        with pytest.raises(identity.CheckpointIdentityError, match="directory changed"):
            identity._scan_checkpoint_tree(root_fd)

        monkeypatch.setattr(identity.os, "fstat", original_fstat)
        monkeypatch.setattr(
            identity.os,
            "scandir",
            lambda _fd: _ScanContext(
                [_Entry("pipe", _stat_value(mode=stat.S_IFIFO | 0o600))]
            ),
        )
        with pytest.raises(identity.CheckpointIdentityError, match="non-regular"):
            identity._scan_checkpoint_tree(root_fd)
    finally:
        os.close(root_fd)


def test_scan_rejects_non_directory_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        identity.os,
        "fstat",
        lambda _fd: _stat_value(mode=stat.S_IFREG | 0o600),
    )
    with pytest.raises(identity.CheckpointIdentityError, match="root is not"):
        identity._scan_checkpoint_tree(123)


def test_open_and_hash_wrap_file_races(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_checkpoint(tmp_path)
    root_fd = os.open(tmp_path, identity._directory_open_flags())
    try:
        with pytest.raises(identity.CheckpointIdentityError, match="securely opened"):
            identity._open_checkpoint_file(root_fd, "missing/model.safetensors")

        expected = identity._stat_identity((tmp_path / "config.json").stat())
        with pytest.raises(identity.CheckpointIdentityError, match="before hashing"):
            identity._hash_checkpoint_file(
                hashlib.sha256(),
                root_fd=root_fd,
                relative="config.json",
                expected=replace(expected, size=expected.size + 1),
            )

        original_read = os.read
        monkeypatch.setattr(
            identity.os,
            "read",
            lambda _fd, _size: (_ for _ in ()).throw(OSError("read failed")),
        )
        with pytest.raises(identity.CheckpointIdentityError, match="while hashing"):
            identity._hash_checkpoint_file(
                hashlib.sha256(),
                root_fd=root_fd,
                relative="config.json",
                expected=expected,
            )
        monkeypatch.setattr(identity.os, "read", original_read)

        directory_fd = os.open(tmp_path, identity._directory_open_flags())
        directory_identity = identity._stat_identity(os.fstat(directory_fd))
        monkeypatch.setattr(
            identity, "_open_checkpoint_file", lambda *_a, **_kw: directory_fd
        )
        with pytest.raises(identity.CheckpointIdentityError, match="before hashing"):
            identity._hash_checkpoint_file(
                hashlib.sha256(),
                root_fd=root_fd,
                relative="config.json",
                expected=directory_identity,
            )
    finally:
        os.close(root_fd)


def test_root_match_and_observation_fail_closed_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    root_fd = os.open(checkpoint, identity._directory_open_flags())
    try:
        assert identity._root_path_matches_fd(tmp_path / "missing", root_fd) is False
    finally:
        os.close(root_fd)

    monkeypatch.setattr(identity, "_SECURE_FD_TRAVERSAL_AVAILABLE", False)
    with pytest.raises(identity.CheckpointIdentityError, match="unavailable"):
        identity.checkpoint_tree_observation(checkpoint)
    monkeypatch.setattr(identity, "_SECURE_FD_TRAVERSAL_AVAILABLE", True)

    with pytest.raises(identity.CheckpointIdentityError, match="regular directory"):
        identity.checkpoint_tree_observation(tmp_path / "missing")
    regular_file = tmp_path / "file"
    regular_file.write_text("x", encoding="utf-8")
    with pytest.raises(identity.CheckpointIdentityError, match="regular directory"):
        identity.checkpoint_tree_observation(regular_file)

    original_open = os.open
    monkeypatch.setattr(
        identity.os,
        "open",
        lambda path, flags, *args, **kwargs: (
            (_ for _ in ()).throw(OSError("blocked"))
            if Path(path) == checkpoint
            else original_open(path, flags, *args, **kwargs)
        ),
    )
    with pytest.raises(identity.CheckpointIdentityError, match="securely opened"):
        identity.checkpoint_tree_observation(checkpoint)
    monkeypatch.setattr(identity.os, "open", original_open)

    monkeypatch.setattr(identity, "_root_path_matches_fd", lambda *_args: False)
    with pytest.raises(identity.CheckpointIdentityError, match="root changed"):
        identity.checkpoint_tree_observation(checkpoint)


def test_observation_rejects_snapshot_and_root_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    root_fd = os.open(checkpoint, identity._directory_open_flags())
    try:
        snapshot = identity._scan_checkpoint_tree(root_fd)
    finally:
        os.close(root_fd)
    changed = replace(snapshot, directories=(("changed", snapshot.root),))
    scans = iter((snapshot, changed))
    monkeypatch.setattr(identity, "_scan_checkpoint_tree", lambda _fd: next(scans))
    monkeypatch.setattr(identity, "_root_path_matches_fd", lambda *_args: True)
    with pytest.raises(identity.CheckpointIdentityError, match="tree changed"):
        identity.checkpoint_tree_observation(checkpoint)

    scans = iter((snapshot, snapshot))
    roots = iter((True, False))
    monkeypatch.setattr(identity, "_scan_checkpoint_tree", lambda _fd: next(scans))
    monkeypatch.setattr(identity, "_root_path_matches_fd", lambda *_args: next(roots))
    with pytest.raises(identity.CheckpointIdentityError, match="tree changed"):
        identity.checkpoint_tree_observation(checkpoint)


def test_model_identity_resolution_and_validation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    with pytest.raises(identity.CheckpointIdentityError, match="cannot also declare"):
        identity.resolve_model_identity(
            str(checkpoint), revision="a" * 40, strict=True, side="baseline"
        )
    local_identity = identity.resolve_model_identity(
        str(checkpoint), revision=None, strict=True, side="baseline"
    )
    assert local_identity is not None
    assert local_identity["kind"] == "local_checkpoint_tree"
    assert identity.canonical_checkpoint_tree_digest(local_identity["sha256"])

    assert (
        identity.resolve_model_identity(
            "remote/model", revision=None, strict=False, side="subject"
        )
        is None
    )
    with pytest.raises(identity.CheckpointIdentityError, match="remote model revision"):
        identity.resolve_model_identity(
            "remote/model", revision="main", strict=False, side="subject"
        )
    with pytest.raises(identity.CheckpointIdentityError, match="remote model revision"):
        identity.resolve_model_identity(
            "remote/model", revision="main", strict=True, side="subject"
        )
    with pytest.raises(identity.CheckpointIdentityError, match="remote model revision"):
        identity.resolve_model_identity(
            "remote/model", revision=None, strict=True, side="subject"
        )
    revision = "a" * 40
    assert identity.resolve_model_identity(
        "remote/model", revision=revision, strict=True, side="subject"
    ) == {"kind": "remote_revision", "revision": revision}

    digest = "sha256:" + "b" * 64
    assert identity.validated_model_identity(None) is None
    assert identity.validated_model_identity({"kind": "remote_revision"}) is None
    assert (
        identity.validated_model_identity(
            {"kind": "remote_revision", "revision": "main"}
        )
        is None
    )
    assert identity.validated_model_identity(
        {"kind": "remote_revision", "revision": revision}
    ) == {"kind": "remote_revision", "revision": revision}
    assert (
        identity.validated_model_identity(
            {"kind": "local_checkpoint_tree", "sha256": "bad"}
        )
        is None
    )
    assert identity.validated_model_identity(
        {"kind": "local_checkpoint_tree", "sha256": digest}
    ) == {"kind": "local_checkpoint_tree", "sha256": digest}
    assert (
        identity.validated_model_identity({"kind": "unknown", "sha256": digest}) is None
    )
