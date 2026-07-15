from __future__ import annotations

import os
from pathlib import Path

import pytest

from invarlock.runtime_behavior import io as runtime_behavior_io
from invarlock.runtime_behavior.contracts import RuntimeBehaviorError
from invarlock.runtime_behavior.io import atomic_write_new, require_real_parent


def test_require_real_parent_returns_normalized_absolute_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    parent = require_real_parent(Path("discarded/../evidence/side"))

    assert parent == tmp_path / "evidence"
    assert parent.is_absolute()
    assert parent.is_dir()


def test_require_real_parent_rejects_nested_symlink_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    nested = real_parent / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(RuntimeBehaviorError, match="real directory"):
        require_real_parent(alias / "nested" / "side")


def test_atomic_write_new_rejects_nested_symlink_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    nested = real_parent / "nested"
    nested.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(RuntimeBehaviorError, match="real directory"):
        atomic_write_new(alias / "nested" / "receipt.json", b"new")

    assert not (nested / "receipt.json").exists()


def test_atomic_write_new_preserves_no_clobber_after_parent_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("evidence/../evidence/receipt.json")

    atomic_write_new(path, b"first")
    with pytest.raises(RuntimeBehaviorError, match="already exists"):
        atomic_write_new(path, b"second")

    assert (tmp_path / "evidence" / "receipt.json").read_bytes() == b"first"


def test_atomic_write_new_removes_replaced_temporary_from_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "receipt.json"
    trusted: Path | None = None
    real_link = runtime_behavior_io.os.link

    def replace_temporary_then_link(
        source: str | bytes,
        target: str | bytes,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal trusted
        assert isinstance(source, str)
        assert src_dir_fd is not None
        assert dst_dir_fd is not None
        trusted_name = source + ".trusted"
        trusted = tmp_path / trusted_name
        os.rename(
            source,
            trusted_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=src_dir_fd,
        )
        replacement = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=src_dir_fd,
        )
        try:
            os.write(replacement, b"attacker")
        finally:
            os.close(replacement)
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(runtime_behavior_io.os, "link", replace_temporary_then_link)

    with pytest.raises(RuntimeBehaviorError, match="does not match"):
        atomic_write_new(destination, b"trusted")

    assert not destination.exists()
    assert trusted is not None
    assert trusted.read_bytes() == b"trusted"


def test_atomic_write_new_rolls_back_when_parent_is_swapped_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "evidence"
    parent.mkdir()
    original_parent = tmp_path / "opened-evidence-parent"
    destination = parent / "receipt.json"
    real_link = runtime_behavior_io.os.link

    def swap_parent_then_link(
        source: str | bytes,
        target: str | bytes,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        parent.rename(original_parent)
        parent.mkdir()
        (parent / "attacker.txt").write_text("replacement", encoding="utf-8")
        real_link(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(runtime_behavior_io.os, "link", swap_parent_then_link)

    with pytest.raises(RuntimeBehaviorError, match="parent identity changed"):
        atomic_write_new(destination, b"trusted")

    assert not destination.exists()
    assert not (original_parent / "receipt.json").exists()
    assert (parent / "attacker.txt").read_text(encoding="utf-8") == "replacement"


@pytest.mark.parametrize("path", [Path("."), Path(".."), Path("/")])
def test_atomic_write_new_rejects_invalid_output_basename(path: Path) -> None:
    with pytest.raises(RuntimeBehaviorError, match="one file entry"):
        atomic_write_new(path, b"payload")
