from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from invarlock.core.checkpoint_identity import (
    CheckpointIdentityError,
    checkpoint_tree_observation,
    checkpoint_tree_sha256,
)


def _write_checkpoint(root: Path, *, weight: bytes = b"weights-v1") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"gpt2"}\n', encoding="utf-8")
    (root / "model.safetensors").write_bytes(weight)
    (root / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")


def test_checkpoint_tree_digest_is_order_independent_and_mutation_sensitive(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_checkpoint(left)
    right.mkdir()
    (right / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (right / "model.safetensors").write_bytes(b"weights-v1")
    (right / "config.json").write_text('{"model_type":"gpt2"}\n', encoding="utf-8")

    expected = checkpoint_tree_sha256(left)
    assert checkpoint_tree_sha256(right) == expected

    (right / "model.safetensors").write_bytes(b"weights-v2")
    assert checkpoint_tree_sha256(right) != expected


def test_checkpoint_observation_keeps_ephemeral_tree_stat_tokens(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)

    observation = checkpoint_tree_observation(checkpoint)

    assert observation.digest == checkpoint_tree_sha256(checkpoint)
    assert observation.root.device == checkpoint.stat().st_dev
    assert observation.root.inode == checkpoint.stat().st_ino
    assert {relative for relative, _stat in observation.files} == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }


def test_checkpoint_tree_digest_excludes_cache_logs_and_unrelated_sidecars(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    original = checkpoint_tree_sha256(checkpoint)

    (checkpoint / "logs").mkdir()
    (checkpoint / "logs" / "run.log").write_text("first\n", encoding="utf-8")
    (checkpoint / ".cache").mkdir()
    (checkpoint / ".cache" / "download.tmp").write_bytes(b"cache-v1")
    (checkpoint / "training_receipt.json").write_text("{}\n", encoding="utf-8")
    assert checkpoint_tree_sha256(checkpoint) == original

    (checkpoint / "logs" / "run.log").write_text("second\n", encoding="utf-8")
    (checkpoint / ".cache" / "download.tmp").write_bytes(b"cache-v2")
    (checkpoint / "training_receipt.json").write_text(
        '{"changed":true}\n', encoding="utf-8"
    )
    assert checkpoint_tree_sha256(checkpoint) == original


def test_checkpoint_tree_digest_covers_packed_quant_and_adapter_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    (checkpoint / "quantize_config.json").write_text(
        '{"bits":4,"group_size":16}\n', encoding="utf-8"
    )
    adapter = checkpoint / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r":2}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v1")
    original = checkpoint_tree_sha256(checkpoint)

    (checkpoint / "quantize_config.json").write_text(
        '{"bits":8,"group_size":16}\n', encoding="utf-8"
    )
    quant_changed = checkpoint_tree_sha256(checkpoint)
    assert quant_changed != original

    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v2")
    assert checkpoint_tree_sha256(checkpoint) != quant_changed


def test_checkpoint_tree_digest_rejects_symlinks_and_no_checkpoint_files(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CheckpointIdentityError, match="no checkpoint files"):
        checkpoint_tree_sha256(empty)

    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    (checkpoint / "linked-config.json").symlink_to(checkpoint / "config.json")
    with pytest.raises(CheckpointIdentityError, match="symlink"):
        checkpoint_tree_sha256(checkpoint)


def test_checkpoint_tree_digest_rejects_intermediate_parent_symlink(
    tmp_path: Path,
) -> None:
    actual_parent = tmp_path / "actual"
    checkpoint = actual_parent / "checkpoint"
    _write_checkpoint(checkpoint)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(CheckpointIdentityError, match="symbolic links"):
        checkpoint_tree_sha256(linked_parent / checkpoint.name)


def test_checkpoint_tree_digest_rejects_final_component_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    weights = checkpoint / "model.safetensors"
    backup = tmp_path / "original.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(b"replacement")
    original_open = os.open
    swapped = False

    def swapping_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "model.safetensors" and dir_fd is not None and not swapped:
            swapped = True
            weights.replace(backup)
            weights.symlink_to(replacement)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swapping_open)

    with pytest.raises(CheckpointIdentityError, match="changed|symlink|open"):
        checkpoint_tree_sha256(checkpoint)
    assert swapped is True


def test_checkpoint_tree_digest_detects_change_and_revert_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, weight=b"x" * (2 * 1024 * 1024))
    weights = checkpoint / "model.safetensors"
    original_inode = weights.stat().st_ino
    backup = tmp_path / "original.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(b"y" * (2 * 1024 * 1024))
    start = threading.Event()
    reverted = threading.Event()
    original_read = os.read

    def swapping_worker() -> None:
        if not start.wait(timeout=2):
            return
        weights.replace(backup)
        replacement.replace(weights)
        weights.replace(replacement)
        backup.replace(weights)
        reverted.set()

    worker = threading.Thread(target=swapping_worker, daemon=True)
    worker.start()

    def synchronized_read(fd: int, size: int) -> bytes:
        if os.fstat(fd).st_ino == original_inode and not start.is_set():
            start.set()
            assert reverted.wait(timeout=2)
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", synchronized_read)

    with pytest.raises(CheckpointIdentityError, match="changed while hashing"):
        checkpoint_tree_sha256(checkpoint)
    worker.join(timeout=2)
    assert reverted.is_set()
