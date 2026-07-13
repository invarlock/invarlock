from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import invarlock.training_state_evidence as state_evidence
from invarlock.training_state_evidence import (
    TrainingStateEvidenceError,
    state_manifest,
    state_manifest_sha256,
    streaming_lora_delta_evidence,
    tensor_content_sha256,
    tensor_state_sha256,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_chunked_content_digest_matches_exact_bytes(dtype: torch.dtype) -> None:
    tensor = torch.arange(33, dtype=torch.float32).to(dtype).reshape(3, 11)
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    expected = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert tensor_content_sha256(tensor, torch=torch) == expected


def test_noncontiguous_state_fails_closed() -> None:
    tensor = torch.arange(33, dtype=torch.float32).reshape(3, 11).T
    with pytest.raises(TrainingStateEvidenceError, match="requires contiguous"):
        tensor_state_sha256({"weight": tensor}, torch=torch)


def test_manifest_digest_changes_with_tensor_content() -> None:
    state = {"weight": torch.arange(17, dtype=torch.float32)}
    before = state_manifest(state, torch=torch)
    state["weight"][3] += 1
    after = state_manifest(state, torch=torch)
    assert before != after
    assert state_manifest_sha256(before) != state_manifest_sha256(after)


def test_streaming_delta_binds_exact_scope_and_rejects_out_of_scope() -> None:
    before = {
        "target": torch.arange(29, dtype=torch.float32),
        "frozen": torch.arange(17, dtype=torch.float32),
    }
    manifest = state_manifest(before, torch=torch)
    after = {name: tensor.clone() for name, tensor in before.items()}
    after["target"][5:9] += 0.25
    digest, count, maximum, changed = streaming_lora_delta_evidence(
        baseline_manifest=manifest,
        baseline_targets={"target": before["target"]},
        after=after,
        torch=torch,
    )
    assert digest.startswith("sha256:")
    assert count == 1
    assert maximum == 0.25
    assert changed == {"target"}

    after["frozen"][0] += 1
    with pytest.raises(TrainingStateEvidenceError, match="out-of-scope"):
        streaming_lora_delta_evidence(
            baseline_manifest=manifest,
            baseline_targets={"target": before["target"]},
            after=after,
            torch=torch,
        )


def test_directory_hash_rejects_file_mutation_during_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    target = artifact / "model.bin"
    target.write_bytes(b"a" * (2 * 1024 * 1024))
    original_read = state_evidence.os.read
    mutated = False

    def mutate_after_read(fd: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(fd, size)
        if chunk and not mutated:
            mutated = True
            target.write_bytes(b"b" * (2 * 1024 * 1024))
        return chunk

    monkeypatch.setattr(state_evidence.os, "read", mutate_after_read)
    with pytest.raises(TrainingStateEvidenceError, match="changed during hashing"):
        state_evidence.directory_sha256(artifact)


def test_directory_hash_rejects_path_replacement_during_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    target = artifact / "model.bin"
    replacement = artifact / "replacement.bin"
    target.write_bytes(b"a" * (2 * 1024 * 1024))
    replacement.write_bytes(b"b" * (2 * 1024 * 1024))
    original_read = state_evidence.os.read
    replaced = False

    def replace_after_read(fd: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(fd, size)
        if chunk and not replaced:
            replaced = True
            replacement.replace(target)
        return chunk

    monkeypatch.setattr(state_evidence.os, "read", replace_after_read)
    with pytest.raises(TrainingStateEvidenceError, match="changed during hashing"):
        state_evidence.directory_sha256(artifact)
