from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from invarlock.evidence_pack_transformation_contract import _canonical_json_sha256
from invarlock.pruning_contract import (
    PRUNING_ALGORITHM,
    PRUNING_REPLAY_SCHEMA,
    PRUNING_STORAGE_POLICY,
    PRUNING_TARGET_MANIFEST_SCHEMA,
)
from invarlock.pruning_contract import (
    PRUNING_SCOPE_POLICY_VERSION as PRUNING_SCOPE_POLICY,
)
from scripts.evidence_packs.python.editing.implementations import (
    build_validation_edit_metadata,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_checkpoint(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    metadata: dict[str, object] | None = None,
    config: dict[str, object] | None = None,
) -> None:
    path.mkdir(parents=True)
    _write_json(path / "config.json", config or {"model_type": "qwen2"})
    _write_json(path / "tokenizer_config.json", {"model_max_length": 128})
    save_file(tensors, path / "model.safetensors")
    if metadata is not None:
        _write_json(path / "edit_metadata.json", metadata)


def _metadata(
    *,
    scope: str = "ffn",
    target_sparsity: float = 0.5,
    edited_tensors: int = 1,
    edited_params: int = 4,
    target_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = target_manifest or _target_manifest(scope=scope)
    manifest_digest = _canonical_json_sha256(manifest)
    assert manifest_digest is not None
    return build_validation_edit_metadata(
        edit_type="magnitude_prune",
        scope=scope,
        parameters={"target_sparsity": target_sparsity},
        coverage={
            "edited_tensors": edited_tensors,
            "edited_params": edited_params,
            "total_params": 8,
            "coverage_ratio": 0.5,
        },
        extra={
            "target_sparsity": target_sparsity,
            "actual_sparsity": 0.5,
            "effective_changed_params": 2,
            "scope_policy": PRUNING_SCOPE_POLICY,
            "pruning_algorithm": PRUNING_ALGORITHM,
            "storage_policy": PRUNING_STORAGE_POLICY,
            "model_type": "qwen2",
            "pruning_architecture": "decoder",
            "config_sha256": "sha256:" + "c" * 64,
            "target_manifest": manifest,
            "target_manifest_sha256": manifest_digest,
        },
    )


def _target_manifest(*, scope: str = "ffn") -> dict[str, object]:
    return {
        "schema": PRUNING_TARGET_MANIFEST_SCHEMA,
        "scope": scope,
        "scope_policy": PRUNING_SCOPE_POLICY,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": "sha256:" + "c" * 64,
        "targets": [
            {
                "name": "model.layers.0.mlp.up_proj.weight",
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
            }
        ],
    }


def _replay_payload(
    *,
    artifact_identity: dict[str, str],
    baseline_identity: dict[str, str],
    scope: str = "ffn",
    target_sparsity: float = 0.5,
    target_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = target_manifest or _target_manifest(scope=scope)
    manifest_digest = _canonical_json_sha256(manifest)
    assert manifest_digest is not None
    return {
        "schema": PRUNING_REPLAY_SCHEMA,
        "ok": True,
        "edit_type": "magnitude_prune",
        "scope": scope,
        "target_sparsity": target_sparsity,
        "scope_policy": PRUNING_SCOPE_POLICY,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": "sha256:" + "c" * 64,
        "target_manifest": manifest,
        "target_manifest_sha256": manifest_digest,
        "baseline_identity": baseline_identity,
        "artifact_identity": artifact_identity,
        "checked_tensors": 2,
        "selected_tensors": 1,
        "selected_params": 4,
        "total_params": 8,
        "expected_pruned_params": 2,
        "expected_changed_params": 2,
        "observed_changed_params": 2,
        "original_zero_params": 0,
        "observed_zero_params": 2,
        "out_of_scope_tensors_checked": 1,
        "out_of_scope_bytes_checked": 16,
        "support_files_checked": 2,
        "issues": [],
    }


def _pruning_scenario(
    scenario_id: str,
    edit_spec: str,
    *,
    strictness: str = "informational",
) -> dict[str, object]:
    """Build the same closed scenario record the pack dispatcher consumes."""

    return {
        "id": scenario_id,
        "artifact_class": "validation_subject_checkpoint",
        "strictness": strictness,
        "generation": {
            "kind": "edit",
            "edit_spec": edit_spec,
            "version": "clean" if edit_spec.endswith(":clean") else "stress",
        },
    }


def _write_indexed_checkpoint(
    path: Path,
    *,
    shard_name: str,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, object] | None = None,
    config: dict[str, object] | None = None,
) -> None:
    path.mkdir(parents=True)
    _write_json(path / "config.json", config or {"model_type": "qwen2"})
    _write_json(path / "tokenizer_config.json", {"model_max_length": 128})
    _write_json(
        path / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 0},
            "weight_map": dict.fromkeys(tensors, shard_name),
        },
    )
    if metadata is not None:
        _write_json(path / "edit_metadata.json", metadata)


def _self_consistent_pruning_sidecars(
    *,
    manifest: dict[str, object],
    scope: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Build sidecars whose ordinary bindings all agree with a forged manifest."""

    artifact_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "a" * 64,
    }
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "b" * 64,
    }
    payload = _replay_payload(
        artifact_identity=artifact_identity,
        baseline_identity=baseline_identity,
        scope=scope,
        target_manifest=manifest,
    )
    metadata = _metadata(scope=scope, target_manifest=manifest)
    targets = manifest["targets"]
    assert isinstance(targets, list)
    selected_params = sum(
        int(target["numel"])
        for target in targets
        if isinstance(target, dict) and isinstance(target.get("numel"), int)
    )
    selected_tensors = len(targets)
    expected_pruned = selected_params // 2
    for field in ("scope", "model_type", "architecture", "config_sha256"):
        payload[field] = manifest[field]
    metadata["scope"] = manifest["scope"]
    metadata["model_type"] = manifest["model_type"]
    metadata["pruning_architecture"] = manifest["architecture"]
    metadata["config_sha256"] = manifest["config_sha256"]
    manifest_digest = _canonical_json_sha256(manifest)
    assert manifest_digest is not None
    payload["target_manifest_sha256"] = manifest_digest
    metadata["target_manifest_sha256"] = manifest_digest
    payload["selected_tensors"] = selected_tensors
    payload["selected_params"] = selected_params
    payload["total_params"] = selected_params + 4
    payload["checked_tensors"] = selected_tensors + 1
    payload["expected_pruned_params"] = expected_pruned
    payload["expected_changed_params"] = expected_pruned
    payload["observed_changed_params"] = expected_pruned
    payload["observed_zero_params"] = expected_pruned
    metadata["effective_changed_params"] = expected_pruned
    coverage = metadata["coverage"]
    assert isinstance(coverage, dict)
    coverage["edited_tensors"] = selected_tensors
    coverage["edited_params"] = selected_params
    coverage["total_params"] = selected_params + 4
    coverage["coverage_ratio"] = selected_params / (selected_params + 4)
    report = {
        "meta": {"model_identity": artifact_identity},
        "baseline_ref": {"model_identity": baseline_identity},
    }
    return report, metadata, payload


def _make_targets_noncanonical(manifest: dict[str, object]) -> None:
    manifest["targets"] = [
        {
            "name": "model.layers.1.mlp.down_proj.weight",
            "dtype": "torch.float32",
            "shape": [2, 2],
            "numel": 4,
        },
        {
            "name": "model.layers.0.mlp.up_proj.weight",
            "dtype": "torch.float32",
            "shape": [2, 2],
            "numel": 4,
        },
    ]
