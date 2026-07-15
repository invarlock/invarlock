from __future__ import annotations

import hashlib
import json
from pathlib import Path

from invarlock.evidence_pack_edit_common import (
    RUNTIME_RELOAD_PROOF_SCHEMA,
    RUNTIME_RELOAD_PROOF_SIDECAR,
    TRANSFORMATION_CONTRACT_VERSION,
    TRANSFORMATION_EXECUTION_POLICY,
    TRANSFORMATION_MATERIALIZATION_RECEIPT,
    TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA,
    TRANSFORMATION_REPLAY_SCHEMA,
    TRANSFORMATION_REPLAY_SIDECAR,
    TRANSFORMATION_SCOPE_POLICY,
    TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
)
from invarlock.evidence_pack_transformation_contract import _canonical_json_sha256
from scripts.evidence_packs.python.editing.attach_transformation_selection_receipt import (
    attach_transformation_selection_receipt,
)
from scripts.evidence_packs.python.editing.implementations import (
    build_validation_edit_metadata,
)
from scripts.evidence_packs.python.editing.transformation_contract import (
    canonical_transformation_spec,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: object) -> str:
    digest = _canonical_json_sha256(payload)
    assert digest is not None
    return digest


def _runtime_reload_proof(replay: dict[str, object]) -> dict[str, object]:
    artifact_identity = replay["artifact_identity"]
    return {
        "schema": RUNTIME_RELOAD_PROOF_SCHEMA,
        "ok": True,
        "replay_schema": replay["schema"],
        "edit_type": replay["edit_type"],
        "artifact_identity": artifact_identity,
        "replay_artifact_identity": artifact_identity,
        "prompt_sha256": "sha256:"
        + hashlib.sha256(
            b"InvarLock verifier-grade transformation runtime proof."
        ).hexdigest(),
        "device": "cpu",
        "input_device": "cpu",
        "reload_runs": 2,
        "token_ids_sha256": "sha256:" + "2" * 64,
        "token_ids_shape": [1, 4],
        "logits_sha256": "sha256:" + "3" * 64,
        "logits_shape": [1, 4, 8],
        "all_logits_finite": True,
        "repeat_deterministic": True,
        "load_diagnostics": {
            "schema": "invarlock/pretrained-load-diagnostics-v1",
            "reloads": [
                {
                    "unexpected_keys": [],
                    "missing_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                },
                {
                    "unexpected_keys": [],
                    "missing_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                },
            ],
        },
        "storage_key_audit": {
            "schema": "invarlock/safetensors-storage-key-audit-v1",
            "reloads": [
                {
                    "artifact_storage_key_count": 1,
                    "artifact_storage_keys_sha256": "sha256:" + "4" * 64,
                    "model_state_key_count": 2,
                    "model_state_keys_sha256": "sha256:" + "5" * 64,
                    "unexpected_storage_keys": [],
                },
                {
                    "artifact_storage_key_count": 1,
                    "artifact_storage_keys_sha256": "sha256:" + "4" * 64,
                    "model_state_key_count": 2,
                    "model_state_keys_sha256": "sha256:" + "5" * 64,
                    "unexpected_storage_keys": [],
                },
            ],
        },
    }


_IDENTITY_A = {"kind": "local_checkpoint_tree", "sha256": "sha256:" + "a" * 64}
_IDENTITY_B = {"kind": "local_checkpoint_tree", "sha256": "sha256:" + "b" * 64}
_IDENTITY_C = {"kind": "local_checkpoint_tree", "sha256": "sha256:" + "c" * 64}
_CONFIG_DIGEST = "sha256:" + "c" * 64
_LAYER_COUNT = 2
_SOURCE_DIGEST = "sha256:" + "d" * 64
_INDEX_DIGEST = "sha256:" + "e" * 64
_OUTPUT_DIGEST = "sha256:" + "f" * 64
_TARGET_NAME = "model.layers.0.mlp.up_proj.weight"
_OUT_OF_SCOPE_NAME = "model.layers.0.self_attn.q_proj.weight"


def _candidate_target_manifest(
    *, transformation: dict[str, object], scope: str
) -> dict[str, object]:
    role, name = {
        "ffn": ("ffn", _TARGET_NAME),
        "attn": ("attn", _OUT_OF_SCOPE_NAME),
    }[scope]
    return {
        "schema": TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "edit_type": transformation["edit_type"],
        "algorithm": transformation["algorithm"],
        "parameters": transformation["parameters"],
        "scope": scope,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": _CONFIG_DIGEST,
        "layer_count": _LAYER_COUNT,
        "targets": [
            {
                "name": name,
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
                "role": role,
                "layer": 0,
            }
        ],
    }


def _parameters(edit_type: str) -> dict[str, object]:
    values: dict[str, dict[str, object]] = {
        "quant_rtn": {"bits": 4, "group_size": 2},
        "synthetic_lowrank_delta": {"rank": 2, "scale": 2.0},
        "synthetic_dense_update": {"step_size": 0.001, "iterations": 2},
    }
    return dict(values[edit_type])


def _stress_edit_spec(edit_type: str) -> str:
    values = {
        "quant_rtn": "quant_rtn:4:2:ffn",
        "synthetic_lowrank_delta": "synthetic_lowrank_delta:2:2:ffn",
        "synthetic_dense_update": "synthetic_dense_update:0.001:2:ffn",
    }
    return values[edit_type]


def _synthetic_provenance(
    *, edit_type: str, transformation: dict[str, object]
) -> dict[str, object] | None:
    if edit_type == "synthetic_lowrank_delta":
        return {
            "edit_family": edit_type,
            "edit_method": transformation["algorithm"],
            "edit_count": 1,
            "dynamic_runtime_required": False,
            "synthetic": True,
            "trained_adapter": False,
            "adapter_merge_performed": False,
        }
    if edit_type == "synthetic_dense_update":
        parameters = transformation["parameters"]
        assert isinstance(parameters, dict)
        return {
            "edit_family": edit_type,
            "edit_method": transformation["algorithm"],
            "edit_count": parameters["iterations"],
            "dynamic_runtime_required": False,
            "synthetic": True,
            "optimization_performed": False,
            "training_data_used": False,
        }
    return None


def _alternative_parameters(edit_type: str) -> dict[str, object]:
    """Return a semantically distinct candidate for selection fixtures."""

    values: dict[str, dict[str, object]] = {
        "quant_rtn": {"bits": 8, "group_size": 2},
        "synthetic_lowrank_delta": {"rank": 1, "scale": 1.0},
        "synthetic_dense_update": {"step_size": 0.002, "iterations": 1},
    }
    return dict(values[edit_type])


def _make_pack(
    tmp_path: Path,
    *,
    edit_type: str = "quant_rtn",
    clean: bool = False,
    scenario_id: str | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    scenario_id = scenario_id or f"{edit_type}_{'clean' if clean else 'stress'}"
    parameters = _parameters(edit_type)
    transformation = canonical_transformation_spec(edit_type, parameters)
    assert isinstance(transformation, dict)
    canonical_parameters = transformation["parameters"]
    assert isinstance(canonical_parameters, dict)
    scope = "ffn"
    target_manifest = {
        "schema": TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "edit_type": edit_type,
        "algorithm": transformation["algorithm"],
        "parameters": canonical_parameters,
        "scope": scope,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": _CONFIG_DIGEST,
        "layer_count": _LAYER_COUNT,
        "targets": [
            {
                "name": _TARGET_NAME,
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
                "role": "ffn",
                "layer": 0,
            }
        ],
    }
    target_digest = _digest(target_manifest)
    source_plan = {
        "source_shards": [
            {
                "path": "model.safetensors",
                "sha256": _SOURCE_DIGEST,
                "tensor_names": [_TARGET_NAME, _OUT_OF_SCOPE_NAME],
                "byte_count": 32,
            }
        ]
    }
    source_plan_digest = _digest(source_plan)
    output_plan = {
        "source_shard_plan_sha256": source_plan_digest,
        "target_manifest_sha256": target_digest,
        "chunks": [
            {
                "name": "model-00001-of-00001.safetensors",
                "source_path": "model.safetensors",
                "source_sha256": _SOURCE_DIGEST,
                "tensor_names": [_TARGET_NAME, _OUT_OF_SCOPE_NAME],
                "byte_count": 32,
            }
        ],
    }
    output_plan_digest = _digest(output_plan)
    output_weights = {
        "index_sha256": _INDEX_DIGEST,
        "shards": [
            {
                "name": "model-00001-of-00001.safetensors",
                "sha256": _OUTPUT_DIGEST,
            }
        ],
    }
    output_weights = {
        "sha256": _digest(output_weights),
        **output_weights,
    }
    actual_changes = {
        "value_changed_tensors": 1,
        "value_changed_params": 4,
        "byte_changed_tensors": 1,
        "byte_changed_params": 4,
    }
    pack = tmp_path / "pack"
    report_dir = pack / "reports" / "org__model" / scenario_id / "run_1"
    report_dir.mkdir(parents=True)
    edit_spec = f"{edit_type}:clean" if clean else _stress_edit_spec(edit_type)
    _write_json(
        pack / "metadata" / "scenarios.json",
        {
            "scenarios": [
                {
                    "id": scenario_id,
                    "artifact_class": "validation_subject_checkpoint",
                    "strictness": "informational",
                    "generation": {
                        "kind": "edit",
                        "edit_spec": edit_spec,
                        "version": "clean" if clean else "stress",
                    },
                }
            ]
        },
    )
    _write_json(
        report_dir / "evaluation.report.json",
        {
            "meta": {"model_identity": _IDENTITY_A},
            "baseline_ref": {"model_identity": _IDENTITY_B},
        },
    )
    metadata = build_validation_edit_metadata(
        edit_type=edit_type,
        scope=scope,
        parameters=canonical_parameters,
        coverage={
            "edited_tensors": 1,
            "edited_params": 4,
            "total_params": 8,
            "coverage_ratio": 0.5,
        },
        edit_provenance=_synthetic_provenance(
            edit_type=edit_type, transformation=transformation
        ),
        extra={
            "transformation_contract": transformation,
            "scope_policy": TRANSFORMATION_SCOPE_POLICY,
            "model_type": "qwen2",
            "transformation_architecture": "decoder",
            "config_sha256": _CONFIG_DIGEST,
            "layer_count": _LAYER_COUNT,
            "target_manifest": target_manifest,
            "target_manifest_sha256": target_digest,
            "max_output_shard_bytes": 1024 * 1024,
            "source_shard_plan": source_plan,
            "source_shard_plan_sha256": source_plan_digest,
            "output_shard_plan": output_plan,
            "output_shard_plan_sha256": output_plan_digest,
            "selected_tensors": 1,
            "selected_params": 4,
            "actual_changes": actual_changes,
            "materialization": "resumable_bounded_safetensors_v1",
            "execution_policy": TRANSFORMATION_EXECUTION_POLICY,
        },
    )
    _write_json(report_dir / "edit_metadata.json", metadata)
    receipt = {
        "schema": TRANSFORMATION_MATERIALIZATION_RECEIPT_SCHEMA,
        "ok": True,
        "baseline_identity": _IDENTITY_B,
        "transformation": transformation,
        "scope": scope,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": _CONFIG_DIGEST,
        "layer_count": _LAYER_COUNT,
        "target_manifest": target_manifest,
        "target_manifest_sha256": target_digest,
        "max_output_shard_bytes": 1024 * 1024,
        "source_shard_plan": source_plan,
        "source_shard_plan_sha256": source_plan_digest,
        "output_shard_plan": output_plan,
        "output_shard_plan_sha256": output_plan_digest,
        "output_weights": output_weights,
        "execution_policy": TRANSFORMATION_EXECUTION_POLICY,
        "output_shards": 1,
        "resume_count": 0,
        "selected_tensors": 1,
        "selected_params": 4,
        "out_of_scope_tensors": 1,
        "out_of_scope_params": 4,
        "total_tensors": 2,
        "total_params": 8,
        "actual_changes": actual_changes,
    }
    receipt_path = report_dir / TRANSFORMATION_MATERIALIZATION_RECEIPT
    _write_json(receipt_path, receipt)
    replay: dict[str, object] = {
        "schema": TRANSFORMATION_REPLAY_SCHEMA,
        "ok": True,
        "edit_type": edit_type,
        "transformation": transformation,
        "algorithm": transformation["algorithm"],
        "parameters": canonical_parameters,
        "scope": scope,
        "scope_policy": TRANSFORMATION_SCOPE_POLICY,
        "model_type": "qwen2",
        "architecture": "decoder",
        "config_sha256": _CONFIG_DIGEST,
        "layer_count": _LAYER_COUNT,
        "target_manifest": target_manifest,
        "target_manifest_sha256": target_digest,
        "baseline_identity": _IDENTITY_B,
        "artifact_identity": _IDENTITY_A,
        "materialization_receipt_sha256": _sha256_file(receipt_path),
        "edit_metadata_sha256": _sha256_file(report_dir / "edit_metadata.json"),
        "source_shard_plan": source_plan,
        "source_shard_plan_sha256": source_plan_digest,
        "output_shard_plan": output_plan,
        "output_shard_plan_sha256": output_plan_digest,
        "max_output_shard_bytes": 1024 * 1024,
        "output_weights": output_weights,
        "execution_policy": TRANSFORMATION_EXECUTION_POLICY,
        "checked_tensors": 2,
        "selected_tensors": 1,
        "selected_params": 4,
        "total_tensors": 2,
        "total_params": 8,
        "actual_changes": actual_changes,
        "out_of_scope_tensors_checked": 1,
        "out_of_scope_bytes_checked": 16,
        "support_files_checked": 2,
        "issues": [],
    }
    selection_bundle_path: Path | None = None
    if clean:
        from tests.evidence_packs._support_transformation_clean_selection import (
            _write_clean_selection_bundle,
        )

        selection_bundle_path = _write_clean_selection_bundle(
            pack=pack,
            edit_type=edit_type,
            parameters=canonical_parameters,
            transformation=transformation,
            scope=scope,
        )
    replay_path = report_dir / TRANSFORMATION_REPLAY_SIDECAR
    _write_json(replay_path, replay)
    if selection_bundle_path is not None:
        attach_transformation_selection_receipt(
            replay_path=replay_path,
            selection_bundle_path=selection_bundle_path,
            scenario_id=scenario_id,
            model_key="org/model",
            edit_type=edit_type,
            parameters=canonical_parameters,
            scope=scope,
        )
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
    _write_json(
        report_dir / RUNTIME_RELOAD_PROOF_SIDECAR, _runtime_reload_proof(replay)
    )
    return pack, report_dir, replay


def _rewrite_fully_crosslinked_target_manifest(
    report_dir: Path, *, target_name: str
) -> None:
    """Forge every retained cross-link while preserving manifest semantics.

    This fixture models the historical fake-green path: a generator could
    select a visual tensor, then make its replay, metadata, receipt, and shard
    plans all agree.  The package target-policy oracle must still reject it.
    """

    replay_path = report_dir / TRANSFORMATION_REPLAY_SIDECAR
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    target_manifest = replay["target_manifest"]
    assert isinstance(target_manifest, dict)
    targets = target_manifest["targets"]
    assert isinstance(targets, list) and len(targets) == 1
    target = targets[0]
    assert isinstance(target, dict)
    original_name = target["name"]
    assert isinstance(original_name, str)
    target["name"] = target_name
    target_digest = _digest(target_manifest)

    source_plan = replay["source_shard_plan"]
    assert isinstance(source_plan, dict)
    source_shards = source_plan["source_shards"]
    assert isinstance(source_shards, list)
    for shard in source_shards:
        assert isinstance(shard, dict)
        names = shard["tensor_names"]
        assert isinstance(names, list)
        shard["tensor_names"] = sorted(
            target_name if name == original_name else name for name in names
        )
    source_digest = _digest(source_plan)

    output_plan = replay["output_shard_plan"]
    assert isinstance(output_plan, dict)
    output_plan["source_shard_plan_sha256"] = source_digest
    output_plan["target_manifest_sha256"] = target_digest
    chunks = output_plan["chunks"]
    assert isinstance(chunks, list)
    for chunk in chunks:
        assert isinstance(chunk, dict)
        names = chunk["tensor_names"]
        assert isinstance(names, list)
        chunk["tensor_names"] = sorted(
            target_name if name == original_name else name for name in names
        )
    output_digest = _digest(output_plan)

    for path in (
        report_dir / "edit_metadata.json",
        report_dir / TRANSFORMATION_MATERIALIZATION_RECEIPT,
    ):
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        sidecar["target_manifest"] = target_manifest
        sidecar["target_manifest_sha256"] = target_digest
        sidecar["source_shard_plan"] = source_plan
        sidecar["source_shard_plan_sha256"] = source_digest
        sidecar["output_shard_plan"] = output_plan
        sidecar["output_shard_plan_sha256"] = output_digest
        _write_json(path, sidecar)

    replay["target_manifest"] = target_manifest
    replay["target_manifest_sha256"] = target_digest
    replay["source_shard_plan"] = source_plan
    replay["source_shard_plan_sha256"] = source_digest
    replay["output_shard_plan"] = output_plan
    replay["output_shard_plan_sha256"] = output_digest
    replay["edit_metadata_sha256"] = _sha256_file(report_dir / "edit_metadata.json")
    replay["materialization_receipt_sha256"] = _sha256_file(
        report_dir / TRANSFORMATION_MATERIALIZATION_RECEIPT
    )
    _write_json(replay_path, replay)
