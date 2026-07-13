from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from scripts.evidence_packs.python.editing import (
    streaming_transform,
    streaming_transform_core,
)
from scripts.evidence_packs.python.editing import validate_artifact as validator_module
from scripts.evidence_packs.python.editing.artifact_tensor_validation import (
    _canonical_json_sha256,
)
from scripts.evidence_packs.python.editing.streaming_transform import (
    TRANSFORMATION_MATERIALIZATION_RECEIPT,
    materialize_transformation_artifact,
)
from scripts.evidence_packs.python.editing.transformation_oracle import (
    TRANSFORMATION_CONTRACT_VERSION,
    TRANSFORMATION_REPLAY_SCHEMA,
    TRANSFORMATION_SCOPE_POLICY_VERSION,
    TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
    build_transformation_oracle,
)
from scripts.evidence_packs.python.editing.validate_artifact import (
    main,
    validate_transformation_artifact,
)
from scripts.evidence_packs.python.editing.validate_transformation import (
    _output_weight_identity,
)
from tests.evidence_packs._support_transformation_replay import (
    _artifact_tensors,
    _baseline_tensors,
    _materialize,
    _rewrite_artifact_tensors,
    _write_baseline,
    _write_json,
)


@pytest.mark.parametrize(
    ("edit_type", "parameters"),
    [
        ("quant_rtn", {"bits": 4, "group_size": 2}),
        ("synthetic_lowrank_delta", {"rank": 2, "scale": 2.0}),
        ("synthetic_dense_update", {"step_size": 0.001, "iterations": 2}),
    ],
)
def test_replay_accepts_exact_streaming_transform_and_binds_identities(
    tmp_path: Path,
    edit_type: str,
    parameters: dict[str, object],
) -> None:
    baseline, artifact, _ = _materialize(
        tmp_path,
        edit_type=edit_type,
        parameters=parameters,
    )

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type=edit_type,
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is True, result["issues"]
    assert result["schema"] == TRANSFORMATION_REPLAY_SCHEMA
    assert result["transformation"]["parameters"] == parameters
    assert result["selected_tensors"] == 1
    assert result["actual_changes"]["value_changed_params"] > 0
    assert result["actual_changes"]["byte_changed_params"] > 0
    assert result["max_output_shard_bytes"] >= 1024 * 1024
    assert result["source_shard_plan"]["source_shards"]
    assert result["output_shard_plan"]["chunks"]
    assert result["baseline_identity"] == {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(baseline),
    }
    assert result["artifact_identity"] == {
        "kind": "local_checkpoint_tree",
        "sha256": checkpoint_tree_sha256(artifact),
    }


def test_replay_accepts_bfloat16_rank_six_across_multiple_row_chunks(
    tmp_path: Path,
) -> None:
    """Exercise a size that exposed the former GEMM-versus-oracle mismatch."""

    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    baseline.mkdir()
    _write_json(
        baseline / "config.json", {"model_type": "qwen2", "num_hidden_layers": 1}
    )
    _write_json(baseline / "tokenizer_config.json", {"model_max_length": 128})
    target = (
        torch.linspace(-1.0, 1.0, 1024 * 128, dtype=torch.float32)
        .reshape(1024, 128)
        .to(torch.bfloat16)
    )
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": target,
            "model.layers.0.self_attn.q_proj.weight": torch.eye(
                4, dtype=torch.bfloat16
            ),
        },
        baseline / "model.safetensors",
        metadata={"format": "pt"},
    )
    parameters = {"rank": 6, "scale": 3.141592653589793}
    materialize_transformation_artifact(
        baseline_path=baseline,
        output_path=artifact,
        edit_type="synthetic_lowrank_delta",
        parameters=parameters,
        scope="ffn",
    )

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="synthetic_lowrank_delta",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is True, result["issues"]
    tensors, _ = _artifact_tensors(artifact)
    output = tensors["model.layers.0.mlp.up_proj.weight"]
    assert (
        hashlib.sha256(
            output.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        == "846aadb9163dd37d5d378079786253aefeacad53525ee6d10efc81bd692f047e"
    )


def test_materializer_rejects_injected_target_beyond_configured_layer_count(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)
    tensors = _baseline_tensors(baseline)
    tensors["model.layers.999.mlp.up_proj.weight"] = torch.ones(
        (2, 2), dtype=torch.float32
    )
    save_file(tensors, baseline / "model.safetensors", metadata={"format": "pt"})

    with pytest.raises(
        streaming_transform.TransformationContractError,
        match="declared layer count",
    ):
        materialize_transformation_artifact(
            baseline_path=baseline,
            output_path=artifact,
            edit_type="quant_rtn",
            parameters={"bits": 4, "group_size": 2},
            scope="ffn@layer=999",
        )
    assert not artifact.exists()


def test_replay_rejects_copied_baseline_even_with_original_receipts(
    tmp_path: Path,
) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    _rewrite_artifact_tensors(artifact, _baseline_tensors(baseline))

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any("exact transformation replay" in issue for issue in result["issues"])
    assert any(
        "no effective value and byte changes" in issue for issue in result["issues"]
    )


def test_replay_rejects_a_self_consistent_generator_math_defect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)
    original = streaming_transform.replay_transformation_tensor

    def corrupt_generator_math(
        tensor: torch.Tensor,
        *,
        edit_type: str,
        parameters: dict[str, object],
    ) -> torch.Tensor:
        return original(tensor, edit_type=edit_type, parameters=parameters).add(
            torch.tensor(0.125, dtype=tensor.dtype)
        )

    monkeypatch.setattr(
        streaming_transform, "replay_transformation_tensor", corrupt_generator_math
    )
    parameters = {"bits": 4, "group_size": 2}
    materialize_transformation_artifact(
        baseline_path=baseline,
        output_path=artifact,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any(
        "artifact does not match exact transformation replay" in issue
        for issue in result["issues"]
    )


@pytest.mark.parametrize(
    ("edit_type", "parameters"),
    [
        ("quant_rtn", {"bits": 4, "group_size": 2}),
        ("synthetic_lowrank_delta", {"rank": 2, "scale": 2.0}),
        ("synthetic_dense_update", {"step_size": 0.001, "iterations": 2}),
    ],
)
def test_replay_accepts_artifacts_numerically_computed_by_the_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit_type: str,
    parameters: dict[str, object],
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)
    oracle = build_transformation_oracle(
        baseline,
        edit_type=edit_type,
        parameters=parameters,
        scope="ffn",
    )

    def external_reference_math(
        tensor: torch.Tensor,
        *,
        edit_type: str,
        parameters: dict[str, object],
    ) -> torch.Tensor:
        assert edit_type == oracle.spec["edit_type"]
        assert parameters == oracle.spec["parameters"]
        return oracle.replay_tensor(tensor)

    monkeypatch.setattr(
        streaming_transform, "replay_transformation_tensor", external_reference_math
    )
    materialize_transformation_artifact(
        baseline_path=baseline,
        output_path=artifact,
        edit_type=edit_type,
        parameters=parameters,
        scope="ffn",
    )

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type=edit_type,
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is True, result["issues"]


def test_replay_rejects_a_generator_selected_vision_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    _write_baseline(baseline)
    parameters = {"bits": 4, "group_size": 2}
    original_target_check = streaming_transform_core.is_transformation_target

    def poisoned_target_check(
        name: str,
        *,
        scope: str,
        contract: object,
        ndim: int,
    ) -> bool:
        return (
            name == "model.visual.blocks.0.mlp.up_proj.weight"
            or original_target_check(name, scope=scope, contract=contract, ndim=ndim)
        )

    def forged_target_manifest(
        *,
        edit_type: str,
        parameters: dict[str, object],
        scope: str,
        contract: object,
        targets: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "schema": TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
            "contract_version": TRANSFORMATION_CONTRACT_VERSION,
            "scope_policy": TRANSFORMATION_SCOPE_POLICY_VERSION,
            "edit_type": edit_type,
            "algorithm": "groupwise_rtn_dequantized_per_row_v1",
            "parameters": parameters,
            "scope": scope,
            "model_type": contract.model_type,
            "architecture": contract.architecture,
            "config_sha256": contract.config_sha256,
            "targets": [{**target, "role": "ffn", "layer": 0} for target in targets],
        }

    def canonical_digest(payload: object) -> str:
        encoded = json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    monkeypatch.setattr(
        streaming_transform_core, "is_transformation_target", poisoned_target_check
    )
    monkeypatch.setattr(
        streaming_transform_core,
        "transformation_target_manifest",
        forged_target_manifest,
    )
    monkeypatch.setattr(
        streaming_transform_core,
        "transformation_target_manifest_sha256",
        canonical_digest,
    )
    materialize_transformation_artifact(
        baseline_path=baseline,
        output_path=artifact,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any("out-of-scope tensor changed" in issue for issue in result["issues"])


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        ("out_of_scope_tensor", "out-of-scope tensor changed"),
        ("tokenizer", "support file changed: tokenizer.json"),
    ],
)
def test_replay_rejects_out_of_scope_and_support_file_drift(
    tmp_path: Path,
    mutation: str,
    expected_issue: str,
) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    if mutation == "out_of_scope_tensor":
        tensors, _ = _artifact_tensors(artifact)
        tensors["model.layers.0.self_attn.q_proj.weight"] += 1.0
        _rewrite_artifact_tensors(artifact, tensors)
    elif mutation == "tokenizer":
        (artifact / "tokenizer.json").write_text(
            '{"version":"changed"}\n', encoding="utf-8"
        )
    else:  # pragma: no cover - parametrization contract
        raise AssertionError(f"unknown mutation: {mutation}")

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any(expected_issue in issue for issue in result["issues"])


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        ("metadata", "edit_metadata.parameters does not match transformation replay"),
        (
            "manifest",
            "materialization receipt target_manifest_sha256 does not match replay",
        ),
        ("config", "configuration does not match baseline"),
        (
            "baseline_identity",
            "materialization receipt baseline_identity does not match replay",
        ),
    ],
)
def test_replay_rejects_each_bound_metadata_or_identity_claim(
    tmp_path: Path,
    mutation: str,
    expected_issue: str,
) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    if mutation == "metadata":
        metadata_path = artifact / "edit_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["parameters"] = {"bits": 8, "group_size": 2}
        _write_json(metadata_path, metadata)
    elif mutation == "manifest":
        receipt_path = artifact / TRANSFORMATION_MATERIALIZATION_RECEIPT
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["target_manifest_sha256"] = "sha256:" + "0" * 64
        _write_json(receipt_path, receipt)
    elif mutation == "config":
        _write_json(
            artifact / "config.json",
            {"model_type": "qwen3", "num_hidden_layers": 1},
        )
    elif mutation == "baseline_identity":
        source = _baseline_tensors(baseline)
        source["model.layers.0.mlp.up_proj.weight"] += 1.0
        save_file(source, baseline / "model.safetensors", metadata={"format": "pt"})
    else:  # pragma: no cover - parametrization contract
        raise AssertionError(f"unknown mutation: {mutation}")

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any(expected_issue in issue for issue in result["issues"])


def test_replay_rejects_an_unsafe_weight_index(tmp_path: Path) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    index_path = artifact / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["model.layers.0.mlp.up_proj.weight"] = (
        "../baseline/model.safetensors"
    )
    _write_json(index_path, index)

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any(
        "path must be checkpoint-relative" in issue for issue in result["issues"]
    )


def test_replay_rejects_forged_output_plan_digest(tmp_path: Path) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    forged_digest = "sha256:" + "b" * 64
    receipt_path = artifact / TRANSFORMATION_MATERIALIZATION_RECEIPT
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_shard_plan_sha256"] = forged_digest
    _write_json(receipt_path, receipt)
    metadata_path = artifact / "edit_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_shard_plan_sha256"] = forged_digest
    _write_json(metadata_path, metadata)

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any(
        "output_shard_plan_sha256 does not match replay" in issue
        for issue in result["issues"]
    )


def test_replay_rejects_forged_source_plan(tmp_path: Path) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    receipt_path = artifact / TRANSFORMATION_MATERIALIZATION_RECEIPT
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged_plan = json.loads(json.dumps(receipt["source_shard_plan"]))
    forged_plan["source_shards"][0]["byte_count"] += 1
    receipt["source_shard_plan"] = forged_plan
    receipt["source_shard_plan_sha256"] = _canonical_json_sha256(forged_plan)
    _write_json(receipt_path, receipt)
    metadata_path = artifact / "edit_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_shard_plan"] = forged_plan
    metadata["source_shard_plan_sha256"] = receipt["source_shard_plan_sha256"]
    _write_json(metadata_path, metadata)

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any(
        "source_shard_plan does not match replay" in issue for issue in result["issues"]
    )


def test_replay_rejects_resharding_with_forged_plan_and_receipt(tmp_path: Path) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    tensors, _ = _artifact_tensors(artifact)
    original_index_path = artifact / "model.safetensors.index.json"
    original_index = json.loads(original_index_path.read_text(encoding="utf-8"))
    original_shards = set(original_index["weight_map"].values())
    split_names = sorted(tensors)
    shard_names = ["forged-00001.safetensors", "forged-00002.safetensors"]
    first_names, second_names = split_names[:1], split_names[1:]
    for shard_name, names in zip(shard_names, (first_names, second_names), strict=True):
        save_file(
            {name: tensors[name].contiguous() for name in names},
            artifact / shard_name,
            metadata={"format": "pt"},
        )
    for original_shard in original_shards:
        (artifact / original_shard).unlink()
    forged_weight_map = dict.fromkeys(first_names, shard_names[0])
    forged_weight_map.update(dict.fromkeys(second_names, shard_names[1]))
    _write_json(
        original_index_path,
        {"metadata": {"total_size": 96}, "weight_map": forged_weight_map},
    )

    receipt_path = artifact / TRANSFORMATION_MATERIALIZATION_RECEIPT
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    source_shard = receipt["source_shard_plan"]["source_shards"][0]
    forged_chunks = []
    for shard_name, names in zip(shard_names, (first_names, second_names), strict=True):
        forged_chunks.append(
            {
                "name": shard_name,
                "source_path": source_shard["path"],
                "source_sha256": source_shard["sha256"],
                "tensor_names": names,
                "byte_count": sum(
                    int(tensors[name].numel() * tensors[name].element_size())
                    for name in names
                ),
            }
        )
    forged_plan = {
        "source_shard_plan_sha256": receipt["source_shard_plan_sha256"],
        "target_manifest_sha256": receipt["target_manifest_sha256"],
        "chunks": forged_chunks,
    }
    receipt["output_shard_plan"] = forged_plan
    receipt["output_shard_plan_sha256"] = _canonical_json_sha256(forged_plan)
    receipt["output_shards"] = len(forged_chunks)
    receipt["output_weights"] = _output_weight_identity(
        artifact,
        weight_map={
            name: artifact / shard for name, shard in forged_weight_map.items()
        },
        index_path=original_index_path,
    )
    _write_json(receipt_path, receipt)
    metadata_path = artifact / "edit_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_shard_plan"] = forged_plan
    metadata["output_shard_plan_sha256"] = receipt["output_shard_plan_sha256"]
    _write_json(metadata_path, metadata)

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any(
        "artifact safetensors index does not match canonical output shard plan" in issue
        for issue in result["issues"]
    )


@pytest.mark.parametrize("corrupt_tree", ["baseline", "artifact"])
def test_replay_fails_closed_for_malformed_safetensors(
    tmp_path: Path,
    corrupt_tree: str,
) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    if corrupt_tree == "baseline":
        target = baseline / "model.safetensors"
    else:
        _, weight_map = _artifact_tensors(artifact)
        target = artifact / next(iter(weight_map.values()))
    target.write_bytes(b"not a safetensors file")

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any("unreadable" in issue for issue in result["issues"])


def test_replay_rejects_a_mismatched_scope(tmp_path: Path) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="attn",
    )

    assert result["ok"] is False
    assert any(
        "receipt scope does not match replay" in issue for issue in result["issues"]
    )


def test_replay_fails_closed_when_a_checkpoint_changes_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    original_identity = validator_module.checkpoint_tree_sha256
    artifact_identity_calls = 0

    def unstable_identity(path: Path) -> str:
        nonlocal artifact_identity_calls
        digest = original_identity(path)
        if Path(path) == artifact:
            artifact_identity_calls += 1
            if artifact_identity_calls == 2:
                return "sha256:" + "f" * 64
        return digest

    monkeypatch.setattr(validator_module, "checkpoint_tree_sha256", unstable_identity)
    result = validator_module.validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters=parameters,
        scope="ffn",
    )

    assert result["ok"] is False
    assert any(
        "artifact checkpoint changed during transformation replay validation" in issue
        for issue in result["issues"]
    )


@pytest.mark.parametrize("edit_type", ["fp8_quant", "lowrank_svd"])
def test_replay_rejects_unsupported_generated_transforms(
    tmp_path: Path,
    edit_type: str,
) -> None:
    baseline, artifact, _ = _materialize(tmp_path)

    result = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type=edit_type,
        parameters={"bits": 4, "group_size": 2},
        scope="ffn",
    )

    assert result["ok"] is False
    assert any(
        "no verifier-grade generated-lane contract" in issue
        for issue in result["issues"]
    )


def test_transform_cli_writes_an_identity_bound_replay_sidecar(tmp_path: Path) -> None:
    baseline, artifact, parameters = _materialize(tmp_path)
    replay_path = tmp_path / "replay.json"

    exit_code = main(
        [
            "validate_artifact.py",
            "transform",
            str(artifact),
            "--baseline",
            str(baseline),
            "--edit-type",
            "quant_rtn",
            "--parameters-json",
            json.dumps(parameters),
            "--scope",
            "ffn",
            "--out",
            str(replay_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["artifact_identity"]["sha256"] == checkpoint_tree_sha256(artifact)
    assert payload["materialization_receipt_sha256"].startswith("sha256:")
