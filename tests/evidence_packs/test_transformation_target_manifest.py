from __future__ import annotations

import ast
import inspect

import pytest

from invarlock import transformation_target_manifest as target_manifest_contract
from invarlock.transformation_target_manifest import (
    TransformationTargetManifestError,
    transformation_target_manifest_sha256,
    validate_transformation_target_manifest,
)


def _manifest(
    *,
    model_type: str = "qwen2",
    architecture: str = "decoder",
    scope: str = "ffn",
    layer_count: int = 2,
    target: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema": target_manifest_contract.TRANSFORMATION_TARGET_MANIFEST_SCHEMA,
        "contract_version": target_manifest_contract.TRANSFORMATION_CONTRACT_VERSION,
        "scope_policy": target_manifest_contract.TRANSFORMATION_SCOPE_POLICY,
        "edit_type": "quant_rtn",
        "algorithm": "groupwise_rtn_dequantized_per_row_v1",
        "parameters": {"bits": 4, "group_size": 32},
        "scope": scope,
        "model_type": model_type,
        "architecture": architecture,
        "config_sha256": "sha256:" + "a" * 64,
        "layer_count": layer_count,
        "targets": [
            target
            or {
                "name": "model.layers.0.mlp.up_proj.weight",
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
                "role": "ffn",
                "layer": 0,
            }
        ],
    }


def test_package_target_manifest_contract_has_no_generator_or_pruning_dependency() -> (
    None
):
    tree = ast.parse(inspect.getsource(target_manifest_contract))
    forbidden = {
        "scripts.evidence_packs",
        "invarlock.pruning_contract",
        "invarlock.transformation_contract",
        "invarlock.streaming_transform",
    }
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        imported_name == forbidden_name
        or imported_name.startswith(forbidden_name + ".")
        for imported_name in imported
        for forbidden_name in forbidden
    )


def test_validates_canonical_qwen_target_and_digest() -> None:
    manifest = _manifest()

    assert validate_transformation_target_manifest(manifest) == manifest
    assert transformation_target_manifest_sha256(manifest).startswith("sha256:")


def test_rejects_retired_manifest_versions_and_boolean_layers() -> None:
    retired = _manifest()
    retired["schema"] = "invarlock/transformation-target-manifest-v2"
    with pytest.raises(TransformationTargetManifestError, match="schema"):
        validate_transformation_target_manifest(retired)

    boolean_layer = _manifest(
        target={
            "name": "model.layers.1.mlp.up_proj.weight",
            "dtype": "torch.float32",
            "shape": [2, 2],
            "numel": 4,
            "role": "ffn",
            "layer": True,
        }
    )
    with pytest.raises(TransformationTargetManifestError, match="layer"):
        validate_transformation_target_manifest(boolean_layer)


def test_rejects_scope_or_target_outside_bound_layer_count() -> None:
    with pytest.raises(TransformationTargetManifestError, match="layer qualifier"):
        validate_transformation_target_manifest(_manifest(scope="ffn@layer=2"))
    forged = _manifest(
        target={
            "name": "model.layers.999.mlp.up_proj.weight",
            "dtype": "torch.float32",
            "shape": [2, 2],
            "numel": 4,
            "role": "ffn",
            "layer": 999,
        }
    )
    with pytest.raises(TransformationTargetManifestError, match="layer"):
        validate_transformation_target_manifest(forged)


@pytest.mark.parametrize(
    ("edit_type", "algorithm", "parameters"),
    (
        (
            "synthetic_lowrank_delta",
            "deterministic_synthetic_lowrank_delta_v1",
            {"rank": 33, "scale": 1.0},
        ),
        (
            "synthetic_dense_update",
            "deterministic_synthetic_dense_update_v1",
            {"step_size": 0.001, "iterations": 17},
        ),
    ),
)
def test_rejects_unbounded_transform_work(
    edit_type: str, algorithm: str, parameters: dict[str, object]
) -> None:
    manifest = _manifest()
    manifest.update(
        {
            "edit_type": edit_type,
            "algorithm": algorithm,
            "parameters": parameters,
        }
    )

    with pytest.raises(TransformationTargetManifestError, match="must not exceed"):
        validate_transformation_target_manifest(manifest)


@pytest.mark.parametrize(
    "name",
    (
        "model.visual.blocks.0.mlp.up_proj.weight",
        "model.layers.0.mtp.mlp.up_proj.weight",
        "model.layers.0.auxiliary.mlp.up_proj.weight",
    ),
)
def test_rejects_multimodal_mtp_and_auxiliary_qwen_targets(name: str) -> None:
    manifest = _manifest()
    target = manifest["targets"][0]
    assert isinstance(target, dict)
    target["name"] = name

    with pytest.raises(
        TransformationTargetManifestError, match="outside the independent"
    ):
        validate_transformation_target_manifest(manifest)


def test_rejects_mismatched_target_role_and_scope_qualifier() -> None:
    manifest = _manifest()
    target = manifest["targets"][0]
    assert isinstance(target, dict)
    target["role"] = "attn"
    with pytest.raises(TransformationTargetManifestError, match="role"):
        validate_transformation_target_manifest(manifest)

    qualified = _manifest(
        scope="ffn@layers=2,layer=1",
        target={
            "name": "model.layers.1.mlp.up_proj.weight",
            "dtype": "torch.float32",
            "shape": [2, 2],
            "numel": 4,
            "role": "ffn",
            "layer": 1,
        },
    )
    assert validate_transformation_target_manifest(qualified) == qualified
    qualified["scope"] = "ffn@layers=1,layer=0"
    with pytest.raises(TransformationTargetManifestError, match="layers qualifier"):
        validate_transformation_target_manifest(qualified)


@pytest.mark.parametrize(
    ("model_type", "architecture", "error"),
    (
        ("unreviewed_architecture", "decoder", "no independent target resolver"),
        ("mixtral", "mixtral", "no independent target resolver"),
        ("gemma3", "decoder", "no independent target resolver"),
        ("qwen3_5", "decoder", "no independent target resolver"),
        ("deepseek_v3", "decoder", "no independent target resolver"),
        ("gpt_oss", "decoder", "GPT-OSS/MXFP4"),
        ("qwen2", "gpt2", "model_type and architecture mismatch"),
    ),
)
def test_rejects_unknown_or_unsupported_model_identity(
    model_type: str, architecture: str, error: str
) -> None:
    with pytest.raises(TransformationTargetManifestError, match=error):
        validate_transformation_target_manifest(
            _manifest(model_type=model_type, architecture=architecture)
        )


def test_accepts_bloom_transformer_h_target_with_exact_layer() -> None:
    manifest = _manifest(
        model_type="bloom",
        architecture="gpt_neox",
        target={
            "name": "transformer.h.1.mlp.dense_h_to_4h.weight",
            "dtype": "torch.float32",
            "shape": [2, 2],
            "numel": 4,
            "role": "ffn",
            "layer": 1,
        },
    )
    assert validate_transformation_target_manifest(manifest) == manifest


def test_rejects_mxfp_or_noncanonical_algorithm_claims() -> None:
    manifest = _manifest()
    target = manifest["targets"][0]
    assert isinstance(target, dict)
    target["dtype"] = "torch.mxfp4"
    with pytest.raises(TransformationTargetManifestError, match="regular floating"):
        validate_transformation_target_manifest(manifest)

    manifest = _manifest()
    manifest["algorithm"] = "generator-selected-algorithm"
    with pytest.raises(TransformationTargetManifestError, match="algorithm"):
        validate_transformation_target_manifest(manifest)


def test_rejects_lowrank_rank_outside_the_target_matrix() -> None:
    manifest = _manifest()
    manifest["edit_type"] = "synthetic_lowrank_delta"
    manifest["algorithm"] = "deterministic_synthetic_lowrank_delta_v1"
    manifest["parameters"] = {"rank": 3, "scale": 0.1}

    with pytest.raises(TransformationTargetManifestError, match="matrix rank"):
        validate_transformation_target_manifest(manifest)
