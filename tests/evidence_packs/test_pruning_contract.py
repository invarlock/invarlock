from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from invarlock.pruning_contract import (
    PRUNING_SCOPE_POLICY_VERSION,
    PruningContractError,
    checkpoint_pruning_contract,
    is_pruning_target,
    pruning_target_manifest,
    pruning_target_manifest_sha256,
    validate_pruning_target_manifest,
)


def _contract(tmp_path: Path, config: dict[str, object]):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return checkpoint_pruning_contract(checkpoint)


@pytest.mark.parametrize(
    "model_type",
    (
        "albert",
        "deepseek_v2",
        "deepseek_v3",
        "falcon_h1",
        "gemma3",
        "gemma3n",
        "gemma4",
        "mixtral",
        "phi",
        "phi3",
        "phi4",
        "qwen2_moe",
        "qwen3_moe",
        "qwen3_5",
        "qwen3_5_vl",
        "starcoder2",
    ),
)
def test_unreviewed_target_layouts_fail_closed_before_pruning(
    tmp_path: Path, model_type: str
) -> None:
    with pytest.raises(PruningContractError, match="no explicit resolver"):
        _contract(tmp_path, {"model_type": model_type})


def test_distilbert_and_qwen3_language_scope_are_explicit(tmp_path: Path) -> None:
    distilbert = _contract(tmp_path / "distil", {"model_type": "distilbert"})
    assert is_pruning_target(
        "transformer.layer.0.ffn.lin1.weight",
        scope="ffn",
        contract=distilbert,
        ndim=2,
    )
    assert is_pruning_target(
        "transformer.layer.0.attention.q_lin.weight",
        scope="attn",
        contract=distilbert,
        ndim=2,
    )

    qwen3 = _contract(tmp_path / "qwen", {"model_type": "qwen3"})
    assert is_pruning_target(
        "model.language_model.layers.0.mlp.up_proj.weight",
        scope="ffn",
        contract=qwen3,
        ndim=2,
    )
    assert not is_pruning_target(
        "model.visual.blocks.0.mlp.up_proj.weight",
        scope="all",
        contract=qwen3,
        ndim=2,
    )
    for non_language_path in (
        "model.mtp.layers.0.mlp.up_proj.weight",
        "model.layers.0.aux.mlp.up_proj.weight",
        "model.layers.0.auxiliary.mlp.up_proj.weight",
        "model.layers.0.multi_token_prediction.mlp.up_proj.weight",
    ):
        assert not is_pruning_target(
            non_language_path,
            scope="all",
            contract=qwen3,
            ndim=2,
        )


def test_bloom_uses_its_transformer_h_layer_grammar(tmp_path: Path) -> None:
    bloom = _contract(tmp_path, {"model_type": "bloom"})
    assert is_pruning_target(
        "transformer.h.0.mlp.dense_h_to_4h.weight",
        scope="ffn",
        contract=bloom,
        ndim=2,
    )
    assert is_pruning_target(
        "transformer.h.0.self_attention.query_key_value.weight",
        scope="attn",
        contract=bloom,
        ndim=2,
    )


def test_unknown_or_quantized_architecture_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PruningContractError, match="no explicit resolver"):
        _contract(tmp_path / "unknown", {"model_type": "unreviewed_architecture"})
    with pytest.raises(PruningContractError, match="GPT-OSS/MXFP4"):
        _contract(
            tmp_path / "gpt_oss",
            {
                "model_type": "gpt_oss",
                "quantization_config": {"quant_method": "mxfp4"},
            },
        )


def test_target_manifest_binds_policy_config_and_exact_selected_tensors(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, {"model_type": "qwen2"})
    manifest = pruning_target_manifest(
        scope="ffn",
        contract=contract,
        targets=[
            {
                "name": "model.layers.1.mlp.up_proj.weight",
                "dtype": "torch.float16",
                "shape": [2, 4],
                "numel": 8,
            },
            {
                "name": "model.layers.0.mlp.down_proj.weight",
                "dtype": "torch.float16",
                "shape": [4, 2],
                "numel": 8,
            },
        ],
    )

    assert manifest["scope_policy"] == PRUNING_SCOPE_POLICY_VERSION
    assert [entry["name"] for entry in manifest["targets"]] == [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
    ]
    digest = pruning_target_manifest_sha256(manifest)
    assert digest.startswith("sha256:")
    altered = dict(manifest)
    altered["scope_policy"] = "forged-policy"
    with pytest.raises(PruningContractError, match="scope_policy"):
        pruning_target_manifest_sha256(altered)


@pytest.mark.parametrize(
    ("description", "mutate", "error"),
    [
        (
            "vision tensor",
            lambda manifest: manifest["targets"][0].update(
                {"name": "model.visual.blocks.0.mlp.up_proj.weight"}
            ),
            "outside the canonical pruning scope",
        ),
        (
            "MTP tensor",
            lambda manifest: manifest["targets"][0].update(
                {"name": "model.mtp.layers.0.mlp.up_proj.weight"}
            ),
            "outside the canonical pruning scope",
        ),
        (
            "attention tensor in FFN scope",
            lambda manifest: manifest["targets"][0].update(
                {"name": "model.layers.0.self_attn.q_proj.weight"}
            ),
            "outside the canonical pruning scope",
        ),
        (
            "GPT-OSS MXFP4 representation",
            lambda manifest: manifest.update(
                {"model_type": "gpt_oss", "architecture": "decoder"}
            ),
            "model_type and architecture mismatch",
        ),
        (
            "unknown model family",
            lambda manifest: manifest.update(
                {"model_type": "unreviewed_architecture", "architecture": "decoder"}
            ),
            "model_type and architecture mismatch",
        ),
        (
            "retired schema",
            lambda manifest: manifest.update(
                {"schema": "invarlock/pruning-target-manifest-v2"}
            ),
            "schema is invalid",
        ),
        (
            "retired scope policy",
            lambda manifest: manifest.update(
                {"scope_policy": "architecture-aware-pruning-v4"}
            ),
            "scope_policy is invalid",
        ),
    ],
)
def test_external_target_manifest_rejects_semantically_invalid_self_consistent_claims(
    tmp_path: Path,
    description: str,
    mutate: object,
    error: str,
) -> None:
    """A digestable sidecar must still obey the package-owned policy."""

    contract = _contract(tmp_path, {"model_type": "qwen2"})
    manifest = pruning_target_manifest(
        scope="ffn",
        contract=contract,
        targets=[
            {
                "name": "model.layers.0.mlp.up_proj.weight",
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
            }
        ],
    )
    forged = copy.deepcopy(manifest)
    assert callable(mutate), description
    mutate(forged)

    with pytest.raises(PruningContractError, match=error):
        validate_pruning_target_manifest(forged)


def test_external_target_manifest_rejects_duplicate_and_noncanonical_targets(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, {"model_type": "qwen2"})
    manifest = pruning_target_manifest(
        scope="ffn",
        contract=contract,
        targets=[
            {
                "name": "model.layers.0.mlp.down_proj.weight",
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
            },
            {
                "name": "model.layers.1.mlp.up_proj.weight",
                "dtype": "torch.float32",
                "shape": [2, 2],
                "numel": 4,
            },
        ],
    )

    unsorted = copy.deepcopy(manifest)
    unsorted["targets"].reverse()
    with pytest.raises(PruningContractError, match="sorted and unique"):
        validate_pruning_target_manifest(unsorted)

    duplicate = copy.deepcopy(manifest)
    duplicate["targets"].append(copy.deepcopy(duplicate["targets"][0]))
    with pytest.raises(PruningContractError, match="sorted and unique"):
        validate_pruning_target_manifest(duplicate)
