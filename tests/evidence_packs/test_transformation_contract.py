from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evidence_packs.python.editing.transformation_contract import (
    GROUPWISE_RTN_DEQUANTIZED_ALGORITHM,
    SYNTHETIC_DENSE_UPDATE_ALGORITHM,
    SYNTHETIC_LOWRANK_DELTA_ALGORITHM,
    TRANSFORMATION_CONTRACT_VERSION,
    TRANSFORMATION_SCOPE_POLICY_VERSION,
    TransformationContractError,
    UnsupportedTransformationError,
    canonical_transformation_parameters,
    canonical_transformation_spec,
    checkpoint_transformation_contract,
    is_transformation_target,
    parse_transformation_scope,
    transformation_target_manifest,
    transformation_target_manifest_sha256,
)


def _contract(tmp_path: Path, config: dict[str, object]):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True)
    normalized = dict(config)
    model_type = normalized.get("model_type")
    if model_type == "bloom":
        normalized.setdefault("n_layer", 2)
    elif model_type in {
        "gemma",
        "gemma2",
        "llama",
        "mistral",
        "olmo",
        "olmo2",
        "qwen2",
        "qwen3",
        "falcon",
        "gpt_neox",
        "opt",
        "bert",
        "roberta",
    }:
        normalized.setdefault("num_hidden_layers", 2)
    elif model_type in {"gpt2", "gpt_bigcode"}:
        normalized.setdefault("n_layer", 2)
    elif model_type == "distilbert":
        normalized.setdefault("n_layers", 2)
    (checkpoint / "config.json").write_text(json.dumps(normalized), encoding="utf-8")
    return checkpoint_transformation_contract(checkpoint)


def _qwen_targets() -> list[dict[str, object]]:
    return [
        {
            "name": "model.layers.1.mlp.down_proj.weight",
            "dtype": "torch.float16",
            "shape": [4, 2],
            "numel": 8,
        },
        {
            "name": "model.layers.0.mlp.up_proj.weight",
            "dtype": "torch.float16",
            "shape": [2, 4],
            "numel": 8,
        },
    ]


def test_canonical_parameters_bind_the_supported_algorithms() -> None:
    quant = canonical_transformation_spec("quant_rtn", {"bits": 4, "group_size": 64})
    assert quant == {
        "schema": "invarlock/transformation-parameters-v1",
        "contract_version": TRANSFORMATION_CONTRACT_VERSION,
        "edit_type": "quant_rtn",
        "algorithm": GROUPWISE_RTN_DEQUANTIZED_ALGORITHM,
        "parameters": {"bits": 4, "group_size": 64},
    }
    assert (
        canonical_transformation_spec(
            "synthetic_lowrank_delta", {"rank": 2, "scale": 0.25}
        )["algorithm"]
        == SYNTHETIC_LOWRANK_DELTA_ALGORITHM
    )
    assert (
        canonical_transformation_spec(
            "synthetic_dense_update", {"step_size": 0.001, "iterations": 3}
        )["algorithm"]
        == SYNTHETIC_DENSE_UPDATE_ALGORITHM
    )


@pytest.mark.parametrize(
    ("edit_type", "parameters"),
    [
        ("quant_rtn", {"bits": 0, "group_size": 64}),
        ("quant_rtn", {"bits": True, "group_size": 64}),
        ("quant_rtn", {"bits": 4, "group_size": 0}),
        ("quant_rtn", {"bits": 4, "group_size": 64, "ignored": 1}),
        ("synthetic_lowrank_delta", {"rank": 0, "scale": 1.0}),
        ("synthetic_lowrank_delta", {"rank": 33, "scale": 1.0}),
        ("synthetic_lowrank_delta", {"rank": 1, "scale": float("inf")}),
        ("synthetic_dense_update", {"step_size": 0.0, "iterations": 1}),
        ("synthetic_dense_update", {"step_size": 0.1, "iterations": 17}),
        ("synthetic_dense_update", {"step_size": 0.1, "iterations": False}),
    ],
)
def test_zero_nonfinite_and_ambiguous_parameters_fail_closed(
    edit_type: str,
    parameters: dict[str, object],
) -> None:
    with pytest.raises(TransformationContractError):
        canonical_transformation_parameters(edit_type, parameters)


@pytest.mark.parametrize("edit_type", ["fp8_quant", "lowrank_svd"])
def test_unsupported_families_cannot_enter_verifier_grade_lanes(
    edit_type: str,
) -> None:
    with pytest.raises(UnsupportedTransformationError, match="dedicated storage"):
        canonical_transformation_parameters(edit_type, {})


def test_scope_parser_is_strict_and_canonical() -> None:
    parsed = parse_transformation_scope(" ATTN @ layer=0 , layers=2 ")
    assert parsed.base_scope == "attn"
    assert parsed.layer == 0
    assert parsed.layer_limit == 2
    assert parsed.canonical == "attn@layers=2,layer=0"

    for malformed in (
        "ffn@",
        "ffn@layers=0",
        "ffn@layer=-1",
        "ffn@layers=1.0",
        "ffn@layers=01",
        "ffn@layers=2,layers=1",
        "ffn@unknown=1",
        "ffn@layers=1,layer=1",
        "ffn@layer=0,",
        "unknown",
    ):
        with pytest.raises(TransformationContractError):
            parse_transformation_scope(malformed)


def test_architecture_aware_targeting_applies_layer_qualifiers_and_excludes_auxiliary_paths(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, {"model_type": "qwen3"})
    layer_zero = "model.language_model.layers.0.self_attn.q_proj.weight"
    layer_one = "model.language_model.layers.1.mlp.up_proj.weight"

    assert is_transformation_target(
        layer_zero,
        scope="attn@layer=0",
        contract=contract,
        ndim=2,
    )
    assert not is_transformation_target(
        layer_one,
        scope="ffn@layers=1",
        contract=contract,
        ndim=2,
    )
    for non_language_path in (
        "model.visual.blocks.0.mlp.up_proj.weight",
        "model.layers.0.aux.mlp.up_proj.weight",
        "model.layers.0.auxiliary.mlp.up_proj.weight",
        "model.layers.0.multi_token_prediction.mlp.up_proj.weight",
    ):
        assert not is_transformation_target(
            non_language_path,
            scope="all",
            contract=contract,
            ndim=2,
        )


def test_layer_scope_and_injected_target_outside_configured_topology_fail_closed(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, {"model_type": "qwen2", "num_hidden_layers": 1})
    with pytest.raises(TransformationContractError, match="declared layer count"):
        is_transformation_target(
            "model.layers.999.mlp.up_proj.weight",
            scope="ffn@layer=999",
            contract=contract,
            ndim=2,
        )
    assert not is_transformation_target(
        "model.layers.999.mlp.up_proj.weight",
        scope="ffn",
        contract=contract,
        ndim=2,
    )


def test_missing_declared_layer_count_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "qwen2"}), encoding="utf-8"
    )
    with pytest.raises(TransformationContractError, match="num_hidden_layers"):
        checkpoint_transformation_contract(checkpoint)


@pytest.mark.parametrize(
    "model_type",
    ("gemma3", "mixtral", "qwen2_moe", "qwen3_moe", "qwen3_5", "deepseek_v3"),
)
def test_unreviewed_raw_transformation_layouts_fail_closed(
    tmp_path: Path, model_type: str
) -> None:
    with pytest.raises(TransformationContractError, match="no explicit resolver"):
        _contract(tmp_path, {"model_type": model_type})


def test_bloom_raw_targeting_uses_its_transformer_h_layer_grammar(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, {"model_type": "bloom"})
    assert is_transformation_target(
        "transformer.h.1.mlp.dense_h_to_4h.weight",
        scope="ffn@layer=1",
        contract=contract,
        ndim=2,
    )


def test_target_manifest_binds_parameters_scope_architecture_and_exact_targets(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, {"model_type": "qwen2"})
    manifest = transformation_target_manifest(
        edit_type="quant_rtn",
        parameters={"bits": 4, "group_size": 64},
        scope="ffn@layers=2",
        contract=contract,
        targets=_qwen_targets(),
    )

    assert manifest["scope"] == "ffn@layers=2"
    assert manifest["scope_policy"] == TRANSFORMATION_SCOPE_POLICY_VERSION
    assert manifest["algorithm"] == GROUPWISE_RTN_DEQUANTIZED_ALGORITHM
    assert manifest["layer_count"] == 2
    assert [target["name"] for target in manifest["targets"]] == [
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]
    digest = transformation_target_manifest_sha256(manifest)
    assert digest.startswith("sha256:")

    forged = dict(manifest)
    forged["algorithm"] = "made_up_algorithm"
    with pytest.raises(TransformationContractError, match="not canonical"):
        transformation_target_manifest_sha256(forged)

    forged = dict(manifest)
    forged["scope_policy"] = "made_up_policy"
    with pytest.raises(TransformationContractError, match="not canonical"):
        transformation_target_manifest_sha256(forged)

    forged = dict(manifest)
    forged_targets = [dict(target) for target in manifest["targets"]]
    forged_targets[0]["layer"] = 99
    forged["targets"] = forged_targets
    with pytest.raises(TransformationContractError, match="not canonical"):
        transformation_target_manifest_sha256(forged)


def test_manifest_rejects_unselected_targets_and_silent_rank_clamping(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, {"model_type": "qwen2"})
    with pytest.raises(TransformationContractError, match="outside"):
        transformation_target_manifest(
            edit_type="quant_rtn",
            parameters={"bits": 4, "group_size": 64},
            scope="ffn",
            contract=contract,
            targets=[
                {
                    "name": "model.layers.0.self_attn.q_proj.weight",
                    "dtype": "torch.float16",
                    "shape": [2, 4],
                    "numel": 8,
                }
            ],
        )
    with pytest.raises(TransformationContractError, match="matrix rank"):
        transformation_target_manifest(
            edit_type="synthetic_lowrank_delta",
            parameters={"rank": 3, "scale": 1.0},
            scope="ffn",
            contract=contract,
            targets=[
                {
                    "name": "model.layers.0.mlp.up_proj.weight",
                    "dtype": "torch.float16",
                    "shape": [2, 4],
                    "numel": 8,
                }
            ],
        )


def test_unknown_architecture_is_not_a_usable_transformation_subject(
    tmp_path: Path,
) -> None:
    with pytest.raises(TransformationContractError, match="no explicit resolver"):
        _contract(tmp_path, {"model_type": "unreviewed_architecture"})
