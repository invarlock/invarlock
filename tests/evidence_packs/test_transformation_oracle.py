from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.evidence_packs.python.editing.streaming_transform import (
    replay_transformation_tensor,
)
from scripts.evidence_packs.python.editing.transformation_contract import (
    TransformationContractError,
)
from scripts.evidence_packs.python.editing.transformation_oracle import (
    TransformationOracleError,
    build_transformation_oracle,
)


def _oracle(
    tmp_path: Path,
    *,
    edit_type: str,
    parameters: dict[str, object],
    scope: str = "ffn",
    model_type: str = "qwen2",
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    layer_count_key = (
        "n_layer"
        if model_type in {"bloom", "gpt2", "gpt_bigcode"}
        else "num_hidden_layers"
    )
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": model_type, layer_count_key: 2}),
        encoding="utf-8",
    )
    return build_transformation_oracle(
        checkpoint,
        edit_type=edit_type,
        parameters=parameters,
        scope=scope,
    )


@pytest.mark.parametrize(
    ("edit_type", "parameters"),
    [
        ("quant_rtn", {"bits": 2, "group_size": 3}),
        ("synthetic_lowrank_delta", {"rank": 2, "scale": 2.0}),
        ("synthetic_dense_update", {"step_size": 0.001, "iterations": 3}),
    ],
)
def test_oracle_matches_v1_numerical_abi_without_importing_generator(
    tmp_path: Path,
    edit_type: str,
    parameters: dict[str, object],
) -> None:
    tensor = torch.tensor(
        [[0.37, 0.72, -1.11, 2.43], [-0.31, 1.79, 0.16, -2.07]],
        dtype=torch.float32,
    )
    oracle = _oracle(tmp_path, edit_type=edit_type, parameters=parameters)

    actual = oracle.replay_tensor(tensor)
    expected = replay_transformation_tensor(
        tensor, edit_type=edit_type, parameters=parameters
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("tensor", "parameters", "expected_sha256"),
    [
        (
            torch.linspace(-1.0, 1.0, 1024 * 128, dtype=torch.float32)
            .reshape(1024, 128)
            .to(torch.bfloat16),
            {"rank": 6, "scale": 3.141592653589793},
            "846aadb9163dd37d5d378079786253aefeacad53525ee6d10efc81bd692f047e",
        ),
        (
            torch.linspace(-0.75, 0.75, 513 * 31, dtype=torch.float64).reshape(513, 31),
            {"rank": 6, "scale": 2.5},
            "14a52e0e04d765f00eebff02bee9c7cb64de32bb4a62f7a27188ac30cfdfc1bd",
        ),
    ],
    ids=("bfloat16-four-row-chunks", "float64-chunk-boundary"),
)
def test_v1_lowrank_has_frozen_ordered_accumulation_vectors(
    tmp_path: Path,
    tensor: torch.Tensor,
    parameters: dict[str, object],
    expected_sha256: str,
) -> None:
    """Lock the specified component order across low-precision/chunk boundaries."""

    oracle = _oracle(
        tmp_path,
        edit_type="synthetic_lowrank_delta",
        parameters=parameters,
    )
    actual = oracle.replay_tensor(tensor)
    materialized = replay_transformation_tensor(
        tensor,
        edit_type="synthetic_lowrank_delta",
        parameters=parameters,
    )

    digest = hashlib.sha256(
        actual.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()
    assert digest == expected_sha256
    assert torch.equal(actual, materialized)


@pytest.mark.parametrize(
    "non_language_path",
    (
        "model.visual.blocks.0.mlp.up_proj.weight",
        "model.layers.0.aux.mlp.up_proj.weight",
        "model.layers.0.auxiliary.mlp.up_proj.weight",
        "model.layers.0.multi_token_prediction.mlp.up_proj.weight",
    ),
)
def test_oracle_excludes_non_language_qwen_paths(
    tmp_path: Path, non_language_path: str
) -> None:
    oracle = _oracle(
        tmp_path,
        edit_type="quant_rtn",
        parameters={"bits": 4, "group_size": 2},
        scope="ffn",
    )
    language = torch.ones((2, 2), dtype=torch.float32)
    non_language = torch.ones((2, 2), dtype=torch.float32)

    assert oracle.is_target("model.layers.0.mlp.up_proj.weight", language)
    assert not oracle.is_target(non_language_path, non_language)


@pytest.mark.parametrize(
    "model_type",
    ("gemma3", "mixtral", "qwen2_moe", "qwen3_moe", "qwen3_5", "deepseek_v3"),
)
def test_oracle_rejects_unreviewed_raw_transformation_layouts(
    tmp_path: Path, model_type: str
) -> None:
    with pytest.raises(
        TransformationOracleError, match="no independent target resolver"
    ):
        _oracle(
            tmp_path,
            edit_type="quant_rtn",
            parameters={"bits": 4, "group_size": 2},
            scope="ffn",
            model_type=model_type,
        )


def test_oracle_uses_bloom_transformer_h_layer_grammar(tmp_path: Path) -> None:
    oracle = _oracle(
        tmp_path,
        edit_type="quant_rtn",
        parameters={"bits": 4, "group_size": 2},
        scope="ffn@layer=1",
        model_type="bloom",
    )
    assert oracle.is_target(
        "transformer.h.1.mlp.dense_h_to_4h.weight",
        torch.ones((2, 2), dtype=torch.float32),
    )


def test_oracle_rejects_scope_and_tensor_outside_declared_layer_count(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "num_hidden_layers": 1}),
        encoding="utf-8",
    )
    with pytest.raises(TransformationOracleError, match="declared layer count"):
        build_transformation_oracle(
            checkpoint,
            edit_type="quant_rtn",
            parameters={"bits": 4, "group_size": 2},
            scope="ffn@layer=999",
        )
    oracle = build_transformation_oracle(
        checkpoint,
        edit_type="quant_rtn",
        parameters={"bits": 4, "group_size": 2},
        scope="ffn",
    )
    assert not oracle.is_target(
        "model.layers.999.mlp.up_proj.weight",
        torch.ones((2, 2), dtype=torch.float32),
    )


def test_direct_replay_and_oracle_reject_unapproved_float8_storage(
    tmp_path: Path,
) -> None:
    dtype = getattr(torch, "float8_e8m0fnu", None)
    if dtype is None:  # pragma: no cover - older Torch builds
        pytest.skip("current Torch has no float8_e8m0fnu")
    tensor = torch.zeros((2, 2), dtype=dtype)
    parameters = {"bits": 4, "group_size": 2}
    oracle = _oracle(tmp_path, edit_type="quant_rtn", parameters=parameters)

    with pytest.raises(TransformationContractError, match="float16, bfloat16"):
        replay_transformation_tensor(
            tensor, edit_type="quant_rtn", parameters=parameters
        )
    with pytest.raises(TransformationOracleError, match="float16, bfloat16"):
        oracle.replay_tensor(tensor)


@pytest.mark.parametrize(
    ("edit_type", "parameters"),
    (
        ("synthetic_lowrank_delta", {"rank": 33, "scale": 1.0}),
        ("synthetic_dense_update", {"step_size": 0.001, "iterations": 17}),
    ),
)
def test_oracle_rejects_unbounded_transform_work(
    tmp_path: Path, edit_type: str, parameters: dict[str, object]
) -> None:
    with pytest.raises(TransformationOracleError, match="must not exceed"):
        _oracle(tmp_path, edit_type=edit_type, parameters=parameters)


def test_oracle_has_no_generator_or_shared_target_import_boundary() -> None:
    source = Path(
        "scripts/evidence_packs/python/editing/transformation_oracle.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    forbidden = {
        "scripts.evidence_packs.python.editing.streaming_transform",
        "scripts.evidence_packs.python.editing.transformation_contract",
        "invarlock.pruning_contract",
        "streaming_transform",
        "transformation_contract",
        "pruning_contract",
    }
    assert not forbidden.intersection(imports)


def test_raw_transform_validator_does_not_import_the_materializer_or_contract() -> None:
    source = Path(
        "scripts/evidence_packs/python/editing/validate_artifact.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not {
        "streaming_transform",
        "transformation_contract",
        "editing.streaming_transform",
        "editing.transformation_contract",
    }.intersection(imported_modules)
