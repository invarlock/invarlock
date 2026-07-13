from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from invarlock.adapters.gptq_checkpoint_validation import (
    GPTQCheckpointValidationError,
    validate_gptq_checkpoint_bindings,
)


class _QuantLinear:
    __module__ = "gptqmodel.nn_modules.qlinear.test"
    bias = None


class _RuntimeModel:
    def __init__(self, *, loaded_keys: set[str]) -> None:
        self._loaded_keys = loaded_keys

    def named_modules(self):
        return (("model.layers.0.self_attn.q_proj", _QuantLinear()),)

    def state_dict(self):
        return {key: torch.zeros(1) for key in self._loaded_keys}


def _model(root: Path, *, loaded_keys: set[str] | None = None):
    return SimpleNamespace(
        model_local_path=str(root),
        model=_RuntimeModel(loaded_keys=loaded_keys or {"model.embed_tokens.weight"}),
    )


def _write_checkpoint(root: Path, *, ignored_name: str, ignored_value: float) -> None:
    root.mkdir()
    save_file(
        {
            "model.embed_tokens.weight": torch.ones(2, 2),
            ignored_name: torch.tensor([ignored_value]),
        },
        root / "model.safetensors",
    )


def test_exact_unbound_zero_qlinear_bias_is_accepted(tmp_path: Path) -> None:
    key = "model.layers.0.self_attn.q_proj.bias"
    _write_checkpoint(tmp_path / "subject", ignored_name=key, ignored_value=0.0)

    ignored = validate_gptq_checkpoint_bindings(_model(tmp_path / "subject"))

    assert ignored == (key,)


@pytest.mark.parametrize("value", [1.0, float("nan"), float("inf")])
def test_nonzero_or_nonfinite_ignored_bias_is_rejected(
    tmp_path: Path, value: float
) -> None:
    key = "model.layers.0.self_attn.q_proj.bias"
    _write_checkpoint(tmp_path / "subject", ignored_name=key, ignored_value=value)

    with pytest.raises(
        GPTQCheckpointValidationError,
        match="must be a non-empty finite all-zero floating tensor",
    ):
        validate_gptq_checkpoint_bindings(_model(tmp_path / "subject"))


def test_ignored_tensor_outside_exact_qlinear_bias_set_is_rejected(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path / "subject",
        ignored_name="model.layers.0.self_attn.q_proj.extra",
        ignored_value=0.0,
    )

    with pytest.raises(GPTQCheckpointValidationError, match="outside the exact"):
        validate_gptq_checkpoint_bindings(_model(tmp_path / "subject"))


def test_shard_index_must_exactly_bind_tensor_locations(tmp_path: Path) -> None:
    root = tmp_path / "subject"
    root.mkdir()
    save_file(
        {"model.embed_tokens.weight": torch.ones(2, 2)},
        root / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"model.layers.0.self_attn.q_proj.bias": torch.zeros(1)},
        root / "model-00002-of-00002.safetensors",
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
                    "model.layers.0.self_attn.q_proj.bias": (
                        "model-00001-of-00002.safetensors"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GPTQCheckpointValidationError, match="wrong shard"):
        validate_gptq_checkpoint_bindings(_model(root))


def test_resolved_checkpoint_path_is_required() -> None:
    with pytest.raises(GPTQCheckpointValidationError, match="resolved checkpoint path"):
        validate_gptq_checkpoint_bindings(SimpleNamespace())
