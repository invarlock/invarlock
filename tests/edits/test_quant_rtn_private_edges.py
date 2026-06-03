import builtins

import pytest
import torch

from invarlock.core.api import EditRuntime
from invarlock.edits.quant_rtn import QuantTargetSelector, RTNQuantEdit


def test_quant_rtn_supported_module_handles_missing_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.pytorch_utils":
            raise ImportError("transformers missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert QuantTargetSelector._is_supported_module(torch.nn.Embedding(2, 2)) is False


def test_quant_rtn_private_selection_and_runtime_edges() -> None:
    edit = RTNQuantEdit(scope="attn")
    assert (
        edit._include_runtime_debug(EditRuntime(include_runtime_debug=False)) is False
    )
    assert edit._layer_label_from_module_name("transformer.h") is None
    assert edit._layer_label_from_module_name("model.layers.foo.mlp") is None
    assert RTNQuantEdit._tied_group_lookup([["single"], ["a", "b"]]) == {
        "a": "a|b",
        "b": "a|b",
    }

    unsupported = object.__new__(RTNQuantEdit)
    unsupported.scope = "unsupported"
    unsupported.module_selectors = {}
    assert unsupported._has_matching_module_name(["mlp.c_fc"]) is False
