from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


class _OpaqueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.adapter_attn = nn.Linear(4, 4, bias=False)
        self.adapter_mlp = nn.Linear(4, 4, bias=False)

    def named_modules(self, *args, **kwargs):
        yield "", self


class _OpaqueAdapter:
    def describe(self, model) -> dict[str, int]:
        return {"n_layer": 1}

    def get_layer_modules(self, model, index: int):
        assert index == 0
        return {
            "self_attn.o_proj": model.adapter_attn,
            "mlp.down_proj": model.adapter_mlp,
        }


class _GptqQuantLinear(nn.Module):
    __module__ = "gptqmodel.nn_modules.qlinear"

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4))


class _GptqOpaqueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.gptq_attn = _GptqQuantLinear()
        self.gptq_mlp = _GptqQuantLinear()

    def named_modules(self, *args, **kwargs):
        yield "", self


class _GptqAdapter:
    def describe(self, model) -> dict[str, int]:
        return {"n_layer": 1}

    def get_layer_modules(self, model, index: int):
        assert index == 0
        return {
            "self_attn.o_proj": model.gptq_attn,
            "mlp.down_proj": model.gptq_mlp,
        }


class _TensorWeightModule(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight


class _SingleModuleModel(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.proj = module


def test_spectral_guard_uses_adapter_modules_when_model_graph_is_opaque() -> None:
    model = _OpaqueModel()
    guard = SpectralGuard(scope="all", correction_enabled=False)

    result = guard.prepare(model, _OpaqueAdapter(), None, {})

    assert result["ready"] is True
    assert "adapter.layers.0.self_attn.o_proj" in guard.baseline_sigmas
    assert "adapter.layers.0.mlp.down_proj" in guard.baseline_sigmas
    assert guard.module_family_map["adapter.layers.0.self_attn.o_proj"] == "attn"
    assert guard.module_family_map["adapter.layers.0.mlp.down_proj"] == "ffn"


def test_spectral_guard_does_not_block_on_unmeasurable_quantized_weight() -> None:
    dense_model = _SingleModuleModel(_TensorWeightModule(torch.eye(4)))
    guard = SpectralGuard(scope="all", correction_enabled=False)
    guard.prepare(dense_model, adapter=None, calib=None, policy={})

    quantized_model = _SingleModuleModel(
        _TensorWeightModule(torch.ones(4, 4, dtype=torch.int8))
    )

    result = guard.finalize(quantized_model)

    assert result["passed"] is True
    assert result["violations"] == []
    assert result["final_metrics"] == {}
    assert any(
        item["kind"] == "spectral_sigma_unavailable_quantized_weight"
        for item in result["diagnostics"]
    )


def test_spectral_guard_classifies_gptq_adapter_modules_by_projection_role() -> None:
    model = _GptqOpaqueModel()
    guard = SpectralGuard(scope="all", correction_enabled=False)

    result = guard.prepare(model, _GptqAdapter(), None, {})

    assert result["ready"] is True
    assert guard.module_family_map["adapter.layers.0.self_attn.o_proj"] == "attn"
    assert guard.module_family_map["adapter.layers.0.mlp.down_proj"] == "ffn"
