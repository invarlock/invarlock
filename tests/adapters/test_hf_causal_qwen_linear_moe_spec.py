from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from invarlock.adapters.hf_causal_specs import _Qwen35LinearMoEDecoderSpec


def _linear(in_features: int = 4, out_features: int = 4) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False)


def _norm() -> nn.LayerNorm:
    return nn.LayerNorm(4)


def _qwen_moe_layer(*, shared: bool = True) -> nn.Module:
    layer = nn.Module()
    layer.linear_attn = nn.Module()
    layer.linear_attn.in_proj_qkv = _linear(4, 12)
    layer.linear_attn.out_proj = _linear()
    layer.mlp = nn.Module()
    layer.mlp.gate = _linear(4, 256)
    layer.mlp.experts = nn.Module()
    if shared:
        layer.mlp.shared_expert = nn.Module()
        layer.mlp.shared_expert_gate = _linear(4, 1)
    layer.input_layernorm = _norm()
    layer.post_attention_layernorm = _norm()
    return layer


def test_qwen35_linear_moe_decoder_spec_matches_and_exposes_modules() -> None:
    layer = _qwen_moe_layer()
    spec = _Qwen35LinearMoEDecoderSpec()

    assert spec.matches(object(), object(), [layer]) is True
    assert spec.matches(object(), object(), []) is False
    assert spec.infer_mlp_dim(layer, SimpleNamespace(intermediate_size=96), 4) == 96

    modules = spec.layer_modules(object(), layer)
    assert modules["linear_attn.in_proj_qkv"] is layer.linear_attn.in_proj_qkv
    assert modules["mlp.router"] is layer.mlp.gate
    assert modules["mlp.experts"] is layer.mlp.experts
    assert modules["mlp.shared_expert"] is layer.mlp.shared_expert
    assert modules["mlp.shared_expert_gate"] is layer.mlp.shared_expert_gate


def test_qwen35_linear_moe_decoder_spec_covers_dimension_fallbacks() -> None:
    spec = _Qwen35LinearMoEDecoderSpec()

    layer = _qwen_moe_layer(shared=False)
    layer.mlp.experts.intermediate_size = 17
    assert spec.infer_mlp_dim(layer, SimpleNamespace(intermediate_size=0), 4) == 17

    layer = _qwen_moe_layer(shared=False)
    layer.mlp.experts.intermediate_size = -1
    layer.mlp.experts.intermediate_dim = 19
    assert spec.infer_mlp_dim(layer, SimpleNamespace(intermediate_size=0), 4) == 19

    layer = _qwen_moe_layer()
    layer.mlp.shared_expert.gate_proj = _linear(4, 23)
    assert spec.infer_mlp_dim(layer, SimpleNamespace(intermediate_size=0), 4) == 23

    layer = _qwen_moe_layer()
    del layer.mlp.experts
    layer.mlp.shared_expert.gate_proj = _linear(4, 29)
    assert spec.infer_mlp_dim(layer, SimpleNamespace(intermediate_size=0), 4) == 29


def test_qwen35_linear_moe_decoder_spec_optional_modules_and_tying() -> None:
    layer = _qwen_moe_layer(shared=False)
    spec = _Qwen35LinearMoEDecoderSpec()

    modules = spec.layer_modules(object(), layer)
    assert "mlp.shared_expert" not in modules
    assert "mlp.shared_expert_gate" not in modules

    base = nn.Module()
    base.embed_tokens = nn.Embedding(8, 4)
    model = nn.Module()
    model.model = base
    model.lm_head = nn.Linear(4, 8, bias=False)
    model.lm_head.weight = base.embed_tokens.weight
    assert spec.tying_map(model, base) == {
        "lm_head.weight": "model.embed_tokens.weight"
    }
