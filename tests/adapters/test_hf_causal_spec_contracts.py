from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from invarlock.adapters import hf_causal as hf_causal_mod
from invarlock.adapters.hf_causal import HF_Causal_Adapter
from invarlock.adapters.hf_causal_specs import (
    _CausalSpec,
    _coerce_config_int,
    _DenseDecoderSpec,
    _FalconDecoderSpec,
    _first_item,
    _GlmDecoderSpec,
    _GPT2LikeDecoderSpec,
    _GptOssMoEDecoderSpec,
    _has_set_attr,
    _layer_list,
    _mixtral_tensorized_moe_parts,
    _MoEDecoderSpec,
    _NeoXDecoderSpec,
    _OptDecoderSpec,
    _PhiDecoderSpec,
    _Qwen35LinearDecoderSpec,
    _safe_model_device,
    _safe_total_params,
    _shape_ints,
    _weight_shape_dim,
)
from invarlock.adapters.hf_loading import HFLoaderStrategy
from invarlock.core.exceptions import AdapterError, DependencyError, ModelLoadError


def _linear(in_features: int = 4, out_features: int = 4) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False)


def _norm() -> nn.LayerNorm:
    return nn.LayerNorm(4)


def _dense_layer(
    *,
    pre_norm: str = "input_layernorm",
    post_norm: str = "post_attention_layernorm",
) -> nn.Module:
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = _linear()
    layer.self_attn.k_proj = _linear()
    layer.self_attn.v_proj = _linear()
    layer.self_attn.o_proj = _linear()
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = _linear(4, 8)
    layer.mlp.up_proj = _linear(4, 8)
    layer.mlp.down_proj = _linear(8, 4)
    setattr(layer, pre_norm, _norm())
    setattr(layer, post_norm, _norm())
    return layer


def _gate_up_layer() -> nn.Module:
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.q_proj = _linear()
    layer.self_attn.k_proj = _linear()
    layer.self_attn.v_proj = _linear()
    layer.self_attn.o_proj = _linear()
    layer.mlp = nn.Module()
    layer.mlp.gate_up_proj = _linear(4, 16)
    layer.mlp.down_proj = _linear(8, 4)
    layer.input_layernorm = _norm()
    layer.post_attention_layernorm = _norm()
    return layer


def _base_model_with_layers(layers: list[nn.Module]) -> nn.Module:
    base = nn.Module()
    base.layers = nn.ModuleList(layers)
    base.embed_tokens = nn.Embedding(16, 4)
    return base


def _causal_model(base: nn.Module, config: object | None = None) -> nn.Module:
    model = nn.Module()
    model.model = base
    model.config = config or SimpleNamespace(
        model_type="unit",
        num_attention_heads=2,
        hidden_size=4,
        vocab_size=16,
        intermediate_size=8,
    )
    model.lm_head = _linear(4, 16)
    if hasattr(base, "embed_tokens"):
        model.lm_head.weight = base.embed_tokens.weight
    return model


def test_causal_spec_base_and_helpers_cover_fallback_paths() -> None:
    base_spec = _CausalSpec()
    with pytest.raises(NotImplementedError):
        base_spec.matches(object(), object(), [])
    with pytest.raises(NotImplementedError):
        base_spec.infer_mlp_dim(object(), object(), 4)
    with pytest.raises(NotImplementedError):
        base_spec.layer_modules(object(), object())
    assert base_spec.tying_map(object(), object()) == {}

    class IndexBroken:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _idx: int) -> object:
            raise KeyError("bad index")

        def __iter__(self):
            yield "fallback"

    class NotIterable:
        def __iter__(self):
            raise TypeError("not iterable")

    assert _first_item(IndexBroken()) == "fallback"
    assert _first_item(NotIterable()) is None

    module = nn.Module()
    module.register_parameter("weight", nn.Parameter(torch.ones(1)))
    module.register_buffer("cache", torch.ones(1))
    assert _has_set_attr(SimpleNamespace(x=1), "x") is True
    assert _has_set_attr(module, "weight") is True
    assert _has_set_attr(module, "cache") is True
    assert _has_set_attr(module, "missing") is False

    assert _weight_shape_dim(SimpleNamespace(weight=torch.empty(2, 3)), 1) == 3
    assert _weight_shape_dim(SimpleNamespace(), 0) is None
    assert (
        _weight_shape_dim(SimpleNamespace(weight=SimpleNamespace(shape=())), 2) is None
    )
    assert _shape_ints(torch.empty(2, 3)) == (2, 3)
    assert _shape_ints(object()) is None

    class BadShape:
        shape = ("bad",)

    assert _shape_ints(BadShape()) is None
    assert _coerce_config_int(True) is None
    assert _coerce_config_int(3.0) == 3
    assert _coerce_config_int(" 7 ") == 7
    assert _coerce_config_int("7.5") is None

    class LayerProxy:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int) -> str:
            return f"layer-{idx}"

        def __iter__(self):
            raise TypeError("not listable")

    assert _layer_list(LayerProxy()) == ["layer-0", "layer-1"]


def test_causal_safe_metadata_helpers_handle_success_and_failures() -> None:
    model = nn.Linear(2, 2)
    assert _safe_total_params(model) == 6
    assert _safe_model_device(model) == torch.device("cpu")

    class BadParams:
        def parameters(self):
            raise RuntimeError("parameter scan failed")

    assert _safe_total_params(BadParams()) == 0
    assert _safe_model_device(BadParams()) == torch.device("cpu")


def test_dense_spec_alias_norms_and_tying_paths() -> None:
    layer = _dense_layer(
        pre_norm="pre_feedforward_layernorm",
        post_norm="pre_attention_layernorm",
    )
    spec = _DenseDecoderSpec()
    base = _base_model_with_layers([layer])
    model = _causal_model(base)

    assert spec.matches(model, base, base.layers) is True
    assert spec.matches(model, base, []) is False
    assert spec.infer_mlp_dim(layer, SimpleNamespace(intermediate_size=0), 4) == 8

    modules = spec.layer_modules(model, layer)
    assert modules["input_layernorm"] is layer.pre_feedforward_layernorm
    assert modules["pre_feedforward_layernorm"] is layer.pre_feedforward_layernorm
    assert modules["post_attention_layernorm"] is layer.pre_attention_layernorm
    assert modules["pre_attention_layernorm"] is layer.pre_attention_layernorm

    canonical_modules = spec.layer_modules(model, _dense_layer())
    assert "pre_feedforward_layernorm" not in canonical_modules
    assert "pre_attention_layernorm" not in canonical_modules
    no_norm_layer = _dense_layer()
    del no_norm_layer.input_layernorm
    del no_norm_layer.post_attention_layernorm
    no_norm_modules = spec.layer_modules(model, no_norm_layer)
    assert "input_layernorm" not in no_norm_modules
    assert "post_attention_layernorm" not in no_norm_modules

    no_shape_layer = _dense_layer()
    no_shape_layer.mlp.gate_proj = nn.Module()
    assert (
        spec.infer_mlp_dim(no_shape_layer, SimpleNamespace(intermediate_size=11), 4)
        == 11
    )

    outer = nn.Module()
    outer.language_model = base
    model.model = outer
    assert spec.tying_map(model, base)["lm_head.weight"] == (
        "model.language_model.embed_tokens.weight"
    )

    model.lm_head = _linear(4, 16)
    assert spec.tying_map(model, base) == {}


def test_phi_glm_and_qwen_specs_cover_metadata_fallbacks() -> None:
    phi_layer = nn.Module()
    phi_layer.self_attn = nn.Module()
    phi_layer.self_attn.qkv_proj = SimpleNamespace(
        weight=SimpleNamespace(shape=(16, 4))
    )
    phi_layer.self_attn.o_proj = _linear()
    phi_layer.mlp = nn.Module()
    phi_layer.mlp.gate_up_proj = SimpleNamespace(weight=SimpleNamespace(shape=(16, 4)))
    phi_layer.mlp.down_proj = SimpleNamespace()
    phi_layer.input_layernorm = _norm()
    phi_layer.post_attention_layernorm = _norm()

    phi_spec = _PhiDecoderSpec()
    assert phi_spec.matches(object(), object(), [phi_layer]) is True
    assert phi_spec.matches(object(), object(), []) is False
    assert (
        phi_spec.infer_mlp_dim(phi_layer, SimpleNamespace(intermediate_size=1), 4) == 8
    )
    phi_layer.mlp.gate_up_proj = SimpleNamespace()
    assert (
        phi_spec.infer_mlp_dim(phi_layer, SimpleNamespace(intermediate_size=6), 4) == 6
    )

    glm_layer = _gate_up_layer()
    glm_layer.mlp.down_proj = nn.Module()
    glm_spec = _GlmDecoderSpec()
    assert glm_spec.matches(object(), object(), [glm_layer]) is True
    assert glm_spec.matches(object(), object(), []) is False
    assert (
        glm_spec.infer_mlp_dim(glm_layer, SimpleNamespace(intermediate_size=9), 4) == 9
    )
    base = _base_model_with_layers([glm_layer])
    model = _causal_model(base)
    assert (
        glm_spec.tying_map(model, base)["lm_head.weight"] == "model.embed_tokens.weight"
    )
    assert "self_attn.q_proj" in glm_spec.layer_modules(model, glm_layer)
    model.lm_head = _linear(4, 16)
    assert glm_spec.tying_map(model, base) == {}

    qwen_layer = _dense_layer()
    qwen_layer.linear_attn = nn.Module()
    qwen_layer.linear_attn.in_proj_qkv = _linear(4, 12)
    qwen_layer.linear_attn.out_proj = _linear()
    qwen_spec = _Qwen35LinearDecoderSpec()
    assert qwen_spec.matches(object(), object(), [qwen_layer]) is True
    assert qwen_spec.matches(object(), object(), []) is False
    assert qwen_spec.layer_modules(object(), qwen_layer)["linear_attn.out_proj"] is (
        qwen_layer.linear_attn.out_proj
    )


def test_moe_specs_cover_legacy_tensorized_and_error_paths() -> None:
    legacy_layer = _dense_layer()
    legacy_layer.block_sparse_moe = nn.Module()
    expert = nn.Module()
    expert.w1 = _linear(4, 8)
    expert.w2 = _linear(8, 4)
    legacy_layer.block_sparse_moe.experts = nn.ModuleList([expert])

    spec = _MoEDecoderSpec()
    assert spec.matches(object(), object(), [legacy_layer]) is True
    assert spec.matches(object(), object(), []) is False
    assert (
        spec.infer_mlp_dim(legacy_layer, SimpleNamespace(intermediate_size=1), 4) == 8
    )
    assert spec.layer_modules(object(), legacy_layer)["mlp.up_proj"] is expert.w1

    no_shape_expert_layer = _dense_layer()
    no_shape_expert_layer.block_sparse_moe = nn.Module()
    no_shape_expert = nn.Module()
    no_shape_expert.w1 = nn.Module()
    no_shape_expert.w2 = _linear(8, 4)
    no_shape_expert_layer.block_sparse_moe.experts = nn.ModuleList([no_shape_expert])
    assert (
        spec.infer_mlp_dim(
            no_shape_expert_layer, SimpleNamespace(intermediate_size=5), 4
        )
        == 5
    )

    missing_expert_layer = _dense_layer()
    missing_expert_layer.block_sparse_moe = nn.Module()
    missing_expert_layer.block_sparse_moe.experts = []
    with pytest.raises(AdapterError, match="MoE layer missing experts"):
        spec.layer_modules(object(), missing_expert_layer)

    tensor_layer = _dense_layer()
    tensor_layer.mlp = nn.Module()
    tensor_layer.mlp.gate = nn.Module()
    tensor_layer.mlp.gate.weight = nn.Parameter(torch.empty(2, 4))
    tensor_layer.mlp.experts = nn.Module()
    tensor_layer.mlp.experts.gate_up_proj = nn.Parameter(torch.empty(2, 16, 4))
    tensor_layer.mlp.experts.down_proj = nn.Parameter(torch.empty(2, 4, 8))
    assert _mixtral_tensorized_moe_parts(tensor_layer) == (
        tensor_layer.mlp.gate,
        tensor_layer.mlp.experts,
    )
    assert spec.matches(object(), object(), [tensor_layer]) is True
    assert (
        spec.infer_mlp_dim(tensor_layer, SimpleNamespace(intermediate_size=1), 4) == 8
    )
    assert spec.layer_modules(object(), tensor_layer)["mlp.experts"] is (
        tensor_layer.mlp.experts
    )

    tensor_layer.mlp.experts.intermediate_size = 7
    assert (
        spec.infer_mlp_dim(tensor_layer, SimpleNamespace(intermediate_size=1), 4) == 7
    )
    del tensor_layer.mlp.experts.intermediate_size
    tensor_layer.mlp.experts.intermediate_dim = 6
    assert (
        spec.infer_mlp_dim(tensor_layer, SimpleNamespace(intermediate_size=1), 4) == 6
    )

    shape_fallback_layer = SimpleNamespace(
        mlp=SimpleNamespace(
            gate=SimpleNamespace(weight=object()),
            experts=SimpleNamespace(gate_up_proj=object(), down_proj=object()),
        )
    )
    assert (
        spec.infer_mlp_dim(
            shape_fallback_layer, SimpleNamespace(intermediate_size=5), 4
        )
        == 5
    )

    assert _mixtral_tensorized_moe_parts(SimpleNamespace()) == (None, None)
    assert _mixtral_tensorized_moe_parts(SimpleNamespace(mlp=SimpleNamespace())) == (
        None,
        None,
    )
    no_gate_weight = SimpleNamespace(
        mlp=SimpleNamespace(
            gate=nn.Module(),
            experts=SimpleNamespace(gate_up_proj=object(), down_proj=object()),
        )
    )
    assert _mixtral_tensorized_moe_parts(no_gate_weight) == (None, None)
    missing_expert_projection = SimpleNamespace(
        mlp=SimpleNamespace(
            gate=SimpleNamespace(weight=object()),
            experts=SimpleNamespace(gate_up_proj=object()),
        )
    )
    assert _mixtral_tensorized_moe_parts(missing_expert_projection) == (None, None)


def test_gpt_oss_neox_falcon_opt_and_gpt2_specs_cover_fallbacks() -> None:
    oss_layer = _dense_layer()
    oss_layer.mlp = nn.Module()
    oss_layer.mlp.router = nn.Module()
    oss_layer.mlp.router.weight = nn.Parameter(torch.empty(2, 4))
    oss_layer.mlp.experts = nn.Module()
    oss_layer.mlp.experts.gate_up_proj = nn.Parameter(torch.empty(2, 4, 16))
    oss_layer.mlp.experts.down_proj = nn.Parameter(torch.empty(2, 8, 4))
    oss_spec = _GptOssMoEDecoderSpec()
    assert oss_spec.matches(object(), object(), [oss_layer]) is True
    assert oss_spec.matches(object(), object(), []) is False
    assert (
        oss_spec.infer_mlp_dim(oss_layer, SimpleNamespace(intermediate_size=1), 4) == 8
    )
    oss_layer.mlp.experts.intermediate_size = 9
    assert (
        oss_spec.infer_mlp_dim(oss_layer, SimpleNamespace(intermediate_size=1), 4) == 9
    )
    oss_layer.mlp.experts = nn.Module()
    assert (
        oss_spec.infer_mlp_dim(oss_layer, SimpleNamespace(intermediate_size=6), 4) == 6
    )
    oss_layer.mlp.experts = None
    assert (
        oss_spec.infer_mlp_dim(oss_layer, SimpleNamespace(intermediate_size=5), 4) == 5
    )

    neox_layer = nn.Module()
    neox_layer.attention = nn.Module()
    neox_layer.attention.query_key_value = _linear(4, 12)
    neox_layer.attention.dense = _linear()
    neox_layer.mlp = nn.Module()
    neox_layer.mlp.dense_h_to_4h = _linear(4, 8)
    neox_layer.mlp.dense_4h_to_h = _linear(8, 4)
    neox_layer.input_layernorm = _norm()
    neox_layer.post_attention_layernorm = _norm()
    neox_spec = _NeoXDecoderSpec()
    assert neox_spec.matches(object(), object(), [neox_layer]) is True
    assert neox_spec.matches(object(), object(), []) is False
    assert (
        neox_spec.infer_mlp_dim(neox_layer, SimpleNamespace(intermediate_size=1), 4)
        == 8
    )
    assert neox_spec.layer_modules(object(), neox_layer)["attention.dense"] is (
        neox_layer.attention.dense
    )
    no_shape_neox_layer = nn.Module()
    no_shape_neox_layer.mlp = nn.Module()
    no_shape_neox_layer.mlp.dense_h_to_4h = nn.Module()
    assert (
        neox_spec.infer_mlp_dim(
            no_shape_neox_layer, SimpleNamespace(intermediate_size=6), 4
        )
        == 6
    )
    neox_base = nn.Module()
    neox_base.embed_in = nn.Embedding(16, 4)
    neox_model = nn.Module()
    neox_model.embed_out = _linear(4, 16)
    neox_model.embed_out.weight = neox_base.embed_in.weight
    assert neox_spec.tying_map(neox_model, neox_base) == {
        "embed_out.weight": "gpt_neox.embed_in.weight"
    }
    neox_model.embed_out = _linear(4, 16)
    assert neox_spec.tying_map(neox_model, neox_base) == {}

    falcon_layer = nn.Module()
    falcon_layer.self_attention = nn.Module()
    falcon_layer.self_attention.query_key_value = _linear(4, 12)
    falcon_layer.self_attention.dense = _linear()
    falcon_layer.mlp = nn.Module()
    falcon_layer.mlp.dense_h_to_4h = _linear(4, 8)
    falcon_layer.mlp.dense_4h_to_h = _linear(8, 4)
    falcon_layer.input_layernorm = _norm()
    falcon_spec = _FalconDecoderSpec()
    assert falcon_spec.matches(object(), object(), [falcon_layer]) is True
    assert falcon_spec.matches(object(), object(), []) is False
    assert (
        falcon_spec.infer_mlp_dim(falcon_layer, SimpleNamespace(hidden_size=4), 4) == 8
    )
    assert falcon_spec.layer_modules(object(), falcon_layer)[
        "self_attention.dense"
    ] is (falcon_layer.self_attention.dense)
    no_shape_falcon_layer = nn.Module()
    no_shape_falcon_layer.mlp = nn.Module()
    no_shape_falcon_layer.mlp.dense_h_to_4h = nn.Module()
    assert (
        falcon_spec.infer_mlp_dim(
            no_shape_falcon_layer, SimpleNamespace(hidden_size=4), 4
        )
        == 16
    )
    falcon_base = nn.Module()
    falcon_base.word_embeddings = nn.Embedding(16, 4)
    falcon_model = _causal_model(falcon_base)
    falcon_model.lm_head.weight = falcon_base.word_embeddings.weight
    assert falcon_spec.tying_map(falcon_model, falcon_base) == {
        "lm_head.weight": "transformer.word_embeddings.weight"
    }
    falcon_model.lm_head = _linear(4, 16)
    assert falcon_spec.tying_map(falcon_model, falcon_base) == {}

    opt_layer = nn.Module()
    opt_layer.self_attn = nn.Module()
    opt_layer.self_attn.q_proj = _linear()
    opt_layer.self_attn.k_proj = _linear()
    opt_layer.self_attn.v_proj = _linear()
    opt_layer.self_attn.out_proj = _linear()
    opt_layer.self_attn_layer_norm = _norm()
    opt_layer.final_layer_norm = _norm()
    opt_layer.fc1 = _linear(4, 8)
    opt_layer.fc2 = _linear(8, 4)
    opt_spec = _OptDecoderSpec()
    assert opt_spec.matches(object(), object(), [opt_layer]) is True
    assert opt_spec.matches(object(), object(), []) is False
    assert opt_spec.infer_mlp_dim(opt_layer, SimpleNamespace(ffn_dim=1), 4) == 8
    no_shape_opt_layer = nn.Module()
    no_shape_opt_layer.fc1 = nn.Module()
    assert (
        opt_spec.infer_mlp_dim(no_shape_opt_layer, SimpleNamespace(ffn_dim=6), 4) == 6
    )
    decoder = nn.Module()
    decoder.embed_tokens = nn.Embedding(16, 4)
    opt_model = nn.Module()
    opt_model.model = nn.Module()
    opt_model.model.decoder = decoder
    opt_model.lm_head = _linear(4, 16)
    opt_model.lm_head.weight = decoder.embed_tokens.weight
    assert opt_spec.tying_map(opt_model, object()) == {
        "lm_head.weight": "model.decoder.embed_tokens.weight"
    }
    opt_model.lm_head = _linear(4, 16)
    assert opt_spec.tying_map(opt_model, object()) == {}

    gpt2_layer = nn.Module()
    gpt2_layer.attn = nn.Module()
    gpt2_layer.attn.c_attn = _linear(4, 12)
    gpt2_layer.attn.c_proj = _linear()
    gpt2_layer.mlp = nn.Module()
    gpt2_layer.mlp.c_fc = SimpleNamespace(nf="13", weight=torch.empty(8, 4))
    gpt2_layer.mlp.c_proj = _linear(8, 4)
    gpt2_layer.ln_1 = _norm()
    gpt2_layer.ln_2 = _norm()
    gpt2_spec = _GPT2LikeDecoderSpec()
    assert gpt2_spec.matches(object(), object(), [gpt2_layer]) is True
    assert gpt2_spec.matches(object(), object(), []) is False
    assert gpt2_spec.infer_mlp_dim(gpt2_layer, SimpleNamespace(n_inner=1), 4) == 13
    gpt2_layer.mlp.c_fc.nf = None
    assert gpt2_spec.infer_mlp_dim(gpt2_layer, SimpleNamespace(n_inner=1), 4) == 8
    gpt2_layer.mlp.c_fc = SimpleNamespace()
    assert gpt2_spec.infer_mlp_dim(gpt2_layer, SimpleNamespace(n_inner=6), 4) == 6
    gpt2_layer.mlp.c_fc = None
    assert gpt2_spec.infer_mlp_dim(gpt2_layer, SimpleNamespace(n_inner=5), 4) == 5


def test_adapter_unwrap_error_and_text_config_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HF_Causal_Adapter()

    decoder = SimpleNamespace(layers=[_dense_layer()])
    assert adapter._unwrap(SimpleNamespace(model=SimpleNamespace(decoder=decoder)))[
        0
    ] is (decoder)

    language_model = SimpleNamespace(layers=[_dense_layer()])
    decoder_without_layers = SimpleNamespace()
    assert (
        adapter._unwrap(
            SimpleNamespace(
                model=SimpleNamespace(
                    decoder=decoder_without_layers,
                    language_model=language_model,
                )
            )
        )[0]
        is language_model
    )
    assert (
        adapter._unwrap(
            SimpleNamespace(model=SimpleNamespace(language_model=language_model))
        )[0]
        is language_model
    )

    model_layers = SimpleNamespace(layers=[_dense_layer()])
    assert (
        adapter._unwrap(
            SimpleNamespace(
                model=SimpleNamespace(
                    language_model=SimpleNamespace(),
                    layers=model_layers.layers,
                )
            )
        )[1]
        is model_layers.layers
    )
    assert adapter._unwrap(SimpleNamespace(model=model_layers))[0] is model_layers
    neox = SimpleNamespace(layers=[_dense_layer()])
    assert adapter._unwrap(SimpleNamespace(gpt_neox=neox))[0] is neox
    direct_layers = SimpleNamespace(layers=[_dense_layer()])
    assert adapter._unwrap(direct_layers)[0] is direct_layers
    direct_h = SimpleNamespace(h=[_dense_layer()])
    assert adapter._unwrap(direct_h)[1] == direct_h.h

    with pytest.raises(AdapterError, match="unrecognized HF causal LM structure"):
        adapter._unwrap(SimpleNamespace())

    config = SimpleNamespace(
        model_type="text",
        text_config=SimpleNamespace(num_attention_heads="2", hidden_size="4"),
        vocab_size="16",
    )
    base = _base_model_with_layers([_dense_layer()])
    description = adapter.describe(_causal_model(base, config=config))
    assert description["n_heads"] == 2
    assert description["hidden_size"] == 4
    assert description["vocab_size"] == 16

    missing_config_model = _causal_model(base)
    missing_config_model.config = None
    with pytest.raises(AdapterError, match="missing HuggingFace config"):
        adapter.describe(missing_config_model)
    with pytest.raises(AdapterError, match="missing head/hidden size"):
        adapter.describe(_causal_model(base, config=SimpleNamespace(model_type="bad")))

    class AttributeFailSpec:
        spec_name = "attribute-fail"

        def matches(self, model, base, layers):  # noqa: ANN001
            raise AttributeError("probe failed")

    monkeypatch.setattr(hf_causal_mod, "_SPECS", [AttributeFailSpec()])
    assert adapter.can_handle(_causal_model(base)) is False
    with pytest.raises(AdapterError, match="no matching HF causal adapter spec"):
        adapter._select_spec(_causal_model(base), base, base.layers)


def test_adapter_load_model_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyAdapter(HF_Causal_Adapter):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        def _load_pretrained_model(self, loader, model_id, **kwargs):  # noqa: ANN001
            self.calls.append(str(loader))
            if len(self.calls) == 1:
                raise OSError("primary failed")
            return {"model_id": model_id, "loader": loader, "kwargs": kwargs}

        def _safe_to_device(self, model, device):  # noqa: ANN001
            return {"model": model, "device": device}

    strategies = iter(
        (
            HFLoaderStrategy("causal", "direct_submodule", "primary", "primary"),
            HFLoaderStrategy("causal", "auto", "fallback", "fallback"),
        )
    )
    monkeypatch.setattr(
        hf_causal_mod,
        "resolve_core_loader_strategy",
        lambda *args, **kwargs: next(strategies),
    )

    result = DummyAdapter().load_model("demo/model", device="cpu")
    assert result["device"] == "cpu"
    assert result["model"]["loader"] == "fallback"


def test_adapter_load_model_direct_submodule_retry_and_dependency_reraise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyAdapter(HF_Causal_Adapter):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        def _load_pretrained_model(self, loader, model_id, **kwargs):  # noqa: ANN001
            self.calls.append(str(loader))
            if len(self.calls) == 1:
                raise OSError("primary failed")
            return object()

        def _safe_to_device(self, model, device):  # noqa: ANN001
            return (model, device)

    strategies = iter(
        (
            HFLoaderStrategy("causal", "auto", "auto", "auto"),
            HFLoaderStrategy("causal", "direct_submodule", "direct", "direct"),
        )
    )
    monkeypatch.setattr(
        hf_causal_mod,
        "resolve_core_loader_strategy",
        lambda *args, **kwargs: next(strategies),
    )
    adapter = DummyAdapter()
    _model, device = adapter.load_model("demo/model", device="cpu")
    assert device == "cpu"
    assert adapter.calls == ["auto", "direct"]

    def _missing_dependency(*_args, **_kwargs):
        raise DependencyError(code="E203", message="missing dependency")

    monkeypatch.setattr(
        hf_causal_mod, "resolve_core_loader_strategy", _missing_dependency
    )
    with pytest.raises(DependencyError):
        HF_Causal_Adapter().load_model("demo/model")


def test_adapter_load_model_auto_failure_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingAdapter(HF_Causal_Adapter):
        def _load_pretrained_model(self, loader, model_id, **kwargs):  # noqa: ANN001
            raise OSError("auto failed")

    strategies = iter(
        (
            HFLoaderStrategy("causal", "auto", "auto", "auto"),
            HFLoaderStrategy("causal", "auto", "direct", "direct"),
        )
    )
    monkeypatch.setattr(
        hf_causal_mod,
        "resolve_core_loader_strategy",
        lambda *args, **kwargs: next(strategies),
    )

    with pytest.raises(ModelLoadError, match="MODEL-LOAD-FAILED: auto"):
        FailingAdapter().load_model("demo/model")
