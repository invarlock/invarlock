from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.adapters.hf_causal import HF_Causal_Adapter


class _Olmo2Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.v_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(4, 8, bias=False)
        self.mlp.up_proj = nn.Linear(4, 8, bias=False)
        self.mlp.down_proj = nn.Linear(8, 4, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.post_feedforward_layernorm = nn.LayerNorm(4)


class _Olmo2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Olmo2Layer(), _Olmo2Layer()])
        self.embed_tokens = nn.Embedding(16, 4)


class _Olmo2ForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Olmo2Model()
        self.config = SimpleNamespace(
            model_type="olmo2",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


class _Qwen35Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_qkv = nn.Linear(4, 12, bias=False)
        self.linear_attn.out_proj = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(4, 8, bias=False)
        self.mlp.up_proj = nn.Linear(4, 8, bias=False)
        self.mlp.down_proj = nn.Linear(8, 4, bias=False)
        self.input_layernorm = nn.LayerNorm(4)
        self.post_attention_layernorm = nn.LayerNorm(4)


class _Qwen35Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Qwen35Layer(), _Qwen35Layer()])
        self.embed_tokens = nn.Embedding(16, 4)


class _Qwen35ForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Qwen35Model()
        self.config = SimpleNamespace(
            model_type="qwen3_5_text",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


class _GptOssExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intermediate_size = 8
        self.num_experts = 2
        self.gate_up_proj = nn.Parameter(torch.empty(2, 4, 16))
        self.down_proj = nn.Parameter(torch.empty(2, 8, 4))


class _GptOssRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 4))
        self.bias = nn.Parameter(torch.empty(2))


class _GptOssMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _GptOssRouter()
        self.experts = _GptOssExperts()


class _GptOssLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.v_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)
        self.mlp = _GptOssMLP()
        self.input_layernorm = nn.LayerNorm(4)
        self.post_attention_layernorm = nn.LayerNorm(4)


class _GptOssModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_GptOssLayer(), _GptOssLayer()])
        self.embed_tokens = nn.Embedding(16, 4)


class _GptOssForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _GptOssModel()
        self.config = SimpleNamespace(
            model_type="gpt_oss",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


class _MixtralExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intermediate_dim = 8
        self.num_experts = 2
        self.gate_up_proj = nn.Parameter(torch.empty(2, 16, 4))
        self.down_proj = nn.Parameter(torch.empty(2, 4, 8))


class _MixtralGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 4))


class _MixtralMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _MixtralGate()
        self.experts = _MixtralExperts()


class _MixtralLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.v_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)
        self.mlp = _MixtralMLP()
        self.input_layernorm = nn.LayerNorm(4)
        self.post_attention_layernorm = nn.LayerNorm(4)


class _MixtralModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MixtralLayer(), _MixtralLayer()])
        self.embed_tokens = nn.Embedding(16, 4)


class _MixtralForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _MixtralModel()
        self.config = SimpleNamespace(
            model_type="mixtral",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


class _OptLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.v_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.out_proj = nn.Linear(4, 4, bias=False)
        self.self_attn_layer_norm = nn.LayerNorm(4)
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)
        self.final_layer_norm = nn.LayerNorm(4)


class _OptDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_OptLayer(), _OptLayer()])
        self.embed_tokens = nn.Embedding(16, 4)


class _OptModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _OptDecoder()


class _OptForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _OptModel()
        self.config = SimpleNamespace(
            model_type="opt",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            ffn_dim=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.decoder.embed_tokens.weight


class _NeoXAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_key_value = nn.Linear(4, 12, bias=False)
        self.dense = nn.Linear(4, 4, bias=False)


class _NeoXMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(4, 8, bias=False)
        self.dense_4h_to_h = nn.Linear(8, 4, bias=False)


class _NeoXLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.attention = _NeoXAttention()
        self.mlp = _NeoXMLP()


class _NeoXModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_NeoXLayer(), _NeoXLayer()])
        self.embed_in = nn.Embedding(16, 4)


class _NeoXForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = _NeoXModel()
        self.config = SimpleNamespace(
            model_type="gpt_neox",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.embed_out = nn.Linear(4, 16, bias=False)
        self.embed_out.weight = self.gpt_neox.embed_in.weight


class _FalconAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_key_value = nn.Linear(4, 12, bias=False)
        self.dense = nn.Linear(4, 4, bias=False)


class _FalconMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(4, 8, bias=False)
        self.dense_4h_to_h = nn.Linear(8, 4, bias=False)


class _FalconLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attention = _FalconAttention()
        self.mlp = _FalconMLP()
        self.input_layernorm = nn.LayerNorm(4)


class _FalconTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = nn.ModuleList([_FalconLayer(), _FalconLayer()])
        self.word_embeddings = nn.Embedding(16, 4)


class _FalconForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _FalconTransformer()
        self.config = SimpleNamespace(
            model_type="falcon",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.transformer.word_embeddings.weight


class _GlmSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _GlmMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 16, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class _GlmLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _GlmSelfAttention()
        self.mlp = _GlmMLP()
        self.input_layernorm = nn.LayerNorm(4)
        self.post_attention_layernorm = nn.LayerNorm(4)


class _GlmModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_GlmLayer(), _GlmLayer()])
        self.embed_tokens = nn.Embedding(16, 4)


class _GlmForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _GlmModel()
        self.config = SimpleNamespace(
            model_type="glm",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


class _NestedModelForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.model = _GlmModel()
        self.config = SimpleNamespace(
            model_type="nested_dense",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.model.embed_tokens.weight


class _NestedFallbackModelForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _GlmModel()
        self.model.model = nn.Module()
        self.config = SimpleNamespace(
            model_type="nested_fallback_dense",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def test_hf_causal_adapter_handles_olmo2_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()

    assert adapter.can_handle(_Olmo2ForCausalLM()) is True


def test_hf_causal_adapter_describes_olmo2_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()
    description = adapter.describe(_Olmo2ForCausalLM())

    assert description["hf_model_type"] == "olmo2"
    assert description["spec"] == "dense_decoder"
    assert description["mlp_dims"] == [8, 8]


def test_hf_causal_adapter_returns_olmo2_layer_modules() -> None:
    adapter = HF_Causal_Adapter()
    model = _Olmo2ForCausalLM()

    modules = adapter.get_layer_modules(model, 0)

    assert modules["self_attn.q_proj"] is model.model.layers[0].self_attn.q_proj
    assert (
        modules["input_layernorm"] is model.model.layers[0].post_feedforward_layernorm
    )
    assert (
        modules["post_feedforward_layernorm"]
        is model.model.layers[0].post_feedforward_layernorm
    )


def test_hf_causal_adapter_handles_qwen35_linear_attention_layout() -> None:
    adapter = HF_Causal_Adapter()

    assert adapter.can_handle(_Qwen35ForCausalLM()) is True


def test_hf_causal_adapter_describes_qwen35_linear_attention_layout() -> None:
    adapter = HF_Causal_Adapter()
    description = adapter.describe(_Qwen35ForCausalLM())

    assert description["hf_model_type"] == "qwen3_5_text"
    assert description["spec"] == "qwen35_linear_decoder"
    assert description["mlp_dims"] == [8, 8]


def test_hf_causal_adapter_returns_qwen35_layer_modules() -> None:
    adapter = HF_Causal_Adapter()
    model = _Qwen35ForCausalLM()

    modules = adapter.get_layer_modules(model, 0)

    assert (
        modules["linear_attn.in_proj_qkv"]
        is model.model.layers[0].linear_attn.in_proj_qkv
    )
    assert modules["linear_attn.out_proj"] is model.model.layers[0].linear_attn.out_proj
    assert modules["mlp.gate_proj"] is model.model.layers[0].mlp.gate_proj


def test_hf_causal_adapter_handles_gpt_oss_moe_layout() -> None:
    adapter = HF_Causal_Adapter()

    assert adapter.can_handle(_GptOssForCausalLM()) is True


def test_hf_causal_adapter_describes_gpt_oss_moe_layout() -> None:
    adapter = HF_Causal_Adapter()
    description = adapter.describe(_GptOssForCausalLM())

    assert description["hf_model_type"] == "gpt_oss"
    assert description["spec"] == "gpt_oss_moe_decoder"
    assert description["mlp_dims"] == [8, 8]


def test_hf_causal_adapter_returns_gpt_oss_layer_modules() -> None:
    adapter = HF_Causal_Adapter()
    model = _GptOssForCausalLM()

    modules = adapter.get_layer_modules(model, 0)

    assert modules["self_attn.q_proj"] is model.model.layers[0].self_attn.q_proj
    assert modules["mlp.router"] is model.model.layers[0].mlp.router
    assert modules["mlp.experts"] is model.model.layers[0].mlp.experts


def test_hf_causal_adapter_handles_tensorized_mixtral_layout() -> None:
    adapter = HF_Causal_Adapter()

    assert adapter.can_handle(_MixtralForCausalLM()) is True


def test_hf_causal_adapter_describes_tensorized_mixtral_layout() -> None:
    adapter = HF_Causal_Adapter()
    description = adapter.describe(_MixtralForCausalLM())

    assert description["hf_model_type"] == "mixtral"
    assert description["spec"] == "moe_decoder"
    assert description["mlp_dims"] == [8, 8]


def test_hf_causal_adapter_returns_tensorized_mixtral_layer_modules() -> None:
    adapter = HF_Causal_Adapter()
    model = _MixtralForCausalLM()

    modules = adapter.get_layer_modules(model, 0)

    assert modules["self_attn.q_proj"] is model.model.layers[0].self_attn.q_proj
    assert modules["mlp.router"] is model.model.layers[0].mlp.gate
    assert modules["mlp.gate"] is model.model.layers[0].mlp.gate
    assert modules["mlp.experts"] is model.model.layers[0].mlp.experts


def test_hf_causal_adapter_handles_opt_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()

    assert adapter.can_handle(_OptForCausalLM()) is True


def test_hf_causal_adapter_describes_opt_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()
    description = adapter.describe(_OptForCausalLM())

    assert description["hf_model_type"] == "opt"
    assert description["spec"] == "opt_decoder"
    assert description["mlp_dims"] == [8, 8]


def test_hf_causal_adapter_returns_opt_layer_modules() -> None:
    adapter = HF_Causal_Adapter()
    model = _OptForCausalLM()

    modules = adapter.get_layer_modules(model, 0)

    assert (
        modules["self_attn.o_proj"] is model.model.decoder.layers[0].self_attn.out_proj
    )
    assert modules["mlp.c_fc"] is model.model.decoder.layers[0].fc1
    assert modules["mlp.c_proj"] is model.model.decoder.layers[0].fc2


def test_hf_causal_adapter_handles_neox_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()

    assert adapter.can_handle(_NeoXForCausalLM()) is True


def test_hf_causal_adapter_describes_neox_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()
    description = adapter.describe(_NeoXForCausalLM())

    assert description["hf_model_type"] == "gpt_neox"
    assert description["spec"] == "neox_decoder"
    assert description["mlp_dims"] == [8, 8]


def test_hf_causal_adapter_handles_falcon_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()

    assert adapter.can_handle(_FalconForCausalLM()) is True


def test_hf_causal_adapter_describes_falcon_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()
    description = adapter.describe(_FalconForCausalLM())

    assert description["hf_model_type"] == "falcon"
    assert description["spec"] == "falcon_decoder"
    assert description["mlp_dims"] == [8, 8]


def test_hf_causal_adapter_handles_glm_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()

    assert adapter.can_handle(_GlmForCausalLM()) is True


def test_hf_causal_adapter_describes_glm_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()
    description = adapter.describe(_GlmForCausalLM())

    assert description["hf_model_type"] == "glm"
    assert description["spec"] == "glm_decoder"
    assert description["mlp_dims"] == [8, 8]


def test_hf_causal_adapter_handles_nested_model_model_layout() -> None:
    adapter = HF_Causal_Adapter()
    model = _NestedModelForCausalLM()

    assert adapter.can_handle(model) is True

    description = adapter.describe(model)
    modules = adapter.get_layer_modules(model, 0)

    assert description["spec"] == "glm_decoder"
    assert modules["mlp.down_proj"] is model.model.model.layers[0].mlp.down_proj


def test_hf_causal_adapter_falls_back_when_nested_model_has_no_layers() -> None:
    adapter = HF_Causal_Adapter()
    model = _NestedFallbackModelForCausalLM()

    description = adapter.describe(model)
    modules = adapter.get_layer_modules(model, 0)

    assert description["spec"] == "glm_decoder"
    assert modules["mlp.down_proj"] is model.model.layers[0].mlp.down_proj
