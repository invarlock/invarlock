from __future__ import annotations

from types import SimpleNamespace

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
