from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from invarlock.adapters.hf_causal import HF_Causal_Adapter


class _Gemma4TextLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.v_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.q_norm = nn.LayerNorm(4)
        self.self_attn.k_norm = nn.LayerNorm(4)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(4, 8, bias=False)
        self.mlp.up_proj = nn.Linear(4, 8, bias=False)
        self.mlp.down_proj = nn.Linear(8, 4, bias=False)
        self.input_layernorm = nn.LayerNorm(4)
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.pre_feedforward_layernorm = nn.LayerNorm(4)
        self.post_feedforward_layernorm = nn.LayerNorm(4)
        self.post_per_layer_input_norm = nn.LayerNorm(4)


class _Gemma4LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Gemma4TextLayer(), _Gemma4TextLayer()])
        self.norm = nn.LayerNorm(4)
        self.embed_tokens = nn.Embedding(16, 4)


class _Gemma4Container(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _Gemma4LanguageModel()


class _Gemma4ConditionalGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Gemma4Container()
        self.config = SimpleNamespace(
            model_type="gemma4",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


def test_hf_causal_adapter_unwraps_gemma4_conditional_generation_layout() -> None:
    adapter = HF_Causal_Adapter()
    model = _Gemma4ConditionalGeneration()

    base, layers, config = adapter._unwrap(model)

    assert base is model.model.language_model
    assert layers is model.model.language_model.layers
    assert config is model.config
    assert adapter.can_handle(model) is True


def test_hf_causal_adapter_describes_gemma4_conditional_generation_layout() -> None:
    adapter = HF_Causal_Adapter()
    model = _Gemma4ConditionalGeneration()

    description = adapter.describe(model)

    assert description["hf_model_type"] == "gemma4"
    assert description["n_layer"] == 2
    assert description["heads_per_layer"] == [2, 2]
    assert description["mlp_dims"] == [8, 8]
    assert (
        description["tying"]["lm_head.weight"]
        == "model.language_model.embed_tokens.weight"
    )


def test_hf_causal_adapter_uses_nested_text_config_for_gemma4_metadata() -> None:
    adapter = HF_Causal_Adapter()
    model = _Gemma4ConditionalGeneration()
    model.config = SimpleNamespace(
        model_type="gemma4",
        num_hidden_layers=None,
        num_attention_heads=None,
        hidden_size=None,
        vocab_size=None,
        text_config=SimpleNamespace(
            num_attention_heads=2,
            hidden_size=4,
            vocab_size=16,
        ),
    )

    description = adapter.describe(model)

    assert description["hf_model_type"] == "gemma4"
    assert description["n_heads"] == 2
    assert description["hidden_size"] == 4
    assert description["vocab_size"] == 16
