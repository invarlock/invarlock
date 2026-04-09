from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from invarlock.adapters.hf_causal import HF_Causal_Adapter


class _PhiTextLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(4, 12, bias=False)
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(4, 16, bias=False)
        self.mlp.down_proj = nn.Linear(8, 4, bias=False)
        self.input_layernorm = nn.LayerNorm(4)
        self.post_attention_layernorm = nn.LayerNorm(4)


class _PhiLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_PhiTextLayer(), _PhiTextLayer()])
        self.embed_tokens = nn.Embedding(16, 4)


class _PhiConditionalGeneration(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _PhiLanguageModel()
        self.config = SimpleNamespace(
            model_type="phi3",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def test_hf_causal_adapter_handles_phi_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()
    model = _PhiConditionalGeneration()

    assert adapter.can_handle(model) is True


def test_hf_causal_adapter_describes_phi_decoder_layout() -> None:
    adapter = HF_Causal_Adapter()
    model = _PhiConditionalGeneration()

    description = adapter.describe(model)

    assert description["hf_model_type"] == "phi3"
    assert description["n_layer"] == 2
    assert description["heads_per_layer"] == [2, 2]
    assert description["mlp_dims"] == [8, 8]
    assert description["spec"] == "phi_decoder"
    assert description["tying"]["lm_head.weight"] == "model.embed_tokens.weight"


def test_hf_causal_adapter_returns_phi_layer_modules() -> None:
    adapter = HF_Causal_Adapter()
    model = _PhiConditionalGeneration()

    modules = adapter.get_layer_modules(model, 0)

    assert modules["self_attn.qkv_proj"] is model.model.layers[0].self_attn.qkv_proj
    assert modules["self_attn.o_proj"] is model.model.layers[0].self_attn.o_proj
    assert modules["mlp.gate_up_proj"] is model.model.layers[0].mlp.gate_up_proj
    assert modules["mlp.down_proj"] is model.model.layers[0].mlp.down_proj
