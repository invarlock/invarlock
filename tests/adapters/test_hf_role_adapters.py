from __future__ import annotations

from unittest.mock import Mock

import torch.nn as nn

from invarlock.adapters.hf_causal import HF_Causal_Adapter


class MockDenseCausalLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size)

        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.mlp.up_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.mlp.down_proj = nn.Linear(hidden_size * 4, hidden_size)

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)


class MockDenseCausalModel(nn.Module):
    def __init__(
        self,
        n_layer: int = 2,
        hidden_size: int = 32,
        vocab_size: int = 64,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.config = Mock()
        self.config.model_type = "mistral"
        self.config.num_hidden_layers = n_layer
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = 4
        self.config.num_key_value_heads = 2
        self.config.intermediate_size = hidden_size * 4
        self.config.vocab_size = vocab_size
        self.config.max_position_embeddings = 2048
        self.config.rms_norm_eps = 1e-6

        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockDenseCausalLayer(hidden_size) for _ in range(n_layer)]
        )
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight


class MockMoEExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)


class MockMoECausalLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size)

        self.block_sparse_moe = nn.Module()
        self.block_sparse_moe.experts = nn.ModuleList(
            [MockMoEExpert(hidden_size, intermediate_size)]
        )

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)


class MockMoECausalModel(nn.Module):
    def __init__(
        self,
        n_layer: int = 2,
        hidden_size: int = 32,
        intermediate_size: int = 128,
        vocab_size: int = 64,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.config = Mock()
        self.config.model_type = "mixtral"
        self.config.num_hidden_layers = n_layer
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = 4
        self.config.num_key_value_heads = 2
        self.config.intermediate_size = intermediate_size
        self.config.vocab_size = vocab_size
        self.config.max_position_embeddings = 32768
        self.config.rms_norm_eps = 1e-6

        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockMoECausalLayer(hidden_size, intermediate_size) for _ in range(n_layer)]
        )
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight


class TestHFCausalAdapter:
    def test_causal_describe_supports_dense_structure(self):
        adapter = HF_Causal_Adapter()
        model = MockDenseCausalModel(tie_weights=True)

        assert adapter.can_handle(model) is True

        desc = adapter.describe(model)
        assert desc["n_layer"] == model.config.num_hidden_layers
        assert len(desc["mlp_dims"]) == model.config.num_hidden_layers
        assert all(dim == model.config.intermediate_size for dim in desc["mlp_dims"])

        modules = adapter.get_layer_modules(model, 0)
        assert "self_attn.q_proj" in modules
        assert "mlp.down_proj" in modules

    def test_causal_describe_supports_moe_structure(self):
        adapter = HF_Causal_Adapter()
        model = MockMoECausalModel(tie_weights=True)

        assert adapter.can_handle(model) is True

        desc = adapter.describe(model)
        assert desc["n_layer"] == model.config.num_hidden_layers
        assert len(desc["mlp_dims"]) == model.config.num_hidden_layers
        assert all(dim == model.config.intermediate_size for dim in desc["mlp_dims"])

        modules = adapter.get_layer_modules(model, 0)
        assert modules["mlp.gate_proj"] is model.model.layers[0].block_sparse_moe.experts[
            0
        ].w1
        assert modules["mlp.down_proj"] is model.model.layers[0].block_sparse_moe.experts[
            0
        ].w2
        assert modules["mlp.up_proj"] is model.model.layers[0].block_sparse_moe.experts[
            0
        ].w3

