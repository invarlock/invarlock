from __future__ import annotations

from unittest.mock import Mock

import torch.nn as nn

from invarlock.adapters.base import AdapterState, BaseAdapter


class MockGPT2Model(nn.Module):
    def __init__(self, n_layer=2, n_head=4, hidden_size=16):
        super().__init__()
        self.config = Mock()
        self.config.model_type = "gpt2"
        self.config.n_layer = n_layer
        self.config.n_head = n_head
        self.config.hidden_size = hidden_size
        self.config.vocab_size = 1000
        self.config.n_inner = hidden_size * 4

        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList()

        for _i in range(n_layer):
            layer = self._create_layer(n_head, hidden_size)
            self.transformer.h.append(layer)

        self.transformer.wte = nn.Embedding(1000, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 1000, bias=False)

        if hasattr(self, "tie_weights"):
            self.lm_head.weight = self.transformer.wte.weight

    def _create_layer(self, n_head, hidden_size):
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        layer.attn.c_proj = nn.Linear(hidden_size, hidden_size)
        layer.mlp = nn.Module()
        layer.mlp.c_fc = nn.Linear(hidden_size, hidden_size * 4)
        layer.mlp.c_proj = nn.Linear(hidden_size * 4, hidden_size)
        layer.ln_1 = nn.LayerNorm(hidden_size)
        layer.ln_2 = nn.LayerNorm(hidden_size)
        return layer


class MockBertLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = nn.Module()
        self.attention.self.query = nn.Linear(hidden_size, hidden_size)
        self.attention.self.key = nn.Linear(hidden_size, hidden_size)
        self.attention.self.value = nn.Linear(hidden_size, hidden_size)
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(hidden_size, hidden_size)
        self.attention.output.LayerNorm = nn.LayerNorm(hidden_size)

        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(hidden_size, hidden_size * 4)

        self.output = nn.Module()
        self.output.dense = nn.Linear(hidden_size * 4, hidden_size)
        self.output.LayerNorm = nn.LayerNorm(hidden_size)


class MockBertModel(nn.Module):
    def __init__(
        self,
        n_layer: int = 2,
        hidden_size: int = 32,
        vocab_size: int = 128,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.config = Mock()
        self.config.model_type = "bert"
        self.config.num_hidden_layers = n_layer
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = 4
        self.config.intermediate_size = hidden_size * 4
        self.config.vocab_size = vocab_size
        self.config.type_vocab_size = 2
        self.config.max_position_embeddings = 512
        self.config.layer_norm_eps = 1e-12
        self.config.hidden_dropout_prob = 0.1
        self.config.attention_probs_dropout_prob = 0.1

        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [MockBertLayer(hidden_size) for _ in range(n_layer)]
        )

        self.bert = nn.Module()
        self.bert.embeddings = self.embeddings
        self.bert.encoder = self.encoder

        self.pooler = nn.Linear(hidden_size, hidden_size)

        self.cls = nn.Module()
        self.cls.predictions = nn.Module()
        self.cls.predictions.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            self.cls.predictions.decoder.weight = self.embeddings.word_embeddings.weight


class MockRopeDecoderLayer(nn.Module):
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


class MockRopeDecoderModel(nn.Module):
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
            [MockRopeDecoderLayer(hidden_size) for _ in range(n_layer)]
        )
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight


class MockMixtralExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)


class MockMixtralLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size)

        self.block_sparse_moe = nn.Module()
        self.block_sparse_moe.experts = nn.ModuleList(
            [MockMixtralExpert(hidden_size, intermediate_size)]
        )

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)


class MockMixtralModel(nn.Module):
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
            [MockMixtralLayer(hidden_size, intermediate_size) for _ in range(n_layer)]
        )
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.model.embed_tokens.weight


class ConcreteAdapter(BaseAdapter):
    def load_model(self, model_id: str, **kwargs):
        self.state = AdapterState.LOADED
        return {"model_id": model_id}

    def generate(self, prompt: str, **kwargs) -> str:
        return f"Generated response for: {prompt}"

    def tokenize(self, text: str, **kwargs):
        return {"tokens": text.split(), "token_ids": list(range(len(text.split())))}

    def get_capabilities(self):
        return {"supports_generation": True, "supports_tokenization": True}
