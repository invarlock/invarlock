from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from invarlock.adapters.hf_mlm import HF_MLM_Adapter


class _DistilAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_lin = nn.Linear(4, 4, bias=False)
        self.k_lin = nn.Linear(4, 4, bias=False)
        self.v_lin = nn.Linear(4, 4, bias=False)
        self.out_lin = nn.Linear(4, 4, bias=False)


class _DistilFFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 8, bias=False)
        self.lin2 = nn.Linear(8, 4, bias=False)


class _DistilLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _DistilAttention()
        self.sa_layer_norm = nn.LayerNorm(4)
        self.ffn = _DistilFFN()
        self.output_layer_norm = nn.LayerNorm(4)


class _DistilTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_DistilLayer(), _DistilLayer()])


class _DistilEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(16, 4)


class _DistilBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _DistilTransformer()
        self.embeddings = _DistilEmbeddings()


class _DistilBertForMaskedLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.distilbert = _DistilBackbone()
        self.config = SimpleNamespace(
            model_type="distilbert",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )
        self.vocab_projector = nn.Linear(4, 16, bias=False)
        self.vocab_projector.weight = self.distilbert.embeddings.word_embeddings.weight


class _DebertaSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_proj = nn.Linear(4, 4, bias=False)
        self.key_proj = nn.Linear(4, 4, bias=False)
        self.value_proj = nn.Linear(4, 4, bias=False)


class _DebertaAttentionOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(4, 4, bias=False)
        self.LayerNorm = nn.LayerNorm(4)


class _DebertaAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self = _DebertaSelfAttention()
        self.output = _DebertaAttentionOutput()


class _DebertaIntermediate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(4, 8, bias=False)


class _DebertaOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(8, 4, bias=False)
        self.LayerNorm = nn.LayerNorm(4)


class _DebertaLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _DebertaAttention()
        self.intermediate = _DebertaIntermediate()
        self.output = _DebertaOutput()


class _DebertaEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_DebertaLayer(), _DebertaLayer()])


class _DebertaEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(16, 4)


class _DebertaBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _DebertaEncoder()
        self.embeddings = _DebertaEmbeddings()


class _DebertaPredictions(nn.Module):
    def __init__(self, tied_weight: nn.Parameter) -> None:
        super().__init__()
        self.decoder = nn.Linear(4, 16, bias=False)
        self.decoder.weight = tied_weight


class _DebertaCls(nn.Module):
    def __init__(self, tied_weight: nn.Parameter) -> None:
        super().__init__()
        self.predictions = _DebertaPredictions(tied_weight)


class _DebertaV2ForMaskedLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deberta = _DebertaBackbone()
        self.cls = _DebertaCls(self.deberta.embeddings.word_embeddings.weight)
        self.config = SimpleNamespace(
            model_type="deberta-v2",
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=4,
            intermediate_size=8,
            vocab_size=16,
        )


def test_hf_mlm_adapter_handles_distilbert_layout() -> None:
    adapter = HF_MLM_Adapter()

    assert adapter.can_handle(_DistilBertForMaskedLM()) is True


def test_hf_mlm_adapter_describes_distilbert_layout() -> None:
    adapter = HF_MLM_Adapter()
    description = adapter.describe(_DistilBertForMaskedLM())

    assert description["hf_model_type"] == "distilbert"
    assert description["spec"] == "distilbert"
    assert description["mlp_dims"] == [8, 8]
    assert description["tying"] == {
        "vocab_projector.weight": "distilbert.embeddings.word_embeddings.weight"
    }


def test_hf_mlm_adapter_returns_distilbert_layer_modules() -> None:
    adapter = HF_MLM_Adapter()
    model = _DistilBertForMaskedLM()

    modules = adapter.get_layer_modules(model, 0)

    assert modules["attention.self.query"] is model.distilbert.transformer.layer[0].attention.q_lin
    assert modules["intermediate.dense"] is model.distilbert.transformer.layer[0].ffn.lin1
    assert modules["output.LayerNorm"] is model.distilbert.transformer.layer[0].output_layer_norm


def test_hf_mlm_adapter_handles_deberta_v2_layout() -> None:
    adapter = HF_MLM_Adapter()

    assert adapter.can_handle(_DebertaV2ForMaskedLM()) is True


def test_hf_mlm_adapter_describes_deberta_v2_layout() -> None:
    adapter = HF_MLM_Adapter()
    description = adapter.describe(_DebertaV2ForMaskedLM())

    assert description["hf_model_type"] == "deberta-v2"
    assert description["spec"] == "deberta-v2"
    assert description["mlp_dims"] == [8, 8]
    assert description["tying"] == {
        "cls.predictions.decoder.weight": "deberta.embeddings.word_embeddings.weight"
    }


def test_hf_mlm_adapter_returns_deberta_v2_layer_modules() -> None:
    adapter = HF_MLM_Adapter()
    model = _DebertaV2ForMaskedLM()

    modules = adapter.get_layer_modules(model, 0)

    assert modules["attention.self.query"] is model.deberta.encoder.layer[0].attention.self.query_proj
    assert modules["attention.output.dense"] is model.deberta.encoder.layer[0].attention.output.dense
    assert modules["output.LayerNorm"] is model.deberta.encoder.layer[0].output.LayerNorm
