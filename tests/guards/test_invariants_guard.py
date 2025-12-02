import types

import torch
import torch.nn as nn

from invarlock.guards.invariants import InvariantsGuard


class DummyBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = nn.Module()
        self.bert.embeddings = nn.Module()
        self.bert.embeddings.word_embeddings = nn.Embedding(10, 4)
        self.cls = nn.Module()
        self.cls.predictions = nn.Module()
        self.cls.predictions.decoder = nn.Linear(4, 10)
        self.config = types.SimpleNamespace(model_type="bert", is_decoder=False)


def test_invariants_guard_records_profile_checks():
    guard = InvariantsGuard()
    model = DummyBertModel()

    policy = {"profile_checks": ["mlm_mask_alignment"]}
    guard.prepare(model, adapter=None, calib=None, policy=policy)

    assert guard.profile_checks == ("mlm_mask_alignment",)
    baseline_checks = guard.baseline_checks
    assert "profile::mlm_mask_alignment" in baseline_checks
    assert baseline_checks["profile::mlm_mask_alignment"] is True


def test_invariants_guard_detects_non_finite_weights():
    guard = InvariantsGuard(on_fail="abort")
    model = DummyBertModel()

    guard.prepare(model, adapter=None, calib=None, policy={})

    with torch.no_grad():
        weight = model.bert.embeddings.word_embeddings.weight
        weight[0, 0] = float("nan")

    outcome = guard.finalize(model)

    assert outcome.passed is False
    assert outcome.action == "abort"
    assert any(v["type"] == "non_finite_tensor" for v in outcome.violations)


def test_invariants_guard_detects_embedding_vocab_mismatch():
    guard = InvariantsGuard(on_fail="abort")
    model = DummyBertModel()

    guard.prepare(model, adapter=None, calib=None, policy={})

    # Replace embeddings with different vocab size
    model.bert.embeddings.word_embeddings = nn.Embedding(12, 4)

    outcome = guard.finalize(model)

    assert outcome.passed is False
    assert any(v["type"] == "tokenizer_mismatch" for v in outcome.violations)


class LayerNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def test_invariants_guard_detects_missing_layer_norm():
    guard = InvariantsGuard(on_fail="abort")
    model = LayerNormModel()

    guard.prepare(model, adapter=None, calib=None, policy={})

    model.norm = nn.Identity()

    outcome = guard.finalize(model)

    assert outcome.passed is False
    assert any(v["type"] == "layer_norm_missing" for v in outcome.violations)
