import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from invarlock.eval.metrics import (
    _gini_vectorized,
    bootstrap_confidence_interval,
    compute_ppl,
    measure_latency,
)


def test_bootstrap_confidence_interval_validation_errors():
    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        bootstrap_confidence_interval([], n_bootstrap=10)
    with pytest.raises(MValidationError):
        bootstrap_confidence_interval([1.0], alpha=-0.1)
    with pytest.raises(MValidationError):
        bootstrap_confidence_interval([1.0], n_bootstrap=0)
    with pytest.raises(MValidationError):
        bootstrap_confidence_interval(np.ones((2, 2)))  # not 1D


class DummyLM(torch.nn.Module):
    def __init__(self, vocab=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, 4)
        self.proj = torch.nn.Linear(4, vocab)

    def forward(self, input_ids=None, attention_mask=None, return_dict=True):
        x = self.emb(input_ids)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits)


def test_compute_ppl_no_valid_tokens_raises():
    model = DummyLM()
    # All tokens invalid due to attention mask zeros or too-short sequences
    window = SimpleNamespace(
        input_ids=[[1], [2]],
        attention_masks=[[0], [0]],
    )
    from invarlock.eval.metrics import ValidationError as MValidationError

    with pytest.raises(MValidationError):
        compute_ppl(model, adapter=None, window=window, device="cpu")


def test_measure_latency_returns_zero_for_short_samples():
    model = DummyLM()
    # All sequences short (<=10), so measurement returns 0.0
    window = SimpleNamespace(
        input_ids=[[1, 2], [3, 4]],
        attention_masks=[[1, 1], [1, 1]],
    )
    ms = measure_latency(
        model, window, device="cpu", warmup_steps=1, measurement_steps=1
    )
    assert isinstance(ms, float) and ms == 0.0


def test_gini_vectorized_empty_nan():
    vec = torch.zeros(0)
    val = _gini_vectorized(vec)
    assert math.isnan(val)
