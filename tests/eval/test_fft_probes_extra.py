from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

from invarlock.eval.probes.fft import compute_head_energy_scores, fft_head_energy

torch = pytest.importorskip("torch")


def test_fft_head_energy_positive_and_deterministic():
    attn = torch.ones((4, 4))
    e1 = fft_head_energy(attn)
    e2 = fft_head_energy(attn)
    assert e1 > 0 and abs(e1 - e2) < 1e-9


class _DummyConfig:
    n_layer = 2
    n_head = 2


class _FakeAttn(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads

    def forward(self, x, output_attentions=False):
        batch, seq_len = x.shape
        weights = torch.ones(batch, self.heads, seq_len, seq_len, device=x.device)
        return x, weights


class _FakeBlock(nn.Module):
    def __init__(self, heads: int):
        super().__init__()
        self.attn = _FakeAttn(heads)

    def forward(self, x):
        self.attn(x, output_attentions=True)
        return x


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _DummyConfig()
        self.transformer = SimpleNamespace(
            h=[_FakeBlock(self.config.n_head) for _ in range(self.config.n_layer)]
        )
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, output_attentions=False):
        x = input_ids
        for block in self.transformer.h:
            x = block(x)
        return x


def test_compute_head_energy_scores_collects_hooks():
    model = _FakeModel()
    batch = {"input_ids": torch.ones((1, 3), dtype=torch.long)}
    scores = compute_head_energy_scores(model, [batch], oracle_windows=1, device="cpu")
    assert scores.shape == (model.config.n_layer, model.config.n_head)
    assert torch.count_nonzero(scores) > 0


def test_compute_head_energy_scores_supports_inputs_alias():
    model = _FakeModel()
    batch_missing = {"foo": torch.ones((1, 2), dtype=torch.long)}
    batch_inputs = {"inputs": torch.ones((1, 2), dtype=torch.long)}
    scores = compute_head_energy_scores(
        model, [batch_missing, batch_inputs], oracle_windows=2, device="cpu"
    )
    # First batch skipped; second processed
    assert torch.count_nonzero(scores) > 0
