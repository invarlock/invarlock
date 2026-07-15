import types

import pytest
import torch
import torch.nn as nn

from invarlock.eval import metrics as metrics_mod
from invarlock.eval.metrics import MetricsConfig, calculate_lens_metrics_for_model


class Block(nn.Module):
    def __init__(self, d=4):
        super().__init__()

        # MLP with c_fc to support fc1 extraction
        class CF(nn.Module):
            def forward(self, x):
                return torch.randn_like(x)

        self.mlp = types.SimpleNamespace(c_fc=CF())


class Model(nn.Module):
    def __init__(self, vocab=5):
        super().__init__()
        self.vocab = vocab
        self.proj = nn.Linear(4, 4, bias=False)
        self.transformer = types.SimpleNamespace(h=[Block(), Block()])

    def forward(self, input_ids=None, output_hidden_states=False, **kwargs):
        B, T = input_ids.shape
        if output_hidden_states:
            # Provide >=3 hidden states so internals keep 1:-1 slice
            hidden = [torch.randn(B, T, 4) for _ in range(4)]
            return types.SimpleNamespace(hidden_states=hidden)
        # For strict forward (not used here)
        return types.SimpleNamespace(logits=torch.randn(B, T, self.vocab))


def test_calculate_lens_metrics_integration_and_cache(monkeypatch):
    def mi_scores(x, y):
        del y
        return x.abs().mean(dim=0)

    class ExactDependencies:
        available_modules = {
            "mi_scores": mi_scores,
            "scan_model_gains": lambda _model: {
                "spectral_norms": [0.1, 0.2],
                "scanned_modules": 2,
            },
        }

        def is_available(self, name):
            return name in self.available_modules

        def get_module(self, name):
            return self.available_modules[name]

    monkeypatch.setattr(metrics_mod, "DependencyManager", ExactDependencies)

    model = Model().eval()
    dl = [{"input_ids": torch.ones(1, 12, dtype=torch.long)}]
    cfg = MetricsConfig(oracle_windows=1, use_cache=True)

    # Compute (cache is per-instance; we just validate the path executes)
    res1 = calculate_lens_metrics_for_model(model, dl, config=cfg)
    assert set(res1.keys()) == {"sigma_max", "head_energy", "mi_gini"}
    assert res1["sigma_max"] == pytest.approx(0.2)
    assert torch.isfinite(torch.tensor(res1["head_energy"]))
    assert torch.isfinite(torch.tensor(res1["mi_gini"]))
    # Intentionally avoid asserting cache-hit across calls since cache is per-function-instance
