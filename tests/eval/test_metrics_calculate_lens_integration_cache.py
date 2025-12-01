import sys
import types

import torch
import torch.nn as nn

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
    # Stub optional dependencies to be available
    lens2 = types.ModuleType("invarlock.eval.lens2_mi")

    def mi_scores(x, y):
        # Return per-token importance [L, N*T]
        return x.abs().mean(dim=-1)

    lens2.mi_scores = mi_scores

    lens3 = types.ModuleType("invarlock.eval.lens3")

    def scan_model_gains(model, first_batch):
        class DF:
            columns = ["name", "gain"]

            def __len__(self):
                return 2

            def __getitem__(self, mask):
                return self

            @property
            def name(self):
                return ["mlp.c_fc", "mlp.c_fc"]

            @property
            def gain(self):
                return [0.1, 0.2]

        return DF()

    lens3.scan_model_gains = scan_model_gains

    monkeypatch.setitem(sys.modules, "invarlock.eval.lens2_mi", lens2)
    monkeypatch.setitem(sys.modules, "invarlock.eval.lens2_mi", lens2)
    monkeypatch.setitem(sys.modules, "invarlock.eval.lens3", lens3)
    monkeypatch.setitem(sys.modules, "invarlock.eval.lens3", lens3)

    model = Model().eval()
    dl = [{"input_ids": torch.ones(1, 12, dtype=torch.long)}]
    cfg = MetricsConfig(oracle_windows=1, progress_bars=False, use_cache=True)

    # Compute (cache is per-instance; we just validate the path executes)
    res1 = calculate_lens_metrics_for_model(model, dl, config=cfg)
    assert set(res1.keys()) == {"sigma_max", "head_energy", "mi_gini"}
    # Intentionally avoid asserting cache-hit across calls since cache is per-function-instance
