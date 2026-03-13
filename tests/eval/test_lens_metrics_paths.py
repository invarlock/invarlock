import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig, calculate_lens_metrics_for_model


class NoActivationsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, input_ids=None, output_hidden_states=False):  # noqa: D401
        # Return logits only, no hidden_states attribute
        B, T = input_ids.shape
        x = torch.randn(B, T, 8)
        logits = self.proj(x)
        return SimpleNamespace(logits=logits)


def _make_dl(n=2, seq_len=6):
    dl = []
    for _ in range(n):
        ids = torch.randint(0, 16, (1, seq_len))
        dl.append({"input_ids": ids})
    return dl


def test_calculate_lens_metrics_no_activations_path():
    model = NoActivationsModel()
    dl = _make_dl(2, 6)
    cfg = MetricsConfig(oracle_windows=2, max_tokens=4, progress_bars=False)
    result = calculate_lens_metrics_for_model(model, dl, config=cfg)
    # With no activations and missing deps, metrics should be NaN
    assert (
        math.isnan(result["sigma_max"])
        and math.isnan(result["head_energy"])
        and math.isnan(result["mi_gini"])
    )
