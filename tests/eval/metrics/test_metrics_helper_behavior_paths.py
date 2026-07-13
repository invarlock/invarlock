import math

import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig
from invarlock.eval.metrics_activation import (
    _calculate_head_energy,
    _calculate_mi_gini,
    _calculate_sigma_max,
)


def test_head_energy_strict_nan_returns_nan():
    # Hidden states contain NaNs; with strict_validation, validator raises, helper returns NaN
    hs = torch.full((2, 1, 4, 8), float("nan"))
    res = _calculate_head_energy([hs], MetricsConfig(strict_validation=True))
    assert math.isnan(res)


def test_sigma_max_empty_scan_returns_nan():
    class FakeDep:
        def is_available(self, name):
            return name == "scan_model_gains"

        def get_module(self, name):
            def scan_model_gains(model):
                return {"spectral_norms": [], "scanned_modules": 0}

            return scan_model_gains

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1))

    res = _calculate_sigma_max(
        M(),
        FakeDep(),
        MetricsConfig(),
    )
    assert math.isnan(res)


def test_mi_gini_dep_available_but_no_activations():
    class FakeDep2:
        def is_available(self, name):
            return name == "mi_scores"

        def get_module(self, name):
            def mi_scores_fn(x, y):
                return torch.zeros_like(x[..., 0])

            return mi_scores_fn

    activation_data = {"fc1_activations": [], "targets": []}
    res = _calculate_mi_gini(
        nn.Linear(1, 1),
        activation_data,
        FakeDep2(),
        MetricsConfig(),
        torch.device("cpu"),
    )
    assert math.isnan(res)
