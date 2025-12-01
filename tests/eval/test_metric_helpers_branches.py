import math

import torch
import torch.nn as nn

from invarlock.eval.metrics import (
    MetricsConfig,
    _calculate_head_energy,
    _calculate_mi_gini,
    _calculate_sigma_max,
)


def test_head_energy_strict_nan_returns_nan():
    # Hidden states contain NaNs; with strict_validation, validator raises, helper returns NaN
    hs = torch.full((2, 1, 4, 8), float("nan"))
    res = _calculate_head_energy(
        [hs], MetricsConfig(strict_validation=True, progress_bars=False)
    )
    assert math.isnan(res)


def test_sigma_max_empty_filtered_returns_nan():
    # Fake dep_manager to mark scan_model_gains available and return empty gains
    class FakeDep:
        def is_available(self, name):
            return name == "scan_model_gains"

        def get_module(self, name):
            class Gains:
                columns = ["name"]

                def __len__(self):
                    return 0

                def __getitem__(self, mask):
                    return self

                @property
                def values(self):
                    return []

            def scan_model_gains(model, first_batch):
                return Gains()

            return scan_model_gains

    # Model/first_batch placeholders
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1))

    res = _calculate_sigma_max(
        M(),
        {},
        FakeDep(),
        MetricsConfig(progress_bars=False),
        device=torch.device("cpu"),
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
        MetricsConfig(progress_bars=False),
        torch.device("cpu"),
    )
    assert math.isnan(res)
