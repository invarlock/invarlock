import math

import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig, _calculate_mi_gini


def test_mi_gini_cpu_fallback_on_oom():
    class FakeDep:
        def is_available(self, name):
            return name == "mi_scores"

        def get_module(self, name):
            def mi_scores_fn(x, y):
                raise RuntimeError("CUDA out of memory")

            return mi_scores_fn

    # Build minimal activation data with valid shapes
    L, N, T, D = 1, 1, 6, 4
    fc1 = torch.randn(L, N, T, D)
    targets = torch.randint(0, 16, (N, T))
    activation_data = {"fc1_activations": [fc1], "targets": [targets]}
    res = _calculate_mi_gini(
        nn.Linear(1, 1),
        activation_data,
        FakeDep(),
        MetricsConfig(progress_bars=False),
        torch.device("cpu"),
    )
    assert isinstance(res, float) and (math.isnan(res) or math.isfinite(res))
