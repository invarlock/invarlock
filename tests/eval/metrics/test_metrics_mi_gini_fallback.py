from types import SimpleNamespace

import torch

from invarlock.eval import metrics as M
from invarlock.eval.metrics_activation import _calculate_mi_gini


def test_mi_gini_layer_failures_return_nan():
    # Build fake activation_data to feed into _calculate_mi_gini
    # L layers, N*T tokens folded into flat dims by the function
    L, _N, T, D = 2, 1, 3, 4
    hidden = [torch.randn(L, 1, T, D)]
    targets = [torch.randint(0, 5, (1, T - 1))]
    activation_data = {"fc1_activations": hidden, "targets": targets}

    class StubDep:
        def __init__(self):
            self.available_modules = {"mi_scores": self._fn}

        def is_available(self, name):
            return name in self.available_modules

        def get_module(self, name):
            return self.available_modules[name]

        @staticmethod
        def _fn(x, y):
            raise RuntimeError("forced layer scoring failure")

    cfg = M.MetricsConfig()
    # Call private implementation to exercise fallback branch without heavyweight setup
    out = _calculate_mi_gini(
        model=SimpleNamespace(),
        activation_data=activation_data,
        dep_manager=StubDep(),
        config=cfg,
        device=torch.device("cpu"),
    )
    assert isinstance(out, float) and (out != out)
