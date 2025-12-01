from types import SimpleNamespace

import torch

from invarlock.eval import metrics as M


def test_mi_gini_gpu_oom_fallback(monkeypatch):
    # Build fake activation_data to feed into _calculate_mi_gini
    # L layers, N*T tokens folded into flat dims by the function
    L, _N, T, D = 2, 1, 3, 4
    hidden = [torch.randn(L, 1, T, D)]
    targets = [torch.randint(0, 5, (1, T))]
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
            raise RuntimeError("CUDA out of memory: forced test path")

    monkeypatch.setattr(M, "DependencyManager", lambda: StubDep())

    cfg = M.MetricsConfig(progress_bars=False)
    # Call private implementation to exercise fallback branch without heavyweight setup
    out = M._calculate_mi_gini(
        model=SimpleNamespace(),
        activation_data=activation_data,
        dep_manager=M.DependencyManager(),
        config=cfg,
        device=torch.device("cpu"),
    )
    assert isinstance(out, float) or (out != out)  # float or NaN
