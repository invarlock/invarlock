import torch

from invarlock.eval.metrics import MetricsConfig
from invarlock.eval.metrics_activation import _mi_gini_optimized_cpu_path


def test_mi_gini_stack_failure_returns_nan(monkeypatch):
    # Prepare small feature/target tensors: L=2, N=1, D=4
    feats = torch.randn(2, 1, 4)
    targ = torch.randint(0, 5, (1,))

    # Force torch.stack to fail during stacking to hit exception path
    orig_stack = torch.stack

    def boom(seq):
        raise RuntimeError("stack-fail")

    monkeypatch.setattr(torch, "stack", boom)

    try:
        val = _mi_gini_optimized_cpu_path(
            feats,
            targ,
            max_per_layer=10,
            config=MetricsConfig(),
            mi_scores_fn=lambda x, _y: torch.zeros_like(x[0, :]),
        )
        assert isinstance(val, float) and (val != val)  # NaN
    finally:
        # Restore stack for safety
        monkeypatch.setattr(torch, "stack", orig_stack)
