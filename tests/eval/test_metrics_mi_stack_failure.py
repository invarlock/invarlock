import torch

from invarlock.eval.metrics import MetricsConfig, _mi_gini_optimized_cpu_path


def test_mi_gini_stack_failure_returns_nan(monkeypatch):
    # Fake DependencyManager to bypass optional import check
    class FakeDM:
        def __init__(self, *_a, **_k):
            pass

        def is_available(self, name):
            return name == "mi_scores"

        def get_module(self, name):
            def fn(feats, targ):
                # Return a per-token score vector (shape like feats[0])
                return torch.zeros_like(feats[0, :])

            return fn

    monkeypatch.setattr("invarlock.eval.metrics.DependencyManager", FakeDM)

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
            feats, targ, max_per_layer=10, config=MetricsConfig(progress_bars=False)
        )
        assert isinstance(val, float) and (val != val)  # NaN
    finally:
        # Restore stack for safety
        monkeypatch.setattr(torch, "stack", orig_stack)
