import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig, _calculate_mi_gini


def test_mi_gini_gpu_path_success_and_non_oom_error(monkeypatch):
    # Activation data: list of [L,N,T,D] and list of [N,T]
    L, N, T, D = 1, 1, 5, 3
    feats = torch.randn(L, N, T, D)
    targs = torch.randint(0, 7, (N, T))
    activation_data = {"fc1_activations": [feats], "targets": [targs]}

    # mi_scores returns per-token feature importance; shape [L, N*T]
    def mi_scores_fn(fc1_flat, targ_flat):
        # fc1_flat: [L, N*T, D]
        return fc1_flat.abs().mean(dim=-1)

    class DM:
        def is_available(self, name):
            return name == "mi_scores"

        def get_module(self, name):
            return mi_scores_fn

    # Normal (non-OOM) GPU path
    val = _calculate_mi_gini(
        model=nn.Linear(2, 2),
        activation_data=activation_data,
        dep_manager=DM(),
        config=MetricsConfig(progress_bars=False),
        device=torch.device("cpu"),
    )
    assert isinstance(val, float)

    # Non-OOM error path falls through to outer except and returns NaN
    class DMErr(DM):
        def get_module(self, name):
            def fn(*a, **k):
                raise ValueError("bad")

            return fn

    val2 = _calculate_mi_gini(
        model=nn.Linear(2, 2),
        activation_data=activation_data,
        dep_manager=DMErr(),
        config=MetricsConfig(progress_bars=False),
        device=torch.device("cpu"),
    )
    assert isinstance(val2, float) and (val2 != val2)  # NaN
