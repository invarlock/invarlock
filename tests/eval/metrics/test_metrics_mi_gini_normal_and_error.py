import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig
from invarlock.eval.metrics_activation import _calculate_mi_gini


def test_mi_gini_cpu_layer_contract_success_and_error():
    # Activation data: list of [L,N,T,D] and list of [N,T]
    L, N, T, D = 1, 1, 5, 3
    feats = torch.randn(L, N, T, D)
    targs = torch.randint(0, 7, (N, T - 1))
    activation_data = {"fc1_activations": [feats], "targets": [targs]}

    observed_shapes = []

    def mi_scores_fn(layer_features, layer_targets):
        observed_shapes.append((layer_features.shape, layer_targets.shape))
        return layer_features.abs().mean(dim=0)

    class DM:
        def is_available(self, name):
            return name == "mi_scores"

        def get_module(self, name):
            return mi_scores_fn

    val = _calculate_mi_gini(
        model=nn.Linear(2, 2),
        activation_data=activation_data,
        dep_manager=DM(),
        config=MetricsConfig(),
        device=torch.device("cpu"),
    )
    assert isinstance(val, float)
    assert observed_shapes == [(torch.Size([4, 3]), torch.Size([4]))]

    # A failed layer is represented by zero scores; all-zero Gini is NaN.
    class DMErr(DM):
        def get_module(self, name):
            def fn(*a, **k):
                raise ValueError("bad")

            return fn

    val2 = _calculate_mi_gini(
        model=nn.Linear(2, 2),
        activation_data=activation_data,
        dep_manager=DMErr(),
        config=MetricsConfig(),
        device=torch.device("cpu"),
    )
    assert isinstance(val2, float) and (val2 != val2)  # NaN
