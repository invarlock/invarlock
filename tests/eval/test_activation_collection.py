from types import SimpleNamespace

import torch
import torch.nn as nn

from invarlock.eval.metrics import MetricsConfig, _collect_activations


class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = SimpleNamespace(c_fc=nn.Linear(in_dim, out_dim))


class ToyModel(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # Build transformer.h list of blocks with different c_fc output dims
        self.transformer = SimpleNamespace(
            h=[Block(dims[0], dims[0]), Block(dims[0], dims[1])]
        )

    def forward(self, input_ids=None, output_hidden_states=False):  # noqa: D401
        B, T = input_ids.shape
        # Create hidden states list including first and last (to be excluded)
        hs0 = torch.randn(B, T, self.transformer.h[0].mlp.c_fc.in_features)
        hs1 = torch.randn(B, T, self.transformer.h[0].mlp.c_fc.in_features)
        hs2 = torch.randn(B, T, self.transformer.h[0].mlp.c_fc.in_features)
        # hidden_states length > 2 to include internal layers
        return SimpleNamespace(hidden_states=[hs0, hs1, hs2])


def test_collect_activations_shape_reconciliation():
    model = ToyModel(dims=(8, 6))
    # Two batches
    dl = [{"input_ids": torch.randint(0, 16, (1, 5))} for _ in range(2)]
    cfg = MetricsConfig(
        oracle_windows=2, max_tokens=5, progress_bars=False, strict_validation=False
    )
    out = _collect_activations(model, dl, cfg, device=torch.device("cpu"))
    # hidden_states collected
    assert out["hidden_states"]
    # fc1_activations stacked, with consistent shape after reconciliation
    fc1 = out["fc1_activations"]
    assert fc1 and isinstance(fc1[0], torch.Tensor)
    # Expect L dimension == number of blocks with consistent shapes (1 after reconcile)
    # The function stacks along layer dim after dropping mismatched shapes
    # Validate targets collected for MI-Gini path
    assert out["targets"] and out["first_batch"] is not None
