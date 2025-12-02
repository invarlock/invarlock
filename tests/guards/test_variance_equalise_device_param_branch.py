import torch.nn as nn

from invarlock.guards.variance import equalise_residual_variance


class ParamOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(1, 1)


def test_equalise_residual_variance_explicit_device_branch():
    # Explicit device path (else branch at device setup)
    out = equalise_residual_variance(
        ParamOnly(), dataloader=[], allow_empty=True, windows=0, device="cpu"
    )
    assert out == {}
