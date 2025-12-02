import pytest

from invarlock.guards.variance import equalise_residual_variance


def test_equalise_residual_variance_raises_on_empty_when_disallowed():
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, bias=False)

        def forward(self, x):
            return x

    with pytest.raises(ValueError, match="Empty dataloader"):
        equalise_residual_variance(M(), [], windows=0, allow_empty=False)
