import pytest
import torch.nn as nn

from invarlock.core.exceptions import ValidationError
from invarlock.guards.spectral import SpectralGuard


def test_prepare_rejects_multipletesting():
    model = nn.Sequential(nn.Linear(4, 4))
    guard = SpectralGuard()
    with pytest.raises(ValidationError):
        guard.prepare(
            model,
            adapter=None,
            calib=None,
            policy={"multipletesting": {"method": "bh", "alpha": 0.01}},
        )
