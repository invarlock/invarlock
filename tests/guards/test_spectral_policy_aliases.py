import pytest
import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


def test_prepare_rejects_contraction_alias():
    model = nn.Sequential(nn.Linear(4, 4))
    guard = SpectralGuard()
    with pytest.raises(ValueError, match=r"sigma_quantile"):
        guard.prepare(
            model,
            adapter=None,
            calib=None,
            policy={"contraction": 0.9, "family_caps": {"ffn": 2.0}},
        )
