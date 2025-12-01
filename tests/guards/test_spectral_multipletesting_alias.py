import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


def test_prepare_multipletesting_alias_hydration():
    model = nn.Sequential(nn.Linear(4, 4))
    guard = SpectralGuard()
    _ = guard.prepare(
        model,
        adapter=None,
        calib=None,
        policy={"multipletesting": {"method": "bh", "alpha": 0.01}},
    )
    # multiple_testing hydrated into config
    mt = guard.config.get("multiple_testing")
    assert isinstance(mt, dict) and mt.get("alpha") == 0.01
