import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


def test_prepare_hydrates_aliases_and_family_caps():
    model = nn.Sequential(nn.Linear(4, 4))
    guard = SpectralGuard()
    out = guard.prepare(
        model,
        adapter=None,
        calib=None,
        policy={"contraction": 0.9, "family_caps": {"ffn": 2.0}},
    )
    assert out["ready"] in {True, False}  # prepare may run on minimal model
    # Aliases hydrated into sigma_quantile stored in config
    assert abs(guard.config.get("sigma_quantile", 0.0) - 0.9) < 1e-6
    # Family caps normalized mapping present
    fam = guard.config.get("family_caps", {})
    assert isinstance(fam, dict) and "ffn" in fam and "kappa" in fam["ffn"]
