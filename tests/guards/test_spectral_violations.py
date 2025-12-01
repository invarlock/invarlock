from unittest.mock import patch

import torch

from invarlock.guards.spectral import SpectralGuard


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        with torch.no_grad():
            # Inflate singular values to trigger max_spectral_norm
            self.fc.weight.mul_(100.0)

    def named_modules(self, memo=None, prefix=""):
        # Name shaped to be considered FFN by scope logic
        yield "layer.mlp.fc_in", self.fc


def test_spectral_detects_max_norm_and_ill_conditioned_with_thresholds():
    model = DummyModel()
    guard = SpectralGuard(scope="all", max_caps=5)
    # Make thresholds aggressive to trigger both branches
    guard.max_spectral_norm = 0.5
    guard.min_condition_number = 1.0  # treat small min singular as violation

    fake_sigmas = {"layer.mlp.fc_in": 10.0}
    fake_families = {"layer.mlp.fc_in": "ffn"}
    fake_family_stats = {"ffn": {"kappa": 2.0}}

    with (
        patch(
            "invarlock.guards.spectral.capture_baseline_sigmas",
            lambda *a, **k: fake_sigmas,
        ),
        patch(
            "invarlock.guards.spectral.classify_model_families",
            lambda *a, **k: fake_families,
        ),
        patch(
            "invarlock.guards.spectral.compute_family_stats",
            lambda *a, **k: fake_family_stats,
        ),
        patch(
            "invarlock.guards.spectral.scan_model_gains", lambda *a, **k: {"gains": {}}
        ),
        patch(
            "invarlock.guards.spectral._summarize_sigmas",
            lambda *a, **k: {"summary": {}},
        ),
        patch("invarlock.guards.spectral.auto_sigma_target", lambda *a, **k: 1.23),
    ):
        # Prepare to set internal maps
        guard.prepare(model, adapter=None, calib=None, policy={})
        result = guard.validate(model, adapter=None, context={})

    assert isinstance(result, dict)
    assert result["metrics"]["violations_found"] >= 1
    # Ensure at least max_spectral_norm violation or ill_conditioned present
    vtypes = {v.get("type") for v in result.get("violations", [])}
    assert {"max_spectral_norm", "ill_conditioned"} & vtypes
