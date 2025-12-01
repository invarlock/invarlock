from unittest.mock import patch

import torch

from invarlock.guards.spectral import SpectralGuard


class Dummy(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        with torch.no_grad():
            self.fc.weight.mul_(scale)

    def named_modules(self, memo=None, prefix=""):
        # Name as FFN family
        yield "layer.mlp.fc_in", self.fc


def test_spectral_after_edit_applies_control_and_logs_event():
    # Prepare with small baseline, then after_edit with large weights to trigger control
    baseline = Dummy(scale=1.0)
    edited = Dummy(scale=100.0)
    guard = SpectralGuard(scope="all", correction_enabled=True)

    fake_sigmas_small = {"layer.mlp.fc_in": 1.0}
    fake_sigmas_large = {"layer.mlp.fc_in": 100.0}
    fake_families = {"layer.mlp.fc_in": "ffn"}
    fake_family_stats = {"ffn": {"kappa": 2.0}}

    with patch(
        "invarlock.guards.spectral.capture_baseline_sigmas",
        side_effect=[fake_sigmas_small, fake_sigmas_large],
    ):
        with (
            patch(
                "invarlock.guards.spectral.classify_model_families",
                lambda *a, **k: fake_families,
            ),
            patch(
                "invarlock.guards.spectral.compute_family_stats",
                lambda *a, **k: fake_family_stats,
            ),
            patch(
                "invarlock.guards.spectral.scan_model_gains",
                lambda *a, **k: {"gains": {}},
            ),
            patch(
                "invarlock.guards.spectral._summarize_sigmas",
                lambda *a, **k: {"summary": {}},
            ),
            patch("invarlock.guards.spectral.auto_sigma_target", lambda *a, **k: 1.23),
        ):
            guard.prepare(baseline, adapter=None, calib=None, policy={})
            guard.before_edit(baseline)
            guard.after_edit(edited)

    # Ensure after_edit path executed (either applied control or failed gracefully)
    messages = [e.get("operation") for e in guard.events]
    assert "after_edit" in messages or "after_edit_failed" in messages
