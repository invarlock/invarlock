from unittest.mock import patch

import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


def test_spectral_prepare_ready_path_with_mocks():
    guard = SpectralGuard(
        scope="all", multiple_testing={"method": "bh", "alpha": 0.05, "m": 4}
    )
    model = nn.Sequential(nn.Linear(4, 4))

    fake_sigmas = {"layer": 1.0}
    fake_families = {"layer": "ffn"}
    fake_family_stats = {"ffn": {"kappa": 2.0}}

    with (
        patch(
            "invarlock.guards.spectral_measurement.capture_baseline_sigmas",
            lambda *a, **k: fake_sigmas,
        ),
        patch(
            "invarlock.guards.spectral_detection.classify_model_families",
            lambda *a, **k: fake_families,
        ),
        patch(
            "invarlock.guards.spectral_detection.compute_family_stats",
            lambda *a, **k: fake_family_stats,
        ),
        patch(
            "invarlock.guards.spectral_measurement.scan_model_gains",
            lambda *a, **k: {"gains": {}},
        ),
        patch(
            "invarlock.guards.spectral_detection.summarize_sigmas",
            lambda *a, **k: {"summary": {}},
        ),
        patch(
            "invarlock.guards.spectral_measurement.auto_sigma_target",
            lambda *a, **k: 1.23,
        ),
    ):
        out = guard.prepare(
            model,
            adapter=None,
            calib=None,
            policy={"multiple_testing": {"method": "bh", "alpha": 0.05, "m": 4}},
        )
    assert out["ready"] is True and guard.prepared is True
