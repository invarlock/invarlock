from __future__ import annotations

import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


class _NoMatrixWeight(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = None


class _MixedWeightModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip = _NoMatrixWeight()
        self.linear = nn.Linear(4, 4, bias=False)


def test_spectral_guard_prepare_skips_modules_without_matrix_weights() -> None:
    model = _MixedWeightModel()
    guard = SpectralGuard()

    result = guard.prepare(model, adapter=None, calib=None, policy={})

    assert result["ready"] is True
    assert "linear" in guard.baseline_sigmas
    assert "skip" not in guard.baseline_sigmas
    assert "linear" in guard.module_family_map
    assert "skip" not in guard.module_family_map
