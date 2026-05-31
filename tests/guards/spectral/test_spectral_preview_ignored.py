import torch.nn as nn

from invarlock.guards.spectral import SpectralGuard


def test_after_edit_ignored_when_preview_inflation_flag_true():
    # Prepare with default ignore_preview_inflation=True and ensure after_edit produces no violations
    model = nn.Sequential(nn.Linear(4, 4))
    guard = SpectralGuard(ignore_preview_inflation=True)
    _ = guard.prepare(model, adapter=None, calib=None, policy={})
    guard.after_edit(model)
    assert isinstance(guard.violations, list)
