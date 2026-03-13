import torch

from invarlock.core.contracts import (
    enforce_relative_spectral_cap,
    enforce_weight_energy_bound,
    rmt_correction_is_monotone,
)


def test_enforce_relative_spectral_cap_branches():
    w = torch.randn(4, 4)
    # Non-finite baseline -> unchanged
    out = enforce_relative_spectral_cap(w.clone(), float("nan"), 1.0)
    assert torch.allclose(out, w)
    # Cap applies (limit < current sigma)
    out2 = enforce_relative_spectral_cap(w.clone(), 0.1, 1.0)
    assert out2.norm() <= w.norm()


def test_enforce_weight_energy_bound_both_paths():
    exact = torch.ones(4)
    approx_close = exact + 1e-9
    approx_far = exact + 1.0
    out1 = enforce_weight_energy_bound(approx_close, exact, max_relative_error=1e-6)
    assert torch.allclose(out1, approx_close)
    out2 = enforce_weight_energy_bound(approx_far, exact, max_relative_error=1e-6)
    assert torch.allclose(out2, exact)


def test_rmt_correction_is_monotone_branches():
    assert rmt_correction_is_monotone(-1.0, 1.0, 2.0, 0.1) is False
    assert rmt_correction_is_monotone(3.0, 1.0, 2.0, 0.1) is False
    assert rmt_correction_is_monotone(1.1, 1.0, 2.0, 0.05) is False
    assert rmt_correction_is_monotone(1.02, 1.0, 2.0, 0.05) is True
