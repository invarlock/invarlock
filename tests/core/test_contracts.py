"""Unit tests for invarlock.core.contracts helpers."""

import torch

from invarlock.core.contracts import (
    enforce_relative_spectral_cap,
    enforce_weight_energy_bound,
    rmt_correction_is_monotone,
)


def test_enforce_relative_spectral_cap_monotone() -> None:
    weight = torch.randn(4, 4)
    baseline_sigma = 5.0
    cap_ratio = 0.8

    capped = enforce_relative_spectral_cap(weight.clone(), baseline_sigma, cap_ratio)
    sigma_capped = torch.linalg.svdvals(capped)[0]

    assert sigma_capped <= baseline_sigma * cap_ratio + 1e-6


def test_enforce_weight_energy_bound_returns_exact_when_exceeding_threshold() -> None:
    exact = torch.eye(2)
    approx = exact + 0.5 * torch.ones_like(exact)

    result = enforce_weight_energy_bound(approx, exact, max_relative_error=0.1)
    assert torch.allclose(result, exact)


def test_enforce_weight_energy_bound_accepts_small_error() -> None:
    exact = torch.eye(2)
    approx = exact + 0.01 * torch.ones_like(exact)

    result = enforce_weight_energy_bound(approx, exact, max_relative_error=0.1)
    assert torch.allclose(result, approx)


def test_rmt_correction_is_monotone_basic() -> None:
    assert rmt_correction_is_monotone(
        1.05, baseline_sigma=1.0, max_ratio=1.2, deadband=0.1
    )
    assert not rmt_correction_is_monotone(
        1.5, baseline_sigma=1.0, max_ratio=1.2, deadband=0.1
    )
