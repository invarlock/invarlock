from __future__ import annotations

import torch

from invarlock.core.contracts import (
    enforce_relative_spectral_cap,
    enforce_weight_energy_bound,
    rmt_correction_is_monotone,
)


def test_enforce_relative_spectral_cap_negative_safe_limit_clamps_to_zero() -> None:
    weight = torch.eye(2, dtype=torch.float32)

    capped = enforce_relative_spectral_cap(
        weight.clone(),
        baseline_sigma=1.0,
        cap_ratio=-0.5,
    )

    assert torch.count_nonzero(capped) == 0


def test_enforce_weight_energy_bound_handles_zero_exact_norm_edges() -> None:
    exact = torch.zeros(2, dtype=torch.float32)

    accepted = enforce_weight_energy_bound(
        exact.clone(),
        exact,
        max_relative_error=0.0,
    )
    rejected = enforce_weight_energy_bound(
        torch.ones(2, dtype=torch.float32),
        exact,
        max_relative_error=0.0,
    )

    assert torch.allclose(accepted, exact)
    assert torch.allclose(rejected, exact)


def test_enforce_relative_spectral_cap_accepts_tensor_baseline_sigma() -> None:
    weight = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    capped = enforce_relative_spectral_cap(
        weight.clone(),
        baseline_sigma=torch.tensor(1.0),
        cap_ratio=1.0,
    )

    assert torch.linalg.svdvals(capped)[0] <= 1.0 + 1e-6


def test_rmt_correction_is_monotone_rejects_non_positive_limits() -> None:
    assert (
        rmt_correction_is_monotone(1.0, baseline_sigma=0.0, max_ratio=1.0, deadband=0.1)
        is False
    )
    assert (
        rmt_correction_is_monotone(1.0, baseline_sigma=1.0, max_ratio=0.0, deadband=0.1)
        is False
    )
