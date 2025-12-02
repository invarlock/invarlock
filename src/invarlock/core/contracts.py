"""
InvarLock Contracts
===============

Lightweight runtime assertions for monotonic behaviour of guard/edit operations.
"""

from __future__ import annotations

import torch


def enforce_relative_spectral_cap(
    weight: torch.Tensor, baseline_sigma: float | torch.Tensor, cap_ratio: float
) -> torch.Tensor:
    """Clamp the spectral norm of ``weight`` to ``cap_ratio * baseline_sigma``."""
    baseline_value = float(baseline_sigma)
    if not torch.isfinite(torch.tensor(baseline_value)) or baseline_value <= 0:
        return weight
    with torch.no_grad():
        sigma = _spectral_norm(weight)
        limit = baseline_value * cap_ratio
        if sigma > limit and sigma > 0:
            weight.mul_(limit / sigma)
    return weight


def enforce_weight_energy_bound(
    approx: torch.Tensor, exact: torch.Tensor, max_relative_error: float
) -> torch.Tensor:
    """Return ``approx`` if the relative error against ``exact`` is within bounds."""
    denom = torch.norm(exact).clamp_min(1e-12)
    rel_err = torch.norm(approx - exact) / denom
    if rel_err <= max_relative_error:
        return approx
    return exact


def rmt_correction_is_monotone(
    corrected_sigma: float,
    baseline_sigma: float,
    max_ratio: float,
    deadband: float,
) -> bool:
    """
    Validate monotonicity for RMT correction.

    ``corrected_sigma`` should not exceed ``baseline_sigma * (1 + deadband)``
    and must remain ≤ ``max_ratio``.
    """
    if corrected_sigma < 0 or baseline_sigma <= 0 or max_ratio <= 0:
        return False
    if corrected_sigma > max_ratio:
        return False
    return corrected_sigma <= baseline_sigma * (1.0 + deadband)


def _spectral_norm(weight: torch.Tensor) -> float:
    """Compute the spectral norm (largest singular value) of ``weight``."""
    if weight.ndim != 2:
        weight = weight.view(weight.shape[0], -1)
    try:
        s = torch.linalg.svdvals(weight)
    except RuntimeError:
        s = torch.linalg.svdvals(weight.cpu()).to(weight.device)
    return float(s.max().item())


__all__ = [
    "enforce_relative_spectral_cap",
    "enforce_weight_energy_bound",
    "rmt_correction_is_monotone",
]
