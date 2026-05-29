"""Pure RMT math helpers shared by the RMT owner modules and direct tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch

__all__ = [
    "mp_bulk_edges",
    "mp_bulk_edge",
    "rmt_growth_ratio",
    "within_deadband",
    "clip_full_svd",
]


def mp_bulk_edges(m: int, n: int, whitened: bool = True) -> tuple[float, float]:
    """Compute Marchenko-Pastur bulk edges for an ``m x n`` matrix."""
    if m == 0 or n == 0:
        return 0.0, 0.0

    q = n / m
    if whitened:
        sigma_max = 1.0 + np.sqrt(q)
        sigma_min = abs(1.0 - np.sqrt(q)) if q <= 1 else 0.0
    else:
        sigma_max = np.sqrt(m) * (1.0 + np.sqrt(q))
        sigma_min = np.sqrt(m) * abs(1.0 - np.sqrt(q)) if q <= 1 else 0.0

    return float(sigma_min), float(sigma_max)


def mp_bulk_edge(m: int, n: int, whitened: bool = False) -> float:
    """Compute the upper Marchenko-Pastur bulk edge for an ``m x n`` matrix."""
    return mp_bulk_edges(m, n, whitened=whitened)[1]


def rmt_growth_ratio(
    sigma_cur: float,
    mp_cur: float,
    sigma_base: float,
    mp_base: float,
) -> float:
    """Compute the baseline-aware growth ratio used by RMT checks."""
    r_base = sigma_base / max(mp_base, 1e-12)
    r_cur = sigma_cur / max(mp_cur, 1e-12)
    return r_cur / max(r_base, 1e-12)


def within_deadband(sigma_cur: float, sigma_base: float, deadband: float) -> bool:
    """Check whether the current sigma stays within the allowed deadband."""
    return sigma_cur <= (1.0 + deadband) * sigma_base


def clip_full_svd(
    W: torch.Tensor,
    clip_val: float,
    return_components: bool = False,
) -> (
    torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
):
    """Clip singular values of a matrix using full SVD."""
    if not torch.isfinite(W).all():
        if return_components:
            return None, None, None
        return W

    try:
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        S_clipped = torch.clamp(S, max=clip_val)
        if return_components:
            return U, S_clipped, Vt
        clipped = (U @ torch.diag(S_clipped) @ Vt).to(W.dtype)
        return cast(torch.Tensor, clipped)
    except (RuntimeError, torch.linalg.LinAlgError):
        if return_components:
            return None, None, None
        return W
