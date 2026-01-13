from __future__ import annotations

import math
from typing import Any

import torch

__all__ = [
    "power_iter_sigma_max",
    "frobenius_norm_sq",
    "row_col_norm_extrema",
    "stable_rank_estimate",
]


def _as_matrix(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    return tensor.view(tensor.shape[0], -1)


def power_iter_sigma_max(
    matrix: Any,
    *,
    iters: int,
    init: str = "ones",
    eps: float = 1e-12,
) -> float:
    """Estimate the largest singular value (spectral norm) via fixed-iter power iteration.

    Contract properties (vNext):
    - fixed iteration budget (no convergence stopping)
    - deterministic initialization (`init`)
    - device-resident matvecs (no `.cpu()` transfers)
    """
    try:
        iters_i = int(iters)
    except Exception:
        iters_i = 4
    if iters_i < 1:
        iters_i = 1

    if not isinstance(matrix, torch.Tensor):
        return 0.0
    if matrix.numel() == 0:
        return 0.0
    if matrix.dtype in {torch.int8, torch.uint8}:
        return 0.0

    W = _as_matrix(matrix.detach())
    if W.numel() == 0 or W.shape[0] == 0 or W.shape[1] == 0:
        return 0.0

    device = W.device
    dtype = W.dtype
    n = int(W.shape[1])

    with torch.no_grad():
        if init == "ones":
            v = torch.ones((n,), device=device, dtype=dtype)
        else:
            # Deterministic fallback: unit vector e0.
            v = torch.zeros((n,), device=device, dtype=dtype)
            v[0] = 1

        v_norm = torch.linalg.vector_norm(v.float()).clamp_min(eps)
        v = v / v_norm.to(dtype)

        sigma = 0.0
        for _ in range(iters_i):
            u = W @ v
            u_norm = torch.linalg.vector_norm(u.float()).clamp_min(eps)
            sigma_val = float(u_norm.item())
            if not math.isfinite(sigma_val):
                return 0.0
            u = u / u_norm.to(dtype)
            v = W.T @ u
            v_norm = torch.linalg.vector_norm(v.float()).clamp_min(eps)
            v = v / v_norm.to(dtype)
            sigma = sigma_val
        return float(sigma)


def frobenius_norm_sq(matrix: torch.Tensor) -> float:
    """Return ||matrix||_F^2 with float32 accumulation (device-resident)."""
    W = _as_matrix(matrix.detach())
    if W.numel() == 0:
        return 0.0
    with torch.no_grad():
        # Use a fused reduction to avoid materializing a W*W intermediate.
        norm = torch.linalg.vector_norm(W.reshape(-1), ord=2, dtype=torch.float32)
        out = float((norm * norm).item())
        return out if math.isfinite(out) else 0.0


def row_col_norm_extrema(
    matrix: torch.Tensor, *, eps: float = 1e-12
) -> dict[str, float]:
    """Compute min/median/max of row/col L2 norms with float32 accumulation."""
    W = _as_matrix(matrix.detach())
    if W.numel() == 0 or W.shape[0] == 0 or W.shape[1] == 0:
        return {
            "row_min": 0.0,
            "row_median": 0.0,
            "row_max": 0.0,
            "col_min": 0.0,
            "col_median": 0.0,
            "col_max": 0.0,
        }
    with torch.no_grad():
        # Avoid materializing W*W: use fused reductions.
        row = torch.linalg.vector_norm(W, ord=2, dim=1, dtype=torch.float32).clamp_min(
            eps
        )
        col = torch.linalg.vector_norm(W, ord=2, dim=0, dtype=torch.float32).clamp_min(
            eps
        )

        row_sorted, _ = torch.sort(row)
        col_sorted, _ = torch.sort(col)

        def _median(sorted_vec: torch.Tensor) -> float:
            n = int(sorted_vec.numel())
            if n <= 0:
                return 0.0
            mid = n // 2
            if n % 2 == 1:
                return float(sorted_vec[mid].item())
            return float((sorted_vec[mid - 1] + sorted_vec[mid]).mul(0.5).item())

        return {
            "row_min": float(row_sorted[0].item()),
            "row_median": _median(row_sorted),
            "row_max": float(row_sorted[-1].item()),
            "col_min": float(col_sorted[0].item()),
            "col_median": _median(col_sorted),
            "col_max": float(col_sorted[-1].item()),
        }


def stable_rank_estimate(
    matrix: torch.Tensor, *, sigma_max: float, eps: float = 1e-12
) -> float:
    """Estimate stable rank: ||W||_F^2 / ||W||_2^2, using a provided σ̂max."""
    try:
        denom = float(sigma_max) ** 2
    except Exception:
        return 0.0
    if not math.isfinite(denom) or denom <= 0.0:
        return 0.0
    denom = max(denom, eps)
    num = frobenius_norm_sq(matrix)
    out = float(num) / denom if denom > 0 else 0.0
    return out if math.isfinite(out) else 0.0
