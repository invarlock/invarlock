"""Device-aware exact singular-value decomposition helpers."""

from __future__ import annotations

from typing import cast

import torch


def _exact_svd_input(matrix: torch.Tensor) -> torch.Tensor:
    """Return the exact-SVD input without changing CPU/MPS execution policy."""
    value = matrix.float()
    if value.device.type == "mps":
        return value.cpu()
    return value


def exact_svdvals(matrix: torch.Tensor) -> torch.Tensor:
    """Compute all singular values exactly, with a CPU fallback on device failure."""
    value = _exact_svd_input(matrix)
    try:
        return cast(torch.Tensor, torch.linalg.svdvals(value))
    except (RuntimeError, torch.linalg.LinAlgError):
        if value.device.type == "cpu":
            raise
        return cast(torch.Tensor, torch.linalg.svdvals(matrix.float().cpu()))
