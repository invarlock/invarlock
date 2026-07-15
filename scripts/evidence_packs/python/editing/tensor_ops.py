"""Exact tensor primitive retained by the streaming pruning verifier."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - metadata/spec parsing can run without torch

    class _TorchUnavailable:
        def no_grad(self):
            return lambda func: func

        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError("No module named 'torch'")

    torch = _TorchUnavailable()  # type: ignore[assignment]


@torch.no_grad()
def magnitude_prune_tensor(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    flat_abs = weight.abs().flatten()
    k = int(flat_abs.numel() * sparsity)
    if k == 0:
        return weight
    if k >= flat_abs.numel():
        return torch.zeros_like(weight)

    threshold = torch.kthvalue(flat_abs, k).values
    below = flat_abs < threshold
    ties = flat_abs == threshold
    below_count = int(below.sum().item())
    tie_prune_count = k - below_count

    keep = flat_abs > threshold
    if tie_prune_count < 0:
        raise RuntimeError("magnitude pruning threshold accounting underflowed")
    if tie_prune_count < int(ties.sum().item()):
        tie_indices = torch.nonzero(ties, as_tuple=False).flatten()
        keep[tie_indices[tie_prune_count:]] = True
    return weight * keep.reshape(weight.shape).to(weight.dtype)


__all__ = [
    "magnitude_prune_tensor",
]
