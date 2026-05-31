import math

import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TensorOutSameShape(nn.Module):
    def forward(self, x, labels=None):
        # Return a tensor with the same shape as labels to hit MSE path
        if labels is not None:
            return labels.float() * 0.0
        return x.float()


class TensorOutDifferentShape(nn.Module):
    def forward(self, x, labels=None):
        B, T = x.shape
        # Different shape to trigger mean-of-squares fallback
        return torch.ones(B, T, 2)


def test_compute_ppl_for_batches_mse_and_mean_square_fallbacks():
    g = VarianceGuard()
    device = torch.device("cpu")
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    ppl_mse, loss_mse = g._compute_ppl_for_batches(
        TensorOutSameShape(), [batch], device
    )
    assert ppl_mse and loss_mse and all(math.isfinite(v) for v in loss_mse)

    ppl_ms, loss_ms = g._compute_ppl_for_batches(
        TensorOutDifferentShape(), [batch], device
    )
    assert ppl_ms and loss_ms and all(math.isfinite(v) for v in loss_ms)
