import torch
import torch.nn as nn

from invarlock.guards import variance_batching as variance_batching_mod
from invarlock.guards.variance import VarianceGuard


class TensorOutSameShape(nn.Module):
    def forward(self, x, labels=None):
        # Return a tensor with the same shape as labels to hit MSE path
        if labels is not None:
            return labels.float() * 0.0
        return x.float()


def test_compute_ppl_for_batches_mse_exception_falls_back(monkeypatch):
    g = VarianceGuard()
    device = torch.device("cpu")
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    # Force mse_loss to raise → triggers except branch and continue
    def boom(a, b):
        raise RuntimeError("boom")

    monkeypatch.setattr(variance_batching_mod.torch.nn.functional, "mse_loss", boom)
    ppl, loss = g._compute_ppl_for_batches(TensorOutSameShape(), [batch], device)
    # Sequence may be empty due to continue; assert call ran and returned list types
    assert isinstance(ppl, list) and isinstance(loss, list)
