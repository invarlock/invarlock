import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class TensorOutDifferentShape(nn.Module):
    def forward(self, x, labels=None):
        # Different shape than labels to trigger mean-of-squares path
        B, T = x.shape
        return torch.ones(B, T, 2)


def test_compute_ppl_for_batches_mean_square_exception_falls_back(monkeypatch):
    g = VarianceGuard()
    device = torch.device("cpu")
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    # Force Tensor.pow to raise inside the mean-of-squares fallback
    orig_pow = torch.Tensor.pow

    def boom(self, *args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("boom")

    monkeypatch.setattr(torch.Tensor, "pow", boom, raising=True)
    try:
        ppl, loss = g._compute_ppl_for_batches(
            TensorOutDifferentShape(), [batch], device
        )
        # Sequence may be empty due to continue; assert call ran and returned list types
        assert isinstance(ppl, list) and isinstance(loss, list)
    finally:
        # Restore to avoid side-effects beyond this test
        monkeypatch.setattr(torch.Tensor, "pow", orig_pow, raising=True)
