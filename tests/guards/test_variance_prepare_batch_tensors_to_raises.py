import torch

from invarlock.guards.variance import VarianceGuard


def test_prepare_batch_tensors_falls_back_when_to_raises(monkeypatch):
    g = VarianceGuard()
    device = torch.device("cpu")
    x = torch.ones(1, 3, dtype=torch.long)

    # Monkeypatch Tensor.to to raise for this test to hit the except branches
    original_to = torch.Tensor.to

    def boom(self, *args, **kwargs):
        raise RuntimeError("to failed")

    monkeypatch.setattr(torch.Tensor, "to", boom)
    try:
        ids, labels = g._prepare_batch_tensors(
            {"input_ids": x, "attention_mask": x.clone()}, device
        )
        # Should still return tensors due to clone() fallbacks
        assert isinstance(ids, torch.Tensor) and isinstance(labels, torch.Tensor)
    finally:
        # Restore to to avoid side effects on rest of test suite
        monkeypatch.setattr(torch.Tensor, "to", original_to)
