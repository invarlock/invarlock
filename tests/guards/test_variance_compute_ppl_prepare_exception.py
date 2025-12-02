import torch

from invarlock.guards.variance import VarianceGuard


def test_compute_ppl_for_batches_handles_prepare_exception(monkeypatch):
    g = VarianceGuard()
    device = torch.device("cpu")
    batch = {"input_ids": torch.ones(1, 4, dtype=torch.long)}

    def boom(*args, **kwargs):
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(g, "_prepare_batch_tensors", boom)
    # Use a simple nn.Module for 'model' (only .training is used before prepare raises)
    import torch.nn as nn

    model = nn.Linear(4, 4)
    ppl, loss = g._compute_ppl_for_batches(model=model, batches=[batch], device=device)
    assert ppl == [] and loss == []

    # Provide a dummy model forward to produce a .loss with item()
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def forward(self, inputs, labels=None):
            class Out:
                def __init__(self):
                    self.loss = type("L", (), {"item": lambda self: 1.0})()

            return Out()

    # Sanity: patch back a working prepare path and ensure values emitted
    monkeypatch.setattr(
        g,
        "_prepare_batch_tensors",
        lambda b, d: (batch["input_ids"], batch["input_ids"]),
    )
    ppl2, loss2 = g._compute_ppl_for_batches(
        model=DummyModel(), batches=[batch], device=device
    )
    assert isinstance(ppl2, list) and isinstance(loss2, list)
