import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class NaNLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, inputs, labels=None):
        class Out:
            def __init__(self):
                self.loss = self

            def item(self):
                return float("nan")

        return Out()


def test_compute_ppl_skips_nonfinite_loss_entries():
    g = VarianceGuard()
    device = torch.device("cpu")
    batch = {"input_ids": torch.ones(1, 4, dtype=torch.long)}
    ppl, loss = g._compute_ppl_for_batches(NaNLossModel(), [batch], device)
    assert ppl == [] and loss == []
