import torch
import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class NoLabelsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        class Out:
            def __init__(self, v):
                self.loss = v

        class Loss:
            def __init__(self):
                self._v = 0.5

            def item(self):
                return float(self._v)

        return Out(Loss())


def test_compute_ppl_for_batches_fallback_when_typeerror_on_labels():
    g = VarianceGuard()
    device = torch.device("cpu")
    batch = {"input_ids": torch.ones(1, 4, dtype=torch.long)}
    ppl, loss = g._compute_ppl_for_batches(NoLabelsModel(), [batch], device)
    assert ppl and loss and all(v > 0 for v in ppl)
