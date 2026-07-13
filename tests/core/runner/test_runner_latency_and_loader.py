from types import SimpleNamespace

import torch

from invarlock.core.runner import CoreRunner


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None):
        # Simple constant loss
        return SimpleNamespace(loss=torch.tensor(0.01, device=input_ids.device))


def test_measure_latency_with_token_type_ids():
    runner = CoreRunner()
    model = TinyModel().eval()
    device = next(model.parameters()).device

    sample = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "token_type_ids": [0, 0, 1, 1],
    }
    latency = runner._measure_latency(model, [sample], device)
    assert isinstance(latency, float)
    assert latency >= 0.0


def test_samples_to_dataloader_includes_token_type_and_labels_tensorization():
    runner = CoreRunner()
    samples = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 1],
            "labels": [1, -100, 3],
        }
    ]
    dl = runner._samples_to_dataloader(samples)
    batches = list(iter(dl))
    assert len(batches) == 1
    b = batches[0]
    # Shapes are batched
    assert b["input_ids"].dim() == 2
    assert b["attention_mask"].dim() == 2
    assert b["token_type_ids"].dim() == 2
    assert b["labels"].dim() == 2
