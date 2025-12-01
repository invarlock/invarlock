import torch

from invarlock.guards.variance import VarianceGuard


def test_prepare_batch_tensors_uses_inputs_when_input_ids_missing():
    g = VarianceGuard()
    device = torch.device("cpu")
    x = torch.ones(1, 3, dtype=torch.long)
    ids, labels = g._prepare_batch_tensors({"inputs": x}, device)
    assert isinstance(ids, torch.Tensor) and isinstance(labels, torch.Tensor)
