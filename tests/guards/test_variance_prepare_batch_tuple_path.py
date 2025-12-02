import torch

from invarlock.guards.variance import VarianceGuard


def test_prepare_batch_tensors_tuple_branch():
    g = VarianceGuard()
    device = torch.device("cpu")
    x = torch.ones(2, 3, dtype=torch.long)
    ids, labels = g._prepare_batch_tensors((x, x.clone()), device)
    assert isinstance(ids, torch.Tensor) and isinstance(labels, torch.Tensor)
