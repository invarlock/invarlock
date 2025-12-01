import torch

from invarlock.guards.variance import VarianceGuard


def test_normalize_pairing_ids_preserves_prefixed():
    g = VarianceGuard()
    ids = g._normalize_pairing_ids("preview", ["preview::1", "x"])
    assert ids[0] == "preview::1" and ids[1].startswith("preview::")


def test_prepare_batch_tensors_all_paths():
    g = VarianceGuard()
    device = torch.device("cpu")
    # Dict path
    x = torch.ones(1, 3, dtype=torch.long)
    d = {"input_ids": x, "attention_mask": torch.ones_like(x)}
    ids, attn = g._prepare_batch_tensors(d, device)
    assert isinstance(ids, torch.Tensor) and isinstance(attn, torch.Tensor)
    # Tuple/list path
    ids2, attn2 = g._prepare_batch_tensors([x, torch.ones_like(x)], device)
    assert isinstance(ids2, torch.Tensor) and isinstance(attn2, torch.Tensor)
    # Else path (scalar tensor)
    ids3, labels3 = g._prepare_batch_tensors(x, device)
    assert isinstance(ids3, torch.Tensor) and isinstance(labels3, torch.Tensor)
