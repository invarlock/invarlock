import torch

from invarlock.guards.variance import VarianceGuard


def test_extract_window_ids_defaults_to_index_when_missing():
    g = VarianceGuard()
    batches = [
        {"input_ids": torch.ones(1, 3, dtype=torch.long)},
        {"input_ids": torch.ones(1, 3, dtype=torch.long)},
    ]
    ids = g._extract_window_ids(batches)
    assert ids == ["0", "1"]
