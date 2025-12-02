import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_fingerprint_targets_returns_hash():
    g = VarianceGuard()
    model = nn.Linear(2, 2, bias=False)
    g._target_modules = {"transformer.h.0.mlp.c_proj": model}
    fp = g._fingerprint_targets()
    assert fp is not None and isinstance(fp, str) and len(fp) > 0
