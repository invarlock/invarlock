import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


def test_enable_disable_idempotent_paths():
    g = VarianceGuard()
    model = nn.Linear(2, 2, bias=False)

    # Idempotent enable when already enabled should be a no-op and return a bool
    g._enabled = True
    out = g.enable(model)
    assert isinstance(out, bool)

    # Idempotent disable when already disabled should be a no-op and return a bool
    g._enabled = False
    out2 = g.disable(model)
    assert isinstance(out2, bool)
