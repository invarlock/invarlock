import torch.nn as nn

from invarlock.guards.variance import VarianceGuard


class NonCallableState:
    # Deliberately provide a non-callable state_dict attribute
    state_dict = 123


class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        # Return a dict with a non-tensor value to flip else branch
        return {"key": "not-a-tensor"}


def test_fingerprint_targets_skips_noncallable_and_handles_non_tensor_values():
    g = VarianceGuard()
    g._target_modules = {
        "x.noncallable": NonCallableState(),  # triggers not callable path
        "y.custom": CustomModule(),  # returns non-tensor value in state
    }
    fp = g._fingerprint_targets()
    # Either returns a hex string or None; ensure invocation succeeded
    assert fp is None or isinstance(fp, str)
