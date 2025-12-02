import types

import torch.nn as nn

from invarlock.guards.invariants import InvariantsGuard


class ModelBadParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        raise RuntimeError("parameters() failed")


class ModelBadNamedModules(nn.Module):
    def __init__(self):
        super().__init__()

    def named_modules(self, memo=None, prefix=""):  # type: ignore[override]
        raise RuntimeError("named_modules failed")


class WeightLikeNoPtr:
    # Intentionally missing data_ptr to trigger exception in _is_tied
    pass


class ModelBadTie(nn.Module):
    def __init__(self):
        super().__init__()
        # Provide GPT-2 style attributes
        self.transformer = types.SimpleNamespace(
            wte=types.SimpleNamespace(weight=WeightLikeNoPtr())
        )
        self.lm_head = types.SimpleNamespace(weight=WeightLikeNoPtr())


def test_param_count_exception_path_sets_sentinel():
    guard = InvariantsGuard()
    m = ModelBadParams()
    prep = guard.prepare(m, adapter=None, calib=None, policy={})
    assert prep["ready"] is True
    # Parameter count should be sentinel -1 when exception encountered
    assert guard.baseline_checks.get("parameter_count") == -1


def test_named_modules_exception_yields_empty_layer_norm_paths():
    guard = InvariantsGuard()
    m = ModelBadNamedModules()
    guard.prepare(m, adapter=None, calib=None, policy={})
    # layer_norm_paths should exist and be empty tuple after exception
    assert guard.baseline_checks.get("layer_norm_paths") == ()


def test_weight_tying_exception_path_returns_false_not_crash():
    guard = InvariantsGuard()
    m = ModelBadTie()
    # Should not raise during capture; weight_tying becomes None or Falsey
    guard.prepare(m, adapter=None, calib=None, policy={})
    wt = guard.baseline_checks.get("weight_tying")
    # Either not applicable (None) or False due to _is_tied exception path
    assert wt in (None, False)
