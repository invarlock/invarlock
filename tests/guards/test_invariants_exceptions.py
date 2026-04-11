import types

import torch
import torch.nn as nn

from invarlock.guards.invariants import (
    InvariantsGuard,
    _check_standard_invariants,
    check_all_invariants,
)


class ModelBadParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)

    def parameters(self, recurse: bool = True):
        raise RuntimeError("parameters() failed")


class ModelBadNamedModules(nn.Module):
    def __init__(self):
        super().__init__()

    def named_modules(self, memo=None, prefix=""):
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


class ModelBadNamedParameters(nn.Module):
    def __init__(self):
        super().__init__()

    def named_parameters(self, prefix="", recurse=True):
        raise RuntimeError("named_parameters failed")


class ModelMidStreamNamedParametersFailure(nn.Module):
    def __init__(self):
        super().__init__()
        self.good = nn.Parameter(torch.ones(1))

    def named_parameters(self, prefix="", recurse=True):
        yield "good", self.good
        raise RuntimeError("named_parameters failed mid-stream")


def test_param_count_exception_path_sets_sentinel():
    guard = InvariantsGuard()
    m = ModelBadParams()
    prep = guard.prepare(m, adapter=None, calib=None, policy={})
    assert prep["ready"] is True
    # Parameter count now records an explicit evidence gap instead of a sentinel.
    assert guard.baseline_checks.get("parameter_count") is None
    assert {
        "check": "parameter_count",
        "reason": "RuntimeError",
    } in guard.baseline_checks.get("evidence_gaps", ())


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


def test_check_standard_invariants_fail_closed_on_parameter_errors():
    checks = _check_standard_invariants(ModelBadParams())
    assert checks["parameter_count"]["passed"] is False
    assert checks["no_nan_parameters"]["passed"] is False


def test_check_all_invariants_rejects_missing_named_parameters() -> None:
    outcome = check_all_invariants(object())
    assert outcome.passed is False
    assert outcome.decision == "block"
    assert outcome.violations[0]["type"] == "structure_violation"


def test_check_all_invariants_fail_closed_when_named_parameters_iteration_errors() -> (
    None
):
    outcome = check_all_invariants(ModelBadNamedParameters())

    assert outcome.passed is False
    assert outcome.decision == "block"
    assert outcome.violations[0]["type"] == "structure_violation"
    assert outcome.metrics["parameters_checked"] == 0


def test_check_all_invariants_fail_closed_when_named_parameters_break_mid_stream() -> (
    None
):
    outcome = check_all_invariants(ModelMidStreamNamedParametersFailure())

    assert outcome.passed is False
    assert outcome.decision == "block"
    assert outcome.violations[0]["type"] == "structure_violation"
    assert outcome.metrics["parameters_checked"] == 0
