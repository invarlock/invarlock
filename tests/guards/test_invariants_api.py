import pytest
import torch
import torch.nn as nn

from invarlock.guards.invariants import assert_invariants, check_all_invariants


class SmallModel(nn.Module):
    def __init__(self, bad=False, large=False):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, 2))
        if bad:
            with torch.no_grad():
                self.w[0, 0] = float("nan")
        if large:
            with torch.no_grad():
                self.w.mul_(2000.0)


def test_check_all_invariants_ok_and_violations():
    ok_model = SmallModel()
    res_ok = check_all_invariants(ok_model)
    assert res_ok.passed is True

    nan_model = SmallModel(bad=True)
    res_nan = check_all_invariants(nan_model)
    assert res_nan.passed is False

    large_model = SmallModel(large=True)
    res_large = check_all_invariants(large_model)
    assert res_large.passed is False


def test_assert_invariants_raises():
    with pytest.raises(AssertionError):
        assert_invariants(SmallModel(bad=True))
