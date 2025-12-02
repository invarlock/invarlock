from __future__ import annotations

import math

import pytest

from invarlock.eval.bootstrap import paired_delta_mean_ci
from invarlock.eval.metrics import ValidationError as MValidationError


def test_paired_delta_mean_ci_invalid_method_raises():
    with pytest.raises(MValidationError):
        paired_delta_mean_ci([1.0, 2.0], [1.0, 1.5], method="bad")


@pytest.mark.parametrize("method", ["bca", "percentile"])
def test_paired_delta_mean_ci_basic_shapes(method: str):
    lo, hi = paired_delta_mean_ci(
        [1.0, 2.0, 3.0], [1.0, 1.5, 2.5], reps=100, seed=0, method=method
    )
    assert isinstance(lo, float) and isinstance(hi, float)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo <= hi
