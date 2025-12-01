from __future__ import annotations

import math

from invarlock.reporting.utils import _weighted_mean


def test_weighted_mean_variants() -> None:
    # Normal path
    assert math.isclose(_weighted_mean([1.0, 2.0, 3.0], [1.0, 1.0, 2.0]), 2.25)
    # Mismatched lengths → NaN
    assert math.isnan(_weighted_mean([1.0, 2.0], [1.0]))
    # Non-finite and non-positive weights ignored; leaves only one valid term
    val = _weighted_mean([1.0, 100.0, float("nan")], [0.0, float("nan"), 1.0])
    assert math.isnan(val)
