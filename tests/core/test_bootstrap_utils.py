import math

import numpy as np
import pytest

from invarlock.core import bootstrap as B


def test_ensure_array_and_errors():
    arr = B._ensure_array([1.0, 2.0, 3.0])
    assert isinstance(arr, np.ndarray) and arr.ndim == 1

    with pytest.raises(ValueError):
        B._ensure_array([])
    with pytest.raises(ValueError):
        B._ensure_array([1.0, float("nan")])


def test_compute_logloss_ci_percentile_and_bca():
    samples = [0.1, 0.2, 0.3, 0.4]
    lo, hi = B.compute_logloss_ci(
        samples, method="percentile", replicates=100, alpha=0.1, seed=42
    )
    assert lo <= hi

    # BCa path (small replicates for speed)
    lo2, hi2 = B.compute_logloss_ci(
        samples, method="bca", replicates=100, alpha=0.1, seed=0
    )
    assert lo2 <= hi2


def test_compute_paired_delta_log_ci_and_ratio():
    final = [0.2, 0.2, 0.2, 0.2]
    base = [0.1, 0.1, 0.1, 0.1]
    lo, hi = B.compute_paired_delta_log_ci(
        final, base, method="percentile", replicates=50, alpha=0.1, seed=1
    )
    assert lo <= hi

    # Degenerate equal deltas path returns identical bounds
    lo2, hi2 = B.compute_paired_delta_log_ci(
        [0.5, 0.5], [0.4, 0.4], method="bca", replicates=50, alpha=0.1, seed=0
    )
    assert math.isclose(lo2, hi2)

    # Convert to ratio space
    rlo, rhi = B.logspace_to_ratio_ci((lo, hi))
    assert rlo <= rhi and rlo > 0
