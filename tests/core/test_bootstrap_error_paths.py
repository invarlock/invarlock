import pytest

from invarlock.core.bootstrap import (
    compute_logloss_ci,
    compute_paired_delta_log_ci,
)


def test_ensure_array_error_paths():
    # Non-1D array should raise
    with pytest.raises(ValueError):
        compute_logloss_ci([[1.0, 1.1], [0.9, 1.0]], replicates=10)

    # Empty iterable should raise
    with pytest.raises(ValueError):
        compute_logloss_ci([], replicates=10)

    # Non-finite values should raise
    with pytest.raises(ValueError):
        compute_logloss_ci([1.0, float("nan"), 0.9], replicates=10)


def test_bca_fallback_to_percentile_when_degenerate():
    # Identical samples lead to zero jackknife denominator → percentile fallback
    lo, hi = compute_logloss_ci(
        [1.23, 1.23, 1.23], method="bca", replicates=50, seed=123
    )
    # Interval collapses to a single point
    assert lo == pytest.approx(1.23)
    assert hi == pytest.approx(1.23)


def test_unknown_bootstrap_method_raises():
    with pytest.raises(ValueError):
        compute_logloss_ci([1.0, 1.1, 0.9], method="unknown", replicates=10)


def test_paired_delta_size_mismatch_slices_to_minimum():
    # Different lengths: function should slice to common length rather than error
    final = [1.0, 1.05, 0.98]
    baseline = [1.02, 1.01]
    lo, hi = compute_paired_delta_log_ci(
        final, baseline, method="percentile", replicates=50, seed=7
    )
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo <= hi
