import pytest

from invarlock.core.bootstrap import (
    compute_logloss_ci,
    compute_paired_delta_log_ci,
)


def test_bootstrap_percentile_and_degenerate_cases():
    # Percentile method
    lo, hi = compute_logloss_ci(
        [0.9, 1.0, 1.1], method="percentile", replicates=100, seed=0
    )
    assert lo <= hi

    # Degenerate paired delta: equal arrays -> identical bounds
    lo2, hi2 = compute_paired_delta_log_ci(
        [1.0, 1.0], [1.0, 1.0], replicates=100, seed=0
    )
    assert lo2 == hi2 == 0.0


def test_bootstrap_invalid_replicates_raises():
    with pytest.raises(ValueError):
        compute_logloss_ci([1.0, 1.1], replicates=0)


def test_bootstrap_invalid_alpha_raises():
    with pytest.raises(ValueError):
        compute_logloss_ci([1.0, 1.1], alpha=1.5)
    with pytest.raises(ValueError):
        compute_paired_delta_log_ci([1.0, 1.1], [1.0, 1.0], alpha=-0.1)
