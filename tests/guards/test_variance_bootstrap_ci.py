import numpy as np
import pytest

from invarlock.guards.variance import VarianceGuard


def test_bootstrap_mean_ci_nominal_and_empty_error():
    g = VarianceGuard()
    samples = list(np.random.default_rng(0).normal(loc=0.0, scale=1.0, size=32))
    lo, hi = g._bootstrap_mean_ci(samples, alpha=0.1, n_bootstrap=200, seed=7)
    assert lo <= hi

    with pytest.raises(ValueError):
        g._bootstrap_mean_ci([], alpha=0.1, n_bootstrap=10, seed=0)
