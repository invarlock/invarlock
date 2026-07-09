import numpy as np
import pytest

from invarlock.core.bootstrap import compute_logloss_ci, compute_paired_delta_log_ci

pytestmark = pytest.mark.slow


def test_logloss_ci_empirical_coverage_higher_power() -> None:
    rng = np.random.default_rng(20260709)
    trials = 240
    percentile_covered = 0
    bca_covered = 0

    for trial in range(trials):
        sample = rng.normal(loc=0.0, scale=1.0, size=36)
        pct_lo, pct_hi = compute_logloss_ci(
            sample,
            method="percentile",
            replicates=512,
            alpha=0.1,
            seed=10_000 + trial,
        )
        bca_lo, bca_hi = compute_logloss_ci(
            sample,
            method="bca",
            replicates=512,
            alpha=0.1,
            seed=20_000 + trial,
        )
        percentile_covered += pct_lo <= 0.0 <= pct_hi
        bca_covered += bca_lo <= 0.0 <= bca_hi

    assert 190 <= percentile_covered <= 215
    assert 190 <= bca_covered <= 215


def test_paired_delta_log_ci_empirical_coverage_higher_power() -> None:
    rng = np.random.default_rng(20260710)
    trials = 180
    percentile_covered = 0
    bca_covered = 0

    for trial in range(trials):
        baseline = rng.normal(loc=1.25, scale=0.35, size=32)
        final = baseline + rng.normal(loc=0.0, scale=0.18, size=32)
        pct_lo, pct_hi = compute_paired_delta_log_ci(
            final,
            baseline,
            method="percentile",
            replicates=384,
            alpha=0.1,
            seed=30_000 + trial,
        )
        bca_lo, bca_hi = compute_paired_delta_log_ci(
            final,
            baseline,
            method="bca",
            replicates=384,
            alpha=0.1,
            seed=40_000 + trial,
        )
        percentile_covered += pct_lo <= 0.0 <= pct_hi
        bca_covered += bca_lo <= 0.0 <= bca_hi

    assert 152 <= percentile_covered <= 172
    assert 152 <= bca_covered <= 172
