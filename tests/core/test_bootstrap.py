import math

from invarlock.core.bootstrap import (
    compute_logloss_ci,
    compute_paired_delta_log_ci,
    logspace_to_ratio_ci,
)


def test_compute_logloss_ci_percentile_and_bca():
    data = [3.0, 3.1, 2.9, 3.05, 3.2, 3.15]
    ci_pct = compute_logloss_ci(
        data, method="percentile", replicates=256, alpha=0.1, seed=123
    )
    ci_bca = compute_logloss_ci(data, method="bca", replicates=256, alpha=0.1, seed=123)
    assert isinstance(ci_pct, tuple) and len(ci_pct) == 2
    assert isinstance(ci_bca, tuple) and len(ci_bca) == 2
    assert ci_pct[0] <= ci_pct[1]
    assert ci_bca[0] <= ci_bca[1]


def test_compute_paired_delta_and_ratio_ci_consistency():
    preview = [3.0, 3.1, 3.2, 3.05]
    final = [3.4, 3.3, 3.25, 3.5]
    dlog_ci = compute_paired_delta_log_ci(
        final, preview, method="bca", replicates=256, alpha=0.1, seed=7
    )
    r_ci = logspace_to_ratio_ci(dlog_ci)
    # exp transform consistency
    assert math.isclose(math.exp(dlog_ci[0]), r_ci[0], rel_tol=1e-6)
    assert math.isclose(math.exp(dlog_ci[1]), r_ci[1], rel_tol=1e-6)
