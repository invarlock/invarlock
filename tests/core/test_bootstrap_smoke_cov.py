from __future__ import annotations

from invarlock.core.bootstrap import (
    compute_logloss_ci,
    compute_paired_delta_log_ci,
    logspace_to_ratio_ci,
)


def test_bootstrap_smokes() -> None:
    lo, hi = compute_logloss_ci(
        [1.0, 1.1, 0.9], method="percentile", replicates=100, alpha=0.1, seed=0
    )
    assert lo <= hi
    dlo, dhi = compute_paired_delta_log_ci(
        [1.0, 1.1], [0.9, 1.0], method="percentile", replicates=100, alpha=0.1, seed=0
    )
    rlo, rhi = logspace_to_ratio_ci((dlo, dhi))
    assert rlo <= rhi
