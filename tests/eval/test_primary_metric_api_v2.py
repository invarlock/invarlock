from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from invarlock.core.bootstrap import compute_paired_delta_log_ci
from invarlock.eval.primary_metric import get_metric


def _mk_contribs(values: Iterable[float]) -> list[dict[str, float]]:
    # Minimal contribution representation used by the new API: value + weight
    return [{"value": float(v), "weight": 1.0} for v in values]


def test_ppl_paired_compare_ratio_ci_matches_exp_delta_ci():
    rng = np.random.default_rng(0)
    # Build paired per-example log-loss arrays with a small improvement
    base_log = rng.normal(loc=math.log(10.0), scale=0.05, size=64)
    subj_log = base_log - rng.normal(loc=0.02, scale=0.01, size=64)

    m = get_metric("ppl_causal")
    res = m.paired_compare(
        subject=_mk_contribs(subj_log),
        baseline=_mk_contribs(base_log),
        reps=512,
        seed=7,
        ci_level=0.90,
    )

    # Golden: log-space delta CI, then exponentiate → ratio CI
    dlog_ci = compute_paired_delta_log_ci(
        subj_log, base_log, method="bca", replicates=512, alpha=0.10, seed=7
    )
    exp_ci = (math.exp(dlog_ci[0]), math.exp(dlog_ci[1]))

    rlo, rhi = res["display_ci"]
    assert math.isclose(rlo, exp_ci[0], rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(rhi, exp_ci[1], rel_tol=1e-6, abs_tol=1e-6)


def test_accuracy_paired_ci_narrower_than_unpaired_naive():
    rng = np.random.default_rng(1)
    n = 200
    # Baseline ~70% accuracy; subject improves a bit on the same examples
    base_flags = (rng.random(n) < 0.70).astype(float)
    improvements = (rng.random(n) < 0.05).astype(float)
    subj_flags = np.clip(base_flags + improvements, 0, 1)

    m = get_metric("accuracy")
    res = m.paired_compare(
        subject=_mk_contribs(subj_flags),
        baseline=_mk_contribs(base_flags),
        reps=800,
        seed=13,
        ci_level=0.90,
    )

    # Paired CI in display space is percentage points
    paired_lo, paired_hi = res["display_ci"]
    paired_width = paired_hi - paired_lo

    # Naive unpaired bootstrap CI on delta of means (independent resamples)
    reps = 800
    alpha = 0.10
    rng2 = np.random.default_rng(13)
    stats = np.empty(reps)
    for i in range(reps):
        idx_a = rng2.integers(0, n, size=n)
        idx_b = rng2.integers(0, n, size=n)
        stats[i] = (
            float(np.mean(subj_flags[idx_a]) - np.mean(base_flags[idx_b])) * 100.0
        )
    stats.sort()
    lo = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    unpaired_width = hi - lo

    # Paired should not be wider than naive unpaired (usually narrower)
    assert paired_width <= unpaired_width + 1e-6
