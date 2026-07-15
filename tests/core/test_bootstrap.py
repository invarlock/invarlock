import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from invarlock.core.bootstrap import (
    compute_independent_delta_log_ci,
    compute_logloss_ci,
    compute_paired_delta_log_ci,
    logspace_to_ratio_ci,
)


def _percentile_mean_ci_oracle(
    values: list[float], *, replicates: int, alpha: float, seed: int
) -> tuple[float, float]:
    samples = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.empty(replicates, dtype=float)
    for idx in range(replicates):
        draw = rng.integers(0, samples.size, size=samples.size)
        stats[idx] = float(samples[draw].mean())
    return (
        float(np.percentile(stats, 100.0 * alpha / 2.0)),
        float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0))),
    )


def _weighted_paired_delta_percentile_oracle(
    final: list[float],
    baseline: list[float],
    weights: list[float],
    *,
    replicates: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    delta = np.asarray(final, dtype=float) - np.asarray(baseline, dtype=float)
    sample_weights = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.empty(replicates, dtype=float)
    for idx in range(replicates):
        draw = rng.integers(0, delta.size, size=delta.size)
        stats[idx] = float(
            np.dot(delta[draw], sample_weights[draw])
            / float(sample_weights[draw].sum())
        )
    return (
        float(np.percentile(stats, 100.0 * alpha / 2.0)),
        float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0))),
    )


def _weighted_independent_delta_percentile_oracle(
    final: list[float],
    preview: list[float],
    final_weights: list[float],
    preview_weights: list[float],
    *,
    replicates: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    final_values = np.asarray(final, dtype=float)
    preview_values = np.asarray(preview, dtype=float)
    final_sample_weights = np.asarray(final_weights, dtype=float)
    preview_sample_weights = np.asarray(preview_weights, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.empty(replicates, dtype=float)
    for idx in range(replicates):
        final_draw = rng.integers(0, final_values.size, size=final_values.size)
        preview_draw = rng.integers(0, preview_values.size, size=preview_values.size)
        final_mean = float(
            np.dot(final_values[final_draw], final_sample_weights[final_draw])
            / float(final_sample_weights[final_draw].sum())
        )
        preview_mean = float(
            np.dot(preview_values[preview_draw], preview_sample_weights[preview_draw])
            / float(preview_sample_weights[preview_draw].sum())
        )
        stats[idx] = final_mean - preview_mean
    return (
        float(np.percentile(stats, 100.0 * alpha / 2.0)),
        float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0))),
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


def test_compute_logloss_ci_percentile_matches_seeded_oracle():
    values = [0.8, 1.1, 1.4, 1.7]
    actual = compute_logloss_ci(
        values,
        method="percentile",
        replicates=32,
        alpha=0.1,
        seed=99,
    )
    expected = _percentile_mean_ci_oracle(values, replicates=32, alpha=0.1, seed=99)

    assert actual == pytest.approx(expected, abs=1e-12)


def test_compute_logloss_ci_bca_known_seeded_regression():
    actual = compute_logloss_ci(
        [0.82, 1.03, 1.09, 1.32, 1.55],
        method="bca",
        replicates=512,
        alpha=0.1,
        seed=19,
    )

    assert actual == pytest.approx((1.004, 1.366), abs=1e-12)


def test_compute_paired_delta_log_ci_weighted_percentile_matches_seeded_oracle():
    final = [1.2, 1.7, 2.1, 1.6]
    baseline = [1.0, 1.1, 1.8, 1.5]
    weights = [1.0, 5.0, 2.0, 3.0]
    actual = compute_paired_delta_log_ci(
        final,
        baseline,
        weights=weights,
        method="percentile",
        replicates=40,
        alpha=0.1,
        seed=11,
    )
    expected = _weighted_paired_delta_percentile_oracle(
        final,
        baseline,
        weights,
        replicates=40,
        alpha=0.1,
        seed=11,
    )

    assert actual == pytest.approx(expected, abs=1e-12)


def test_compute_independent_delta_log_ci_resamples_disjoint_arms_independently() -> (
    None
):
    preview = [0.8, 1.1, 1.7]
    final = [1.2, 1.4, 1.9, 2.3]
    preview_weights = [1.0, 2.0, 5.0]
    final_weights = [3.0, 1.0, 4.0, 2.0]

    actual = compute_independent_delta_log_ci(
        final,
        preview,
        final_weights=final_weights,
        preview_weights=preview_weights,
        replicates=64,
        alpha=0.1,
        seed=23,
    )
    expected = _weighted_independent_delta_percentile_oracle(
        final,
        preview,
        final_weights,
        preview_weights,
        replicates=64,
        alpha=0.1,
        seed=23,
    )

    assert actual == pytest.approx(expected, abs=1e-12)


def test_compute_independent_delta_log_ci_accepts_unequal_disjoint_arm_lengths() -> (
    None
):
    lower, upper = compute_independent_delta_log_ci(
        [1.2, 1.4, 1.7, 2.1],
        [0.8, 1.0],
        replicates=32,
        alpha=0.1,
        seed=5,
    )

    assert lower <= upper


def test_compute_paired_delta_and_ratio_ci_consistency():
    baseline_final = [3.0, 3.1, 3.2, 3.05]
    subject_final = [3.4, 3.3, 3.25, 3.5]
    dlog_ci = compute_paired_delta_log_ci(
        subject_final,
        baseline_final,
        method="bca",
        replicates=256,
        alpha=0.1,
        seed=7,
    )
    r_ci = logspace_to_ratio_ci(dlog_ci)
    # exp transform consistency
    assert math.isclose(math.exp(dlog_ci[0]), r_ci[0], rel_tol=1e-6)
    assert math.isclose(math.exp(dlog_ci[1]), r_ci[1], rel_tol=1e-6)


def test_paired_delta_log_ci_rejects_invalid_public_weights():
    with pytest.raises(ValueError, match="weights length"):
        compute_paired_delta_log_ci(
            [1.2, 1.3],
            [1.0, 1.1],
            weights=[1.0],
            method="percentile",
            replicates=32,
        )

    with pytest.raises(ValueError, match="non-negative"):
        compute_paired_delta_log_ci(
            [1.2, 1.3],
            [1.0, 1.1],
            weights=[1.0, -1.0],
            method="percentile",
            replicates=32,
        )


def test_paired_logloss_ratio_ci_known_answer_constant_delta():
    baseline = [1.0, 1.5, 2.0, 2.5]
    delta = math.log(1.05)
    final = [value + delta for value in baseline]
    dlog_ci = compute_paired_delta_log_ci(
        final,
        baseline,
        weights=[1.0, 2.0, 4.0, 8.0],
        method="bca",
        replicates=128,
        alpha=0.05,
        seed=17,
    )
    ratio_ci = logspace_to_ratio_ci(dlog_ci)

    assert dlog_ci == pytest.approx((delta, delta), abs=1e-15)
    assert ratio_ci == pytest.approx((1.05, 1.05), abs=1e-15)


def test_paired_delta_does_not_collapse_small_real_variation() -> None:
    baseline = [1.0, 1.0, 1.0, 1.0]
    final = [1.0, 1.0 + 1e-10, 1.0 + 2e-10, 1.0 + 3e-10]

    lower, upper = compute_paired_delta_log_ci(
        final,
        baseline,
        method="percentile",
        replicates=256,
        alpha=0.1,
        seed=17,
    )

    assert lower < upper


def test_logloss_ci_empirical_coverage_smoke():
    rng = np.random.default_rng(20260708)
    percentile_covered = 0
    bca_covered = 0
    trials = 24

    for trial in range(trials):
        sample = rng.normal(loc=0.0, scale=1.0, size=30)
        pct_lo, pct_hi = compute_logloss_ci(
            sample,
            method="percentile",
            replicates=256,
            alpha=0.1,
            seed=1000 + trial,
        )
        bca_lo, bca_hi = compute_logloss_ci(
            sample,
            method="bca",
            replicates=256,
            alpha=0.1,
            seed=2000 + trial,
        )
        percentile_covered += pct_lo <= 0.0 <= pct_hi
        bca_covered += bca_lo <= 0.0 <= bca_hi

    assert percentile_covered >= 20
    assert bca_covered >= 20


@given(
    values=st.lists(
        st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=12,
    ),
    delta=st.floats(
        min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=25, deadline=None)
def test_paired_delta_log_ci_property_strict_identity(values, delta):
    final = [value + delta for value in values]
    assume(min(final) > 0)
    dlog_ci = compute_paired_delta_log_ci(
        final,
        values,
        method="percentile",
        replicates=64,
        alpha=0.1,
        seed=11,
    )
    ratio_ci = logspace_to_ratio_ci(dlog_ci)
    assert dlog_ci[0] <= dlog_ci[1]
    assert math.isclose(math.exp(dlog_ci[0]), ratio_ci[0], rel_tol=1e-12)
    assert math.isclose(math.exp(dlog_ci[1]), ratio_ci[1], rel_tol=1e-12)


@given(
    left=st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=2, max_size=8),
    extra=st.floats(min_value=0.1, max_value=10.0),
)
@settings(max_examples=20, deadline=None)
def test_paired_delta_log_ci_property_rejects_mismatched_lengths(left, extra):
    right = [*left, extra]
    with pytest.raises(ValueError, match="lengths must match"):
        compute_paired_delta_log_ci(left, right, method="percentile", replicates=32)
