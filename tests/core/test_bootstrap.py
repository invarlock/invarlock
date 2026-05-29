import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
    if min(final) <= 0:
        pytest.skip("generated delta pushed log-loss below positive domain")
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
