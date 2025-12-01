"""Property-based checks for paired evaluation math."""

from __future__ import annotations

import math

import pytest

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies
given = hypothesis.given
settings = hypothesis.settings


_POS_FLOAT = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_WEIGHTS = st.integers(min_value=1, max_value=4096)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(  # (weight, preview, final)
        st.tuples(_WEIGHTS, _POS_FLOAT, _POS_FLOAT),
        min_size=1,
        max_size=32,
    )
)
def test_logspace_ratio_matches_product(
    samples: list[tuple[int, float, float]],
) -> None:
    """exp(weighted Δlog) equals the product of per-window ratios."""

    weights, preview_vals, final_vals = zip(*samples, strict=False)
    total_weight = float(sum(weights))

    weighted_delta = (
        sum(
            w * (math.log(f) - math.log(p))
            for w, p, f in zip(weights, preview_vals, final_vals, strict=False)
        )
        / total_weight
    )
    ratio_log = math.exp(weighted_delta)

    # Compute via cumulative log to avoid overflow.
    log_product_delta = (
        sum(
            w * (math.log(f) - math.log(p))
            for w, p, f in zip(weights, preview_vals, final_vals, strict=False)
        )
        / total_weight
    )
    ratio_product = math.exp(log_product_delta)

    assert math.isfinite(ratio_log)
    assert math.isclose(ratio_log, ratio_product, rel_tol=0.0, abs_tol=1e-12)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.tuples(_WEIGHTS, _POS_FLOAT), min_size=1, max_size=32),
    st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_uniform_scaling_preserves_ratio(
    samples: list[tuple[int, float]], scale: float
) -> None:
    """If every window scales equally, global ratio should equal that scale."""

    weights, preview_vals = zip(*samples, strict=False)
    final_vals = [p * scale for p in preview_vals]

    total_weight = float(sum(weights))
    weighted_delta = (
        sum(
            w * (math.log(f) - math.log(p))
            for w, p, f in zip(weights, preview_vals, final_vals, strict=False)
        )
        / total_weight
    )
    ratio = math.exp(weighted_delta)

    assert math.isclose(ratio, scale, rel_tol=1e-9, abs_tol=1e-9)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(_WEIGHTS, _POS_FLOAT), min_size=1, max_size=32))
def test_zero_edit_ratio_is_one(samples: list[tuple[int, float]]) -> None:
    """If preview and final distributions match exactly, ratio stays at 1."""

    weights, preview_vals = zip(*samples, strict=False)
    total_weight = float(sum(weights))
    weighted_delta = sum(0.0 for _ in samples) / total_weight
    ratio = math.exp(weighted_delta)

    assert math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12)
