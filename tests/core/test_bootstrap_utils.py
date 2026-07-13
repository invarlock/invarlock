import math

import numpy as np
import pytest

from invarlock.core import bootstrap as B


def test_ensure_array_and_errors():
    arr = B._ensure_array([1.0, 2.0, 3.0])
    assert isinstance(arr, np.ndarray) and arr.ndim == 1

    with pytest.raises(ValueError):
        B._ensure_array([])
    with pytest.raises(ValueError):
        B._ensure_array([1.0, float("nan")])


def test_compute_logloss_ci_percentile_and_bca():
    samples = [0.1, 0.2, 0.3, 0.4]
    lo, hi = B.compute_logloss_ci(
        samples, method="percentile", replicates=100, alpha=0.1, seed=42
    )
    assert lo <= hi

    # BCa path (small replicates for speed)
    lo2, hi2 = B.compute_logloss_ci(
        samples, method="bca", replicates=100, alpha=0.1, seed=0
    )
    assert lo2 <= hi2


def test_compute_paired_delta_log_ci_and_ratio():
    final = [0.2, 0.2, 0.2, 0.2]
    base = [0.1, 0.1, 0.1, 0.1]
    lo, hi = B.compute_paired_delta_log_ci(
        final, base, method="percentile", replicates=50, alpha=0.1, seed=1
    )
    assert lo <= hi

    # Degenerate equal deltas path returns identical bounds
    lo2, hi2 = B.compute_paired_delta_log_ci(
        [0.5, 0.5], [0.4, 0.4], method="bca", replicates=50, alpha=0.1, seed=0
    )
    assert math.isclose(lo2, hi2)

    # Convert to ratio space
    rlo, rhi = B.logspace_to_ratio_ci((lo, hi))
    assert rlo <= rhi and rlo > 0


def test_compute_paired_delta_log_ci_resamples_windows_and_weights_statistic():
    final = [0.2, 0.4, 0.6]
    base = [0.1, 0.1, 0.1]
    weights = [1.0, 10.0, 1.0]
    seed = 123
    lo, hi = B.compute_paired_delta_log_ci(
        final,
        base,
        weights=weights,
        method="percentile",
        replicates=1,
        alpha=0.1,
        seed=seed,
    )
    delta = np.array([f - b for f, b in zip(final, base, strict=False)], dtype=float)
    sample_weights = np.array(weights, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=len(delta))
    expected = float(
        np.dot(delta[idx], sample_weights[idx]) / sample_weights[idx].sum()
    )
    assert lo == pytest.approx(expected)
    assert hi == pytest.approx(expected)


def test_weight_helpers_cover_invalid_inputs_and_zero_total_path():
    assert B._normalize_weights(None, 3) is None
    assert B._normalize_weights([1.0, 2.0], 3) is None
    assert B._normalize_weights([1.0, float("nan"), 2.0], 3) is None
    assert B._normalize_weights([1.0, -1.0, 2.0], 3) is None
    assert B._normalize_weights([0.0, 0.0, 0.0], 3) is None
    assert B._normalize_weights([2.0, 2.0, 2.0], 3) is None
    weights = B._normalize_weights([1.0, 2.0, 3.0], 3)
    assert weights is not None
    assert float(weights.sum()) == pytest.approx(1.0)

    mean = B._weighted_mean(
        np.array([1.0, 3.0], dtype=float), np.array([0.0, 0.0], dtype=float)
    )
    assert mean == pytest.approx(2.0)


def test_strict_weight_helpers_raise_on_invalid_inputs_and_zero_pairs():
    with pytest.raises(ValueError, match="1-dimensional"):
        B._normalize_weights_strict([[1.0], [2.0]], 2)
    with pytest.raises(ValueError, match="finite"):
        B._normalize_weights_strict([1.0, float("nan")], 2)
    with pytest.raises(ValueError, match="non-negative"):
        B._normalize_weights_strict([1.0, -1.0], 2)
    with pytest.raises(ValueError, match="positive sum"):
        B._normalize_weights_strict([0.0, 0.0], 2)


def test_compute_paired_delta_log_ci_returns_zero_for_empty_trimmed_pairs(
    monkeypatch: pytest.MonkeyPatch,
):
    arrays = iter(
        [
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        ]
    )
    monkeypatch.setattr(B, "_ensure_array", lambda _samples: next(arrays))

    lo, hi = B.compute_paired_delta_log_ci(
        [],
        [],
        strict_lengths=False,
    )
    assert (lo, hi) == (0.0, 0.0)


def test_weighted_bootstrap_helpers_cover_small_n_and_validation_errors():
    rng = np.random.default_rng(0)
    lo, hi = B._bca_interval_weighted(
        np.array([2.0], dtype=float),
        weights=np.array([1.0], dtype=float),
        replicates=10,
        alpha=0.1,
        rng=rng,
    )
    assert lo == hi == pytest.approx(2.0)

    with pytest.raises(ValueError):
        B._bootstrap_mean_ci_weighted(
            np.array([1.0, 2.0], dtype=float),
            np.array([0.2, 0.8], dtype=float),
            method="percentile",
            replicates=0,
            alpha=0.1,
            seed=0,
        )
    with pytest.raises(ValueError):
        B._bootstrap_mean_ci_weighted(
            np.array([1.0, 2.0], dtype=float),
            np.array([0.2, 0.8], dtype=float),
            method="percentile",
            replicates=10,
            alpha=1.5,
            seed=0,
        )
    with pytest.raises(ValueError):
        B._bootstrap_mean_ci_weighted(
            np.array([1.0, 2.0], dtype=float),
            np.array([0.2, 0.8], dtype=float),
            method="unknown",
            replicates=10,
            alpha=0.1,
            seed=0,
        )


def test_weighted_bootstrap_bca_handles_single_dominant_weight_and_short_weights():
    lo, hi = B._bootstrap_mean_ci_weighted(
        np.array([1.0, 3.0], dtype=float),
        np.array([1.0, 0.0], dtype=float),
        method="bca",
        replicates=32,
        alpha=0.1,
        seed=0,
    )
    assert lo <= hi

    lo2, hi2 = B.compute_paired_delta_log_ci(
        [1.2, 1.3, 1.4],
        [1.0, 1.1, 1.2],
        weights=[10.0],
        method="percentile",
        replicates=10,
        alpha=0.1,
        seed=0,
        strict_lengths=False,
    )
    assert lo2 <= hi2


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([[1.0], [2.0]], "1-dimensional"),
        ([1.0], "length must match"),
        ([1.0, float("inf")], "finite"),
        ([1.0, -1.0], "non-negative"),
        ([0.0, 0.0], "positive sum"),
    ],
)
def test_independent_delta_rejects_invalid_arm_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        B.compute_independent_delta_log_ci(
            [0.3, 0.4],
            [0.1, 0.2],
            final_weights=weights,
            replicates=8,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "bca"}, "only the percentile method"),
        ({"replicates": 0}, "replicates must be positive"),
        ({"alpha": 1.0}, "alpha must be between"),
    ],
)
def test_independent_delta_rejects_invalid_bootstrap_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        B.compute_independent_delta_log_ci([0.3], [0.1], **kwargs)


def test_paired_delta_rejects_non_finite_subtraction_result():
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="deltas must be finite"):
            B.compute_paired_delta_log_ci([1e308], [-1e308])
