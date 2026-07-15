"""
InvarLock Core Bootstrap Utilities
==============================

Numerically stable bootstrap helpers for evaluation metrics.

This module provides paired BCa intervals for baseline/subject comparisons and
an independent two-slice percentile interval for disjoint preview/final data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import NormalDist

import numpy as np

from invarlock.core.exceptions import ValidationError

__all__ = [
    "INDEPENDENT_SLICE_BOOTSTRAP_METHOD",
    "INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET",
    "PAIRED_BASELINE_BOOTSTRAP_METHOD",
    "PAIRED_BASELINE_BOOTSTRAP_SEED_OFFSET",
    "compute_independent_delta_log_ci",
    "compute_logloss_ci",
    "compute_paired_delta_log_ci",
    "logspace_to_ratio_ci",
    "paired_delta_mean_ci",
]


Normal = NormalDist()

# Report generation records the external method name while the numerical helper
# accepts the shorter implementation name.  Keep the external name and seed
# derivation centralized so strict verification can replay the exact producer
# algorithm without copying magic values.
PAIRED_BASELINE_BOOTSTRAP_METHOD = "bca_paired_delta_log"
PAIRED_BASELINE_BOOTSTRAP_SEED_OFFSET = 503
INDEPENDENT_SLICE_BOOTSTRAP_METHOD = "independent_percentile_delta_log"
INDEPENDENT_SLICE_BOOTSTRAP_SEED_OFFSET = 97


def _ensure_array(samples: Iterable[float]) -> np.ndarray:
    """Coerce iterable of floats to a 1-D NumPy array."""
    arr = np.asarray(list(samples), dtype=float)
    if arr.ndim != 1:
        raise ValueError("samples must be 1-dimensional")
    if arr.size == 0:
        raise ValueError("samples cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("samples must be finite")
    return arr


def _normalize_weights(weights: Iterable[float] | None, n: int) -> np.ndarray | None:
    if weights is None:
        return None
    arr = np.asarray(list(weights), dtype=float)
    if arr.ndim != 1 or arr.size != n:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if np.any(arr < 0):
        return None
    total = float(arr.sum())
    if total <= 0.0:
        return None
    if np.allclose(arr, arr[0]):
        return None
    return arr / total


def _normalize_weights_strict(weights: Iterable[float], n: int) -> np.ndarray | None:
    arr = np.asarray(list(weights), dtype=float)
    if arr.ndim != 1:
        raise ValueError("weights must be 1-dimensional")
    if arr.size != n:
        raise ValueError("weights length must match paired samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError("weights must be finite")
    if np.any(arr < 0):
        raise ValueError("weights must be non-negative")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("weights must have a positive sum")
    if np.allclose(arr, arr[0]):
        return None
    return arr / total


def _weighted_mean(samples: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return float(np.mean(samples))
    return float(np.dot(samples, weights) / total)


_BOOTSTRAP_CHUNK_BYTES = 16 * 1024 * 1024


def _bootstrap_chunk_rows(
    sample_width: int,
    replicates: int,
    *,
    arrays_per_index: int,
) -> int:
    """Bound temporary bootstrap index/value matrices to a fixed memory budget."""

    bytes_per_row = max(sample_width, 1) * 8 * max(arrays_per_index, 1)
    return max(1, min(replicates, _BOOTSTRAP_CHUNK_BYTES // bytes_per_row))


def _resampled_mean_statistics(
    samples: np.ndarray,
    *,
    weights: np.ndarray | None,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute bootstrap means in bounded chunks while preserving RNG draw order."""

    size = int(samples.size)
    statistics = np.empty(replicates, dtype=float)
    rows = _bootstrap_chunk_rows(
        size,
        replicates,
        arrays_per_index=3 if weights is not None else 2,
    )
    for start in range(0, replicates, rows):
        stop = min(start + rows, replicates)
        selected = rng.integers(0, size, size=(stop - start, size))
        selected_values = samples[selected]
        if weights is None:
            statistics[start:stop] = np.mean(selected_values, axis=1)
            continue
        selected_weights = weights[selected]
        denominators = np.sum(selected_weights, axis=1)
        numerators = np.sum(selected_values * selected_weights, axis=1)
        chunk_statistics = np.mean(selected_values, axis=1)
        np.divide(
            numerators,
            denominators,
            out=chunk_statistics,
            where=denominators > 0.0,
        )
        statistics[start:stop] = chunk_statistics
    return statistics


def _constant_within_subtraction_roundoff(
    delta: np.ndarray,
    final: np.ndarray,
    baseline: np.ndarray,
) -> bool:
    """Distinguish arithmetic roundoff from small but real paired variation."""

    spread = float(np.max(delta) - np.min(delta))
    operand_scale = max(
        1.0,
        float(np.max(np.abs(final))),
        float(np.max(np.abs(baseline))),
    )
    roundoff_limit = 8.0 * np.finfo(np.float64).eps * operand_scale
    return bool(spread <= roundoff_limit)


def _percentile_interval(stats: np.ndarray, alpha: float) -> tuple[float, float]:
    """Return lower/upper bounds from an array of bootstrap statistics."""
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(stats, lower_q)), float(np.percentile(stats, upper_q))


def _bca_interval_weighted(
    samples: np.ndarray,
    *,
    weights: np.ndarray,
    replicates: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute a BCa interval for a token-weighted mean over window clusters.

    Windows are the independent resampling units.  Each bootstrap replicate
    samples windows uniformly and then recomputes the token-weighted statistic
    from the sampled window values and weights.  Sampling windows with
    probability proportional to token count and then taking an unweighted mean
    would change both the estimand and its sampling variance.
    """
    n = samples.size
    if n < 2:
        stat = _weighted_mean(samples, weights)
        return float(stat), float(stat)

    stats = _resampled_mean_statistics(
        samples,
        weights=weights,
        replicates=replicates,
        rng=rng,
    )

    stats.sort()
    stat_hat = _weighted_mean(samples, weights)

    prop = np.clip((stats < stat_hat).mean(), 1e-6, 1.0 - 1e-6)
    z0 = Normal.inv_cdf(prop)

    sum_w = float(weights.sum())
    sum_wx = float(np.dot(samples, weights))
    jack = np.empty(n, dtype=float)
    for i in range(n):
        w_i = float(weights[i])
        denom = sum_w - w_i
        if denom <= 0.0:
            jack[i] = stat_hat
        else:
            jack[i] = (sum_wx - w_i * float(samples[i])) / denom

    jack_mean = jack.mean()
    numerator = np.sum((jack_mean - jack) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    if denominator == 0.0:
        return _percentile_interval(stats, alpha)

    acc = numerator / denominator

    def _adjust_quantile(z_alpha: float) -> float:
        adj = z0 + (z0 + z_alpha) / max(1.0 - acc * (z0 + z_alpha), 1e-12)
        return float(Normal.cdf(adj))

    lower_pct = _adjust_quantile(Normal.inv_cdf(alpha / 2.0))
    upper_pct = _adjust_quantile(Normal.inv_cdf(1.0 - alpha / 2.0))

    return float(np.quantile(stats, lower_pct)), float(np.quantile(stats, upper_pct))


def _bca_interval(
    samples: np.ndarray,
    *,
    replicates: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Compute a BCa interval for the given statistic.

    Based on Efron & Tibshirani (1994). Handles small-sample edge cases by
    falling back to percentile intervals when the acceleration term cannot
    be computed (e.g., duplicate samples).
    """
    n = samples.size
    if n < 2:
        stat = float(np.mean(samples))
        return float(stat), float(stat)

    stats = _resampled_mean_statistics(
        samples,
        weights=None,
        replicates=replicates,
        rng=rng,
    )

    stats.sort()
    stat_hat = float(np.mean(samples))

    # Bias-correction
    prop = np.clip((stats < stat_hat).mean(), 1e-6, 1.0 - 1e-6)
    z0 = Normal.inv_cdf(prop)

    # Jackknife estimates for acceleration
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack_sample = np.delete(samples, i)
        jack[i] = float(np.mean(jack_sample))

    jack_mean = jack.mean()
    numerator = np.sum((jack_mean - jack) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    if denominator == 0.0:
        # Degenerate case → revert to percentile interval
        return _percentile_interval(stats, alpha)

    acc = numerator / denominator

    def _adjust_quantile(z_alpha: float) -> float:
        adj = z0 + (z0 + z_alpha) / max(1.0 - acc * (z0 + z_alpha), 1e-12)
        return float(Normal.cdf(adj))

    lower_pct = _adjust_quantile(Normal.inv_cdf(alpha / 2.0))
    upper_pct = _adjust_quantile(Normal.inv_cdf(1.0 - alpha / 2.0))

    return float(np.quantile(stats, lower_pct)), float(np.quantile(stats, upper_pct))


def _bootstrap_mean_ci_weighted(
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    method: str,
    replicates: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)
    if method == "percentile":
        stats = _resampled_mean_statistics(
            samples,
            weights=weights,
            replicates=replicates,
            rng=rng,
        )
        stats.sort()
        return _percentile_interval(stats, alpha)
    if method == "bca":
        return _bca_interval_weighted(
            samples,
            weights=weights,
            replicates=replicates,
            alpha=alpha,
            rng=rng,
        )

    raise ValueError(f"Unsupported bootstrap method '{method}'")


def _bootstrap_interval(
    samples: np.ndarray,
    *,
    method: str,
    replicates: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Dispatch helper supporting percentile and BCa intervals."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)
    if method == "percentile":
        stats = _resampled_mean_statistics(
            samples,
            weights=None,
            replicates=replicates,
            rng=rng,
        )
        stats.sort()
        return _percentile_interval(stats, alpha)
    if method == "bca":
        return _bca_interval(
            samples,
            replicates=replicates,
            alpha=alpha,
            rng=rng,
        )

    raise ValueError(f"Unsupported bootstrap method '{method}'")


def compute_logloss_ci(
    logloss_samples: Iterable[float],
    *,
    method: str = "bca",
    replicates: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Compute a confidence interval over mean log-loss.

    Returns (lo, hi) in log-loss space.
    """
    samples = _ensure_array(logloss_samples)

    return _bootstrap_interval(
        samples,
        method=method,
        replicates=replicates,
        alpha=alpha,
        seed=seed,
    )


def _independent_arm_weights(
    weights: Iterable[float] | None,
    size: int,
    *,
    arm: str,
) -> np.ndarray:
    if weights is None:
        return np.ones(size, dtype=float)
    values = np.asarray(list(weights), dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{arm}_weights must be 1-dimensional")
    if values.size != size:
        raise ValueError(f"{arm}_weights length must match {arm} samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{arm}_weights must be finite")
    if np.any(values < 0.0):
        raise ValueError(f"{arm}_weights must be non-negative")
    if float(values.sum()) <= 0.0:
        raise ValueError(f"{arm}_weights must have a positive sum")
    return values


def compute_independent_delta_log_ci(
    final_logloss: Iterable[float],
    preview_logloss: Iterable[float],
    *,
    final_weights: Iterable[float] | None = None,
    preview_weights: Iterable[float] | None = None,
    method: str = "percentile",
    replicates: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap the difference between two independent log-loss slices.

    Preview and final windows are separate sampling units with disjoint IDs.
    Each replicate therefore resamples each arm independently and subtracts
    the two token-weighted arm means.  This function intentionally does not
    offer paired resampling or BCa under a paired-data interpretation.
    """

    if method != "percentile":
        raise ValueError("independent slice delta supports only the percentile method")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    final_values = _ensure_array(final_logloss)
    preview_values = _ensure_array(preview_logloss)
    final_sample_weights = _independent_arm_weights(
        final_weights,
        final_values.size,
        arm="final",
    )
    preview_sample_weights = _independent_arm_weights(
        preview_weights,
        preview_values.size,
        arm="preview",
    )

    rng = np.random.default_rng(seed)
    stats = np.empty(replicates, dtype=float)
    for index in range(replicates):
        final_indices = rng.integers(
            0,
            final_values.size,
            size=final_values.size,
        )
        preview_indices = rng.integers(
            0,
            preview_values.size,
            size=preview_values.size,
        )
        stats[index] = _weighted_mean(
            final_values[final_indices],
            final_sample_weights[final_indices],
        ) - _weighted_mean(
            preview_values[preview_indices],
            preview_sample_weights[preview_indices],
        )
    return _percentile_interval(stats, alpha)


def compute_paired_delta_log_ci(
    final_logloss: Iterable[float],
    baseline_logloss: Iterable[float],
    weights: Iterable[float] | None = None,
    *,
    method: str = "bca",
    replicates: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    strict_lengths: bool = True,
) -> tuple[float, float]:
    """
    Compute a confidence interval over the paired mean delta of log-loss.

    This implementation resamples paired windows as clusters and recomputes the
    token-weighted mean within each replicate when window weights are provided.
    When all weights are equal, it reduces to the ordinary paired bootstrap. See
    docs/assurance/01-eval-math-derivation.md for the derivation.

    Args:
        final_logloss: Iterable of per-window log-loss values after the edit/guard.
        baseline_logloss: Iterable of paired per-window log-loss values (before edit).
        weights: Optional token counts used in the replicate statistic.

    Returns:
        (lo, hi) bounds of Δlog-loss such that ratio CI = exp(bounds).
    """
    final_arr = _ensure_array(final_logloss)
    base_arr = _ensure_array(baseline_logloss)
    if final_arr.size != base_arr.size:
        if strict_lengths:
            raise ValueError("final_logloss and baseline_logloss lengths must match")
        size = min(final_arr.size, base_arr.size)
        final_arr = final_arr[:size]
        base_arr = base_arr[:size]
    weight_arr = None
    if weights is not None:
        if strict_lengths:
            weight_arr = _normalize_weights_strict(weights, final_arr.size)
        else:
            weight_list = list(weights)
            if len(weight_list) >= final_arr.size:
                weight_list = weight_list[: final_arr.size]
            weight_arr = _normalize_weights(weight_list, final_arr.size)
    if final_arr.size == 0:
        return 0.0, 0.0

    delta = final_arr - base_arr
    if not np.all(np.isfinite(delta)):
        raise ValueError("paired log-loss deltas must be finite")
    if _constant_within_subtraction_roundoff(delta, final_arr, base_arr):
        mean_delta = (
            _weighted_mean(delta, weight_arr)
            if weight_arr is not None
            else float(delta.mean())
        )
        return mean_delta, mean_delta

    if weight_arr is not None:
        return _bootstrap_mean_ci_weighted(
            delta,
            weight_arr,
            method=method,
            replicates=replicates,
            alpha=alpha,
            seed=seed,
        )

    return _bootstrap_interval(
        delta,
        method=method,
        replicates=replicates,
        alpha=alpha,
        seed=seed,
    )


def logspace_to_ratio_ci(delta_log_ci: tuple[float, float]) -> tuple[float, float]:
    """Convert Δlog-loss bounds to ratio (perplexity) space."""
    lo, hi = delta_log_ci
    return math.exp(lo), math.exp(hi)


def paired_delta_mean_ci(
    subject: Iterable[float],
    baseline: Iterable[float],
    weights: Iterable[float] | None = None,
    *,
    reps: int = 2000,
    seed: int = 0,
    ci_level: float = 0.95,
    method: str = "bca",
) -> tuple[float, float]:
    """Paired bootstrap CI for the mean delta of paired samples."""
    alpha = 1.0 - float(ci_level)
    if method not in {"bca", "percentile"}:
        raise ValidationError(
            code="E402",
            message="METRICS-VALIDATION-FAILED",
            details={"reason": "method must be 'bca' or 'percentile'"},
        )
    return compute_paired_delta_log_ci(
        list(subject),
        list(baseline),
        weights=list(weights) if weights is not None else None,
        method="bca" if method == "bca" else "percentile",
        replicates=int(reps),
        alpha=alpha,
        seed=int(seed),
    )
