"""
InvarLock Core Bootstrap Utilities
==============================

Numerically stable bootstrap helpers for evaluation metrics.

This module provides bias-corrected and accelerated (BCa) confidence
intervals tailored for paired log-loss statistics used by the runner
and safety certificate reports.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from statistics import NormalDist

import numpy as np

__all__ = [
    "compute_logloss_ci",
    "compute_paired_delta_log_ci",
    "logspace_to_ratio_ci",
]


Normal = NormalDist()


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


def _weighted_mean(samples: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return float(np.mean(samples))
    return float(np.dot(samples, weights) / total)


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
    """Compute a BCa interval for the mean under weighted resampling."""
    n = samples.size
    if n < 2:
        stat = _weighted_mean(samples, weights)
        return float(stat), float(stat)

    prob = weights / float(weights.sum())
    stats = np.empty(replicates, dtype=float)
    for i in range(replicates):
        idx = rng.choice(n, size=n, replace=True, p=prob)
        stats[i] = float(np.mean(samples[idx]))

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
    stat_fn: Callable[[np.ndarray], float],
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
        stat = stat_fn(samples)
        return float(stat), float(stat)

    # Bootstrap replicates of the statistic
    stats = np.empty(replicates, dtype=float)
    for i in range(replicates):
        idx = rng.integers(0, n, size=n)
        stats[i] = stat_fn(samples[idx])

    stats.sort()
    stat_hat = stat_fn(samples)

    # Bias-correction
    prop = np.clip((stats < stat_hat).mean(), 1e-6, 1.0 - 1e-6)
    z0 = Normal.inv_cdf(prop)

    # Jackknife estimates for acceleration
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack_sample = np.delete(samples, i)
        jack[i] = stat_fn(jack_sample)

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
        stats = np.empty(replicates, dtype=float)
        n = samples.size
        prob = weights / float(weights.sum())
        for i in range(replicates):
            idx = rng.choice(n, size=n, replace=True, p=prob)
            stats[i] = float(np.mean(samples[idx]))
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
    stat_fn: Callable[[np.ndarray], float],
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
        stats = np.empty(replicates, dtype=float)
        n = samples.size
        for i in range(replicates):
            idx = rng.integers(0, n, size=n)
            stats[i] = stat_fn(samples[idx])
        stats.sort()
        return _percentile_interval(stats, alpha)
    if method == "bca":
        return _bca_interval(
            samples,
            stat_fn=stat_fn,
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

    def stat_fn(data: np.ndarray) -> float:
        return float(np.mean(data))

    return _bootstrap_interval(
        samples,
        stat_fn=stat_fn,
        method=method,
        replicates=replicates,
        alpha=alpha,
        seed=seed,
    )


def compute_paired_delta_log_ci(
    final_logloss: Iterable[float],
    baseline_logloss: Iterable[float],
    weights: Iterable[float] | None = None,
    *,
    method: str = "bca",
    replicates: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Compute a confidence interval over the paired mean delta of log-loss.

    This implementation uses token-weighted resampling when window weights are
    provided. When all weights are equal, the weighted bootstrap reduces to the
    simple mean. See docs/assurance/01-eval-math-proof.md for the derivation.

    Args:
        final_logloss: Iterable of per-window log-loss values after the edit/guard.
        baseline_logloss: Iterable of paired per-window log-loss values (before edit).
        weights: Optional token counts per window; used for weighted resampling.

    Returns:
        (lo, hi) bounds of Δlog-loss such that ratio CI = exp(bounds).
    """
    final_arr = _ensure_array(final_logloss)
    base_arr = _ensure_array(baseline_logloss)
    if final_arr.size != base_arr.size:
        size = min(final_arr.size, base_arr.size)
        final_arr = final_arr[:size]
        base_arr = base_arr[:size]
    weight_arr = None
    if weights is not None:
        weight_list = list(weights)
        if len(weight_list) >= final_arr.size:
            weight_list = weight_list[: final_arr.size]
        weight_arr = _normalize_weights(weight_list, final_arr.size)
    if final_arr.size == 0:
        return 0.0, 0.0

    delta = final_arr - base_arr
    if np.allclose(delta, delta[0]):
        mean_delta = float(delta.mean())
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

    def stat_fn(data: np.ndarray) -> float:
        return float(np.mean(data))

    return _bootstrap_interval(
        delta,
        stat_fn=stat_fn,
        method=method,
        replicates=replicates,
        alpha=alpha,
        seed=seed,
    )


def logspace_to_ratio_ci(delta_log_ci: tuple[float, float]) -> tuple[float, float]:
    """Convert Δlog-loss bounds to ratio (perplexity) space."""
    lo, hi = delta_log_ci
    return math.exp(lo), math.exp(hi)
