"""Verifier-owned paired BCa replay math.

This implementation intentionally does not call the producer bootstrap helper.
Schemas and recorded method identifiers are shared, while the numerical replay
remains an independent verifier calculation.
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import NormalDist

import numpy as np

_NORMAL = NormalDist()
_BOOTSTRAP_CHUNK_BYTES = 16 * 1024 * 1024


def _array(values: Iterable[float], *, label: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def _normalized_weights(values: Iterable[float], size: int) -> np.ndarray | None:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or result.size != size:
        raise ValueError("weights length must match paired samples")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("weights must have a positive sum")
    if np.allclose(result, result[0]):
        return None
    return result / total


def _mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        return float(np.mean(values))
    total = float(weights.sum())
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.dot(values, weights) / total)


def _resampled_means(
    values: np.ndarray,
    *,
    weights: np.ndarray | None,
    replicates: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Replay bootstrap means in bounded chunks without sharing producer code."""

    size = int(values.size)
    arrays_per_index = 3 if weights is not None else 2
    bytes_per_row = max(size, 1) * 8 * arrays_per_index
    rows = max(1, min(replicates, _BOOTSTRAP_CHUNK_BYTES // bytes_per_row))
    statistics = np.empty(replicates, dtype=float)
    for start in range(0, replicates, rows):
        stop = min(start + rows, replicates)
        selected = generator.integers(0, size, size=(stop - start, size))
        selected_values = values[selected]
        if weights is None:
            statistics[start:stop] = np.mean(selected_values, axis=1)
            continue
        selected_weights = weights[selected]
        denominators = np.sum(selected_weights, axis=1)
        numerators = np.sum(
            selected_values * selected_weights,
            axis=1,
        )
        chunk_statistics = np.mean(selected_values, axis=1)
        np.divide(
            numerators,
            denominators,
            out=chunk_statistics,
            where=denominators > 0.0,
        )
        statistics[start:stop] = chunk_statistics
    return statistics


def _percentile(values: np.ndarray, alpha: float) -> tuple[float, float]:
    return (
        float(np.percentile(values, 100.0 * alpha / 2.0)),
        float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0))),
    )


def _adjusted_interval(
    statistics: np.ndarray,
    *,
    point: float,
    jackknife: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    statistics.sort()
    proportion = np.clip((statistics < point).mean(), 1e-6, 1.0 - 1e-6)
    bias = _NORMAL.inv_cdf(float(proportion))
    jackknife_mean = float(jackknife.mean())
    centered = jackknife_mean - jackknife
    numerator = float(np.sum(centered**3))
    denominator = float(6.0 * (np.sum(centered**2) ** 1.5))
    if denominator == 0.0:
        return _percentile(statistics, alpha)
    acceleration = numerator / denominator

    def adjusted(z_alpha: float) -> float:
        denominator_term = max(
            1.0 - acceleration * (bias + z_alpha),
            1e-12,
        )
        return float(_NORMAL.cdf(bias + (bias + z_alpha) / denominator_term))

    lower = adjusted(_NORMAL.inv_cdf(alpha / 2.0))
    upper = adjusted(_NORMAL.inv_cdf(1.0 - alpha / 2.0))
    return float(np.quantile(statistics, lower)), float(np.quantile(statistics, upper))


def _bca_mean(
    values: np.ndarray,
    *,
    weights: np.ndarray | None,
    replicates: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    size = int(values.size)
    point = _mean(values, weights)
    if size < 2:
        return point, point
    generator = np.random.default_rng(seed)
    statistics = _resampled_means(
        values,
        weights=weights,
        replicates=replicates,
        generator=generator,
    )

    jackknife = np.empty(size, dtype=float)
    if weights is None:
        for index in range(size):
            jackknife[index] = float(np.mean(np.delete(values, index)))
    else:
        sum_weights = float(weights.sum())
        weighted_sum = float(np.dot(values, weights))
        for index in range(size):
            remaining = sum_weights - float(weights[index])
            jackknife[index] = (
                point
                if remaining <= 0.0
                else (weighted_sum - float(weights[index]) * float(values[index]))
                / remaining
            )
    return _adjusted_interval(
        statistics,
        point=point,
        jackknife=jackknife,
        alpha=alpha,
    )


def replay_paired_delta_log_ci(
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
    """Independently replay a paired mean log-loss confidence interval."""

    if method != "bca":
        raise ValueError("verifier replay supports only bca")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    final = _array(final_logloss, label="final_logloss")
    baseline = _array(baseline_logloss, label="baseline_logloss")
    if final.size != baseline.size:
        if strict_lengths:
            raise ValueError("final_logloss and baseline_logloss lengths must match")
        size = min(final.size, baseline.size)
        final = final[:size]
        baseline = baseline[:size]
    normalized = (
        _normalized_weights(weights, int(final.size)) if weights is not None else None
    )
    delta = final - baseline
    if not np.all(np.isfinite(delta)):
        raise ValueError("paired log-loss deltas must be finite")
    spread = float(np.max(delta) - np.min(delta))
    operand_scale = max(
        1.0,
        float(np.max(np.abs(final))),
        float(np.max(np.abs(baseline))),
    )
    if spread <= 8.0 * np.finfo(np.float64).eps * operand_scale:
        point = _mean(delta, normalized)
        return point, point
    return _bca_mean(
        delta,
        weights=normalized,
        replicates=replicates,
        alpha=alpha,
        seed=seed,
    )


__all__ = ["replay_paired_delta_log_ci"]
