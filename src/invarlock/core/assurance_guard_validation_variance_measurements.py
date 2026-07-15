"""Replay strict variance A/B decisions from retained per-window measurements."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from invarlock.core.bootstrap import compute_paired_delta_log_ci
from invarlock.guards.variance_policy import predictive_gate_outcome
from invarlock.utils import (
    bootstrap_mean_statistics,
    percentile_interval_from_statistics,
)

from .assurance_guard_validation_common import (
    _finite_number,
    _mapping,
    _nonnegative_int,
)
from .assurance_guard_validation_variance_measurement_parsing import (
    _arm_values,
    _close,
    _pair_close,
    _window_errors,
)

_BOOTSTRAP_REPLICATES = 500
_DELTA_SEED_OFFSET = 211
_RATIO_METHOD = "percentile_mean_ppl_ratio"
_DELTA_METHOD = "bca_paired_delta_log"


def _metadata_errors(
    measurements: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
    seed: int | None,
    *,
    source: str,
) -> tuple[list[str], float | None]:
    errors: list[str] = []
    alpha = _finite_number(policy.get("alpha")) if policy is not None else None
    if alpha is None or not 0.0 < alpha < 1.0:
        errors.append(f"{source} policy alpha must be finite and between zero and one.")
        alpha = None
    ratio = _mapping(measurements.get("ratio_bootstrap"))
    delta = _mapping(measurements.get("delta_log_bootstrap"))
    expected_ratio = {
        "method": _RATIO_METHOD,
        "replicates": _BOOTSTRAP_REPLICATES,
        "alpha": alpha,
        "seed": seed,
    }
    expected_delta = {
        "method": _DELTA_METHOD,
        "replicates": _BOOTSTRAP_REPLICATES,
        "alpha": alpha,
        "seed": seed + _DELTA_SEED_OFFSET if seed is not None else None,
        "weights": "condition_a_token_counts",
    }
    if ratio is None or ratio != expected_ratio:
        errors.append(f"{source}.ratio_bootstrap must match the producer algorithm.")
    if delta is None or delta != expected_delta:
        errors.append(
            f"{source}.delta_log_bootstrap must match the producer algorithm."
        )
    return errors, alpha


def _replay_intervals(
    condition_a: tuple[list[float], list[float], list[int]],
    condition_b: tuple[list[float], list[float], list[int]],
    *,
    alpha: float,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    ppl_a, loss_a, token_counts = condition_a
    ppl_b, loss_b, _ = condition_b
    ratios = np.asarray(
        [right / left for left, right in zip(ppl_a, ppl_b, strict=True)],
        dtype=float,
    )
    ratio_stats = bootstrap_mean_statistics(
        ratios,
        n_bootstrap=_BOOTSTRAP_REPLICATES,
        random_state=np.random.default_rng(seed),
    )
    ratio_ci = percentile_interval_from_statistics(ratio_stats, alpha=alpha)
    delta_ci = compute_paired_delta_log_ci(
        loss_b,
        loss_a,
        weights=token_counts,
        method="bca",
        replicates=_BOOTSTRAP_REPLICATES,
        alpha=alpha,
        seed=seed + _DELTA_SEED_OFFSET,
    )
    return ratio_ci, delta_ci


def _aggregate_errors(
    metrics: Mapping[str, Any],
    measurements: Mapping[str, Any],
    condition_a: tuple[list[float], list[float], list[int]],
    condition_b: tuple[list[float], list[float], list[int]],
    ratio_ci: tuple[float, float],
    delta_ci: tuple[float, float],
    policy: Mapping[str, Any],
    *,
    source: str,
    no_adjustment: bool,
) -> list[str]:
    errors: list[str] = []
    ppl_a, loss_a, token_counts = condition_a
    ppl_b, loss_b, token_counts_b = condition_b
    if token_counts_b != token_counts:
        errors.append(f"{source} A/B token counts must match for paired windows.")
    if no_adjustment and condition_a != condition_b:
        errors.append(f"{source} virtual no-adjustment condition B must equal A.")

    ppl_a_mean = float(np.mean(np.asarray(ppl_a, dtype=float)))
    ppl_b_mean = float(np.mean(np.asarray(ppl_b, dtype=float)))
    for key, expected in (("ppl_no_ve", ppl_a_mean), ("ppl_with_ve", ppl_b_mean)):
        if not _close(_finite_number(metrics.get(key)), expected):
            errors.append(f"{source} must reproduce {key} from per-window PPL.")
    measured_gain = (ppl_a_mean - ppl_b_mean) / ppl_a_mean
    if not _close(_finite_number(metrics.get("ab_gain")), measured_gain):
        errors.append(f"{source} must reproduce ab_gain from per-window PPL.")
    if not _pair_close(measurements.get("ratio_ci"), ratio_ci):
        errors.append(f"{source}.ratio_ci must match deterministic bootstrap replay.")
    if not _pair_close(metrics.get("ratio_ci"), ratio_ci):
        errors.append(f"{source} must reproduce metrics.ratio_ci from measurements.")
    if not _pair_close(measurements.get("delta_log_ci"), delta_ci):
        errors.append(
            f"{source}.delta_log_ci must match deterministic bootstrap replay."
        )

    if no_adjustment:
        return errors
    predictive = _mapping(metrics.get("predictive_gate")) or {}
    weighted_delta = float(
        np.average(
            np.asarray(loss_b, dtype=float) - np.asarray(loss_a, dtype=float),
            weights=np.asarray(token_counts, dtype=float),
        )
    )
    if not _close(_finite_number(predictive.get("mean_delta")), weighted_delta):
        errors.append(f"{source} must reproduce predictive_gate.mean_delta.")
    if not _pair_close(predictive.get("delta_ci"), delta_ci):
        errors.append(f"{source} must reproduce predictive_gate.delta_ci.")
    expected_gain_ci = (-delta_ci[1], -delta_ci[0])
    if not _pair_close(predictive.get("gain_ci"), expected_gain_ci):
        errors.append(f"{source} must reproduce predictive_gate.gain_ci.")
    min_effect = _finite_number(policy.get("min_effect_lognll"))
    one_sided = policy.get("predictive_one_sided")
    if min_effect is not None and isinstance(one_sided, bool):
        passed, reason = predictive_gate_outcome(
            mean_delta=weighted_delta,
            delta_ci=delta_ci,
            min_effect=min_effect,
            one_sided=one_sided,
        )
        if predictive.get("passed") is not passed or predictive.get("reason") != reason:
            errors.append(f"{source} predictive decision must match replayed evidence.")
    return errors


def _variance_measurement_errors(
    variance: Mapping[str, Any],
    entry: Mapping[str, Any],
    metrics: Mapping[str, Any],
    coverage: int | None,
    policy: Mapping[str, Any] | None,
    *,
    source: str,
    no_adjustment: bool,
) -> list[str]:
    """Validate mirrors and replay all strict variance A/B aggregate claims."""
    if coverage is None or coverage <= 0:
        return [f"{source}.metrics.calibration.coverage must be positive."]
    raw = _mapping(metrics.get("ab_measurements"))
    ab_test = _mapping(variance.get("ab_test"))
    top = _mapping(ab_test.get("measurements")) if ab_test is not None else None
    details = _mapping(entry.get("details")) or {}
    stats = _mapping(details.get("stats")) or {}
    if raw is None:
        return [f"{source}.metrics.ab_measurements is required."]
    errors: list[str] = []
    if top is None or top != raw:
        errors.append(
            f"variance.ab_test.measurements must match {source}.metrics.ab_measurements."
        )
    if stats.get("ab_measurements") != raw:
        errors.append(
            f"{source}.details.stats.ab_measurements must match raw measurements."
        )
    errors.extend(
        _window_errors(
            raw, metrics, coverage, source=f"{source}.metrics.ab_measurements"
        )
    )
    seed = _nonnegative_int(metrics.get("ab_seed_used"))
    metadata_errors, alpha = _metadata_errors(
        raw,
        policy,
        seed,
        source=f"{source}.metrics.ab_measurements",
    )
    errors.extend(metadata_errors)
    arm_a_errors, arm_a = _arm_values(
        raw, "condition_a", coverage, source=f"{source}.metrics.ab_measurements"
    )
    arm_b_errors, arm_b = _arm_values(
        raw, "condition_b", coverage, source=f"{source}.metrics.ab_measurements"
    )
    errors.extend(arm_a_errors)
    errors.extend(arm_b_errors)
    if arm_a is None or arm_b is None or alpha is None or seed is None:
        return errors
    try:
        ratio_ci, delta_ci = _replay_intervals(
            arm_a,
            arm_b,
            alpha=alpha,
            seed=seed,
        )
    except (ArithmeticError, TypeError, ValueError) as exc:
        errors.append(f"{source}.metrics.ab_measurements cannot be replayed: {exc}")
        return errors
    if policy is None:
        errors.append(f"{source}.policy is required for measurement replay.")
        return errors
    errors.extend(
        _aggregate_errors(
            metrics,
            raw,
            arm_a,
            arm_b,
            ratio_ci,
            delta_ci,
            policy,
            source=f"{source}.metrics.ab_measurements",
            no_adjustment=no_adjustment,
        )
    )
    return errors


__all__ = ["_variance_measurement_errors"]
