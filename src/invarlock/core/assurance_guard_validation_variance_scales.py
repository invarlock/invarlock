"""Strict semantic validation for variance mitigation scale evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .assurance_guard_validation_common import (
    _finite_number,
    _finite_pair,
    _mapping,
    _nonnegative_int,
)
from .assurance_guard_validation_variance_scale_selection import (
    _producer_scale_replay_errors,
)


def _scale_policy(
    policy: Mapping[str, Any] | None,
    *,
    source: str,
) -> tuple[
    list[str],
    tuple[float, float] | None,
    float | None,
    float | None,
    int | None,
    float | None,
    int | None,
]:
    if policy is None:
        return (
            [f"{source}.policy is required for scale validation."],
            None,
            None,
            None,
            None,
            None,
            None,
        )
    errors: list[str] = []
    clamp = _finite_pair(policy.get("clamp"))
    if clamp is None or clamp[0] <= 0.0 or clamp[0] > 1.0 or clamp[1] < 1.0:
        errors.append(
            f"{source}.policy.clamp must be positive, ordered, and contain 1.0."
        )
        clamp = None
    max_step = _finite_number(policy.get("max_scale_step"))
    if max_step is None or max_step < 0.0:
        errors.append(f"{source}.policy.max_scale_step must be non-negative.")
        max_step = None
    min_abs = _finite_number(policy.get("min_abs_adjust"))
    if min_abs is None or min_abs < 0.0:
        errors.append(f"{source}.policy.min_abs_adjust must be non-negative.")
        min_abs = None
    topk = _nonnegative_int(policy.get("topk_backstop"))
    if topk is None:
        errors.append(f"{source}.policy.topk_backstop must be a non-negative integer.")
    deadband = _finite_number(policy.get("deadband"))
    if deadband is None or deadband < 0.0:
        errors.append(f"{source}.policy.deadband must be non-negative.")
        deadband = None
    max_adjusted = _nonnegative_int(policy.get("max_adjusted_modules"))
    if max_adjusted is None:
        errors.append(
            f"{source}.policy.max_adjusted_modules must be a non-negative integer."
        )
    return errors, clamp, max_step, min_abs, topk, deadband, max_adjusted


def _scale_map_errors(
    values: Mapping[str, Any] | None,
    *,
    key: str,
    target_names: set[str],
    clamp: tuple[float, float] | None,
    source: str,
) -> list[str]:
    if values is None or not values:
        return [f"{source}.metrics.{key} must be a non-empty object."]
    errors: list[str] = []
    for name, raw_value in values.items():
        value = _finite_number(raw_value)
        if (
            not isinstance(name, str)
            or not name
            or (target_names and name not in target_names)
            or value is None
            or value <= 0.0
        ):
            errors.append(
                f"{source}.metrics.{key} must map declared targets to positive "
                "finite scales."
            )
            continue
        if clamp is not None and not clamp[0] <= value <= clamp[1]:
            errors.append(f"{source}.metrics.{key}.{name} is outside policy.clamp.")
    return errors


def _proposed_relation_errors(
    proposed: Mapping[str, Any] | None,
    raw: Mapping[str, Any] | None,
    *,
    label: str,
    max_step: float | None,
    min_abs: float | None,
    topk: int | None,
    deadband: float | None,
    source: str,
) -> list[str]:
    if proposed is None or raw is None:
        return []
    errors: list[str] = []
    if not set(proposed).issubset(raw):
        errors.append(
            f"{source}.metrics.proposed_scales_{label} keys must be present in "
            f"raw_scales_{label}."
        )
    for name, raw_proposed in proposed.items():
        proposed_value = _finite_number(raw_proposed)
        raw_value = _finite_number(raw.get(name))
        if proposed_value is None or raw_value is None:
            continue
        proposed_delta = proposed_value - 1.0
        raw_delta = raw_value - 1.0
        if math.isclose(proposed_delta, 0.0, abs_tol=1e-12):
            errors.append(
                f"{source}.metrics.proposed_scales_{label}.{name} must be non-identity."
            )
            continue
        if proposed_delta * raw_delta <= 0.0:
            errors.append(
                f"{source}.metrics.proposed_scales_{label}.{name} must preserve "
                "the raw scale direction."
            )
        if abs(proposed_delta) > abs(raw_delta) + 1e-12:
            errors.append(
                f"{source}.metrics.proposed_scales_{label}.{name} cannot exceed "
                "the raw scale adjustment."
            )
        expected_delta = abs(raw_delta)
        if max_step is not None and max_step > 0.0:
            expected_delta = min(expected_delta, max_step)
        expected_value = 1.0 + math.copysign(expected_delta, raw_delta)
        if not math.isclose(
            proposed_value, expected_value, rel_tol=1e-9, abs_tol=1e-12
        ):
            errors.append(
                f"{source}.metrics.proposed_scales_{label}.{name} must be derived "
                "from the raw scale and max_scale_step."
            )
        if min_abs is not None and abs(raw_delta) < min_abs:
            threshold = max((deadband or 0.0) * 0.5, min_abs * 0.5)
            if min_abs > 0.0 and threshold >= min_abs:
                threshold = min_abs * 0.5
            if not topk or abs(raw_delta) < threshold:
                errors.append(
                    f"{source}.metrics.proposed_scales_{label}.{name} does not "
                    "meet min_abs_adjust or the configured backstop threshold."
                )
    return errors


def _variance_gain_scale_errors(
    entry: Mapping[str, Any],
    metrics: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
    *,
    source: str,
) -> list[str]:
    errors, clamp, max_step, min_abs, topk, deadband, max_adjusted = _scale_policy(
        policy, source=source
    )
    proposed_count = _nonnegative_int(metrics.get("proposed_scales"))
    target_count = _nonnegative_int(metrics.get("target_modules"))
    target_names = metrics.get("target_module_names")
    if proposed_count is None or proposed_count <= 0:
        errors.append(f"{source}.metrics.proposed_scales must be positive.")
    if target_count is None or target_count <= 0:
        errors.append(f"{source}.metrics.target_modules must be positive.")
    if (
        not isinstance(target_names, list)
        or not target_names
        or any(not isinstance(name, str) or not name for name in target_names)
        or len(set(target_names)) != len(target_names)
        or (target_count is not None and len(target_names) != target_count)
    ):
        errors.append(
            f"{source}.metrics.target_module_names must enumerate all targets."
        )
    target_name_set = (
        set(target_names)
        if isinstance(target_names, list)
        and all(isinstance(name, str) for name in target_names)
        else set()
    )
    scale_maps: dict[str, Mapping[str, Any] | None] = {}
    for key in (
        "proposed_scales_pre_edit",
        "proposed_scales_post_edit",
        "raw_scales_pre_edit",
        "raw_scales_post_edit",
    ):
        values = _mapping(metrics.get(key))
        scale_maps[key] = values
        errors.extend(
            _scale_map_errors(
                values,
                key=key,
                target_names=target_name_set,
                clamp=clamp,
                source=source,
            )
        )

    for label in ("pre_edit", "post_edit"):
        errors.extend(
            _proposed_relation_errors(
                scale_maps[f"proposed_scales_{label}"],
                scale_maps[f"raw_scales_{label}"],
                label=label,
                max_step=max_step,
                min_abs=min_abs,
                topk=topk,
                deadband=deadband,
                source=source,
            )
        )
        errors.extend(
            _producer_scale_replay_errors(
                scale_maps[f"proposed_scales_{label}"],
                scale_maps[f"raw_scales_{label}"],
                label=label,
                max_step=max_step,
                min_abs=min_abs,
                topk=topk,
                deadband=deadband,
                max_adjusted=max_adjusted,
                source=source,
            )
        )

    post_scales = scale_maps["proposed_scales_post_edit"]
    if (
        post_scales is not None
        and proposed_count is not None
        and len(post_scales) != proposed_count
    ):
        errors.append(
            f"{source}.metrics.proposed_scales must match the post-edit scale map."
        )
    if max_adjusted and proposed_count is not None and proposed_count > max_adjusted:
        errors.append(
            f"{source}.metrics.proposed_scales exceeds policy.max_adjusted_modules."
        )

    details = _mapping(entry.get("details"))
    stats = _mapping(details.get("stats")) if details is not None else None
    detail_scales = _mapping(details.get("proposed_scales")) if details else None
    if detail_scales is None or detail_scales != post_scales:
        errors.append(
            f"{source}.details.proposed_scales must match "
            "metrics.proposed_scales_post_edit exactly."
        )
    if stats is None:
        return errors
    for metric_key, stats_key in (
        ("target_module_names", "target_module_names"),
        ("proposed_scales_pre_edit", "proposed_scales_pre_edit"),
        ("proposed_scales_post_edit", "proposed_scales_post_edit"),
        ("raw_scales_pre_edit", "raw_scales_pre_edit_normalized"),
        ("raw_scales_post_edit", "raw_scales_post_edit_normalized"),
    ):
        if metrics.get(metric_key) != stats.get(stats_key):
            errors.append(
                f"{source}.details.stats.{stats_key} must match "
                f"{source}.metrics.{metric_key}."
            )
    return errors


__all__ = ["_variance_gain_scale_errors"]
