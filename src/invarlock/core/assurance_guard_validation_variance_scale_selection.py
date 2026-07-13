"""Exact replay of variance scale selection for strict assurance."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .assurance_guard_validation_common import _finite_number


def _replay_producer_scale_filter(
    raw: Mapping[str, Any],
    *,
    max_step: float,
    min_abs: float,
    topk: int,
    deadband: float,
    max_adjusted: int,
) -> dict[str, float] | None:
    """Replay ``compute_variance_scales`` filtering on normalized raw scales."""
    raw_values: dict[str, float] = {}
    for name, raw_value in raw.items():
        value = _finite_number(raw_value)
        if not isinstance(name, str) or not name or value is None or value <= 0.0:
            return None
        raw_values[name] = value

    filtered: dict[str, float] = {}
    raw_deltas: dict[str, float] = {}
    best_candidate: tuple[str, float] | None = None
    best_delta = 0.0
    for name, raw_value in raw_values.items():
        raw_delta = abs(raw_value - 1.0)
        raw_deltas[name] = raw_delta
        if raw_delta > best_delta or (
            raw_delta == best_delta
            and (best_candidate is None or name < best_candidate[0])
        ):
            best_candidate = (name, raw_value)
            best_delta = raw_delta
        if raw_delta < min_abs:
            continue
        selected_value = raw_value
        if max_step > 0.0:
            limited_delta = min(raw_delta, max_step)
            selected_value = 1.0 + math.copysign(limited_delta, raw_value - 1.0)
        filtered[name] = selected_value

    if not filtered and topk > 0 and best_candidate is not None:
        threshold = max(deadband * 0.5, min_abs * 0.5)
        if min_abs > 0.0 and threshold >= min_abs:
            threshold = min_abs * 0.5
        if best_delta >= threshold:
            name, raw_value = best_candidate
            selected_value = raw_value
            if max_step > 0.0:
                limited_delta = min(best_delta, max_step)
                selected_value = 1.0 + math.copysign(limited_delta, raw_value - 1.0)
            filtered[name] = selected_value

    if max_adjusted > 0 and len(filtered) > max_adjusted:
        ranked = sorted(
            filtered.items(),
            key=lambda item: (
                -(
                    raw_deltas.get(item[0], abs(item[1] - 1.0))
                    + (2.0 if item[1] >= 1.0 else 0.0)
                ),
                -raw_deltas.get(item[0], abs(item[1] - 1.0)),
                -item[1],
                item[0],
            ),
        )
        filtered = dict(ranked[:max_adjusted])
    return filtered


def _producer_scale_replay_errors(
    proposed: Mapping[str, Any] | None,
    raw: Mapping[str, Any] | None,
    *,
    label: str,
    max_step: float | None,
    min_abs: float | None,
    topk: int | None,
    deadband: float | None,
    max_adjusted: int | None,
    source: str,
) -> list[str]:
    if (
        proposed is None
        or raw is None
        or max_step is None
        or min_abs is None
        or topk is None
        or deadband is None
        or max_adjusted is None
    ):
        return []
    expected = _replay_producer_scale_filter(
        raw,
        max_step=max_step,
        min_abs=min_abs,
        topk=topk,
        deadband=deadband,
        max_adjusted=max_adjusted,
    )
    if expected is None:
        return []
    if set(proposed) != set(expected):
        return [
            f"{source}.metrics.proposed_scales_{label} keys must exactly replay "
            f"producer filtering of raw_scales_{label}."
        ]
    for name, expected_value in expected.items():
        proposed_value = _finite_number(proposed.get(name))
        if proposed_value is None or proposed_value != expected_value:
            return [
                f"{source}.metrics.proposed_scales_{label}.{name} must exactly "
                f"replay producer filtering of raw_scales_{label}."
            ]
    return []


__all__ = ["_producer_scale_replay_errors"]
