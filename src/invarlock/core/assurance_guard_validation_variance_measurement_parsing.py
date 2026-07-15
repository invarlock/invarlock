"""Parse and bind per-window variance A/B measurement evidence."""

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


def _close(left: float | None, right: float | None) -> bool:
    return (
        left is not None
        and right is not None
        and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    )


def _pair_close(left: Any, right: Any) -> bool:
    left_pair = _finite_pair(left)
    right_pair = _finite_pair(right)
    return bool(
        left_pair is not None
        and right_pair is not None
        and _close(left_pair[0], right_pair[0])
        and _close(left_pair[1], right_pair[1])
    )


def _float_list(value: Any, *, positive: bool) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    output: list[float] = []
    for item in value:
        numeric = _finite_number(item)
        if numeric is None or (positive and numeric <= 0.0):
            return None
        output.append(numeric)
    return output


def _token_count_list(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    output: list[int] = []
    for item in value:
        numeric = _nonnegative_int(item)
        if numeric is None or numeric <= 0:
            return None
        output.append(numeric)
    return output


def _arm_values(
    measurements: Mapping[str, Any],
    condition: str,
    coverage: int,
    *,
    source: str,
) -> tuple[list[str], tuple[list[float], list[float], list[int]] | None]:
    errors: list[str] = []
    arm = _mapping(measurements.get(condition))
    if arm is None:
        return [f"{source}.{condition} is required."], None
    ppl = _float_list(arm.get("ppl"), positive=True)
    log_loss = _float_list(arm.get("log_loss"), positive=False)
    token_counts = _token_count_list(arm.get("token_counts"))
    for key, values in (
        ("ppl", ppl),
        ("log_loss", log_loss),
        ("token_counts", token_counts),
    ):
        if values is None or len(values) != coverage:
            errors.append(
                f"{source}.{condition}.{key} must contain exactly one valid value "
                "per calibration window."
            )
    if errors or ppl is None or log_loss is None or token_counts is None:
        return errors, None
    for index, (ppl_value, loss_value) in enumerate(zip(ppl, log_loss, strict=True)):
        try:
            expected_ppl = math.exp(loss_value)
        except OverflowError:
            expected_ppl = float("inf")
        if not _close(ppl_value, expected_ppl):
            errors.append(
                f"{source}.{condition}.ppl[{index}] must equal exp(log_loss[{index}])."
            )
    return errors, (ppl, log_loss, token_counts)


def _window_errors(
    measurements: Mapping[str, Any],
    metrics: Mapping[str, Any],
    coverage: int,
    *,
    source: str,
) -> list[str]:
    window_ids = measurements.get("window_ids")
    if (
        not isinstance(window_ids, list)
        or len(window_ids) != coverage
        or len(set(window_ids)) != coverage
        or any(not isinstance(item, str) or not item for item in window_ids)
    ):
        return [
            f"{source}.window_ids must contain one unique identifier per "
            "calibration window."
        ]
    errors: list[str] = []
    provenance = _mapping(metrics.get("ab_provenance")) or {}
    for condition in ("condition_a", "condition_b"):
        block = _mapping(provenance.get(condition))
        if block is None or block.get("window_ids") != window_ids:
            errors.append(
                f"{source}.window_ids must match ab_provenance.{condition}.window_ids."
            )
    return errors


__all__ = ["_arm_values", "_close", "_pair_close", "_window_errors"]
