"""Shared primitives for verifier-owned spectral assurance replay."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

_REL_TOL = 1e-9
_ABS_TOL = 1e-12


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _measurement_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-7)


def _numeric_map(
    errors: list[str], value: Any, path: str, *, nonnegative: bool = False
) -> dict[str, float] | None:
    raw = _mapping(value)
    if raw is None or not raw:
        errors.append(f"{path} must be a non-empty object of finite measurements.")
        return None
    result: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{path} keys must be non-empty strings.")
            continue
        number = _finite(value)
        if number is None or (nonnegative and number < 0.0):
            qualifier = " finite non-negative" if nonnegative else " finite"
            errors.append(f"{path}.{key} must be a{qualifier} number.")
            continue
        result[key] = number
    return result


def _family_map(errors: list[str], value: Any, path: str) -> dict[str, str] | None:
    raw = _mapping(value)
    if raw is None or not raw:
        errors.append(f"{path} must be a non-empty object.")
        return None
    result: dict[str, str] = {}
    for module, family in raw.items():
        if not isinstance(module, str) or not module:
            errors.append(f"{path} keys must be non-empty strings.")
        elif not isinstance(family, str) or not family.strip():
            errors.append(f"{path}.{module} must name a non-empty family.")
        else:
            result[module] = family.strip()
    return result


def _family_stats(
    baseline: Mapping[str, float], families: Mapping[str, str]
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for module, sigma in baseline.items():
        buckets[families[module]].append(sigma)
    result: dict[str, dict[str, float | int]] = {}
    for family, values in buckets.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result[family] = {
            "count": len(values),
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
        }
    return result


def _compare_tree(errors: list[str], observed: Any, expected: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            errors.append(f"{path} must be an object.")
            return
        if set(observed) != set(expected):
            errors.append(f"{path} keys disagree with independently replayed evidence.")
            return
        for key in sorted(expected):
            _compare_tree(errors, observed[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or observed != expected:
            errors.append(f"{path} disagrees with independently replayed evidence.")
        return
    if isinstance(expected, float):
        number = _finite(observed)
        if number is None or not _close(number, expected):
            errors.append(f"{path} disagrees with independently replayed evidence.")
        return
    if observed != expected:
        errors.append(f"{path} disagrees with independently replayed evidence.")


def _policy_number(
    errors: list[str],
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    minimum: float,
) -> float | None:
    value = _finite(mapping.get(key))
    if value is None or value < minimum:
        errors.append(f"{path}.{key} must be a finite number >= {minimum}.")
        return None
    return value


def _degeneracy_map(
    errors: list[str], value: Any, path: str, modules: set[str]
) -> dict[str, dict[str, float]] | None:
    raw = _mapping(value)
    if raw is None or set(raw) != modules:
        errors.append(f"{path} must cover exactly the measured module inventory.")
        return None
    result: dict[str, dict[str, float]] = {}
    for module in sorted(modules):
        values = _mapping(raw.get(module))
        if values is None:
            errors.append(f"{path}.{module} must be an object.")
            continue
        item: dict[str, float] = {}
        for metric in ("stable_rank", "norm_collapse"):
            number = _finite(values.get(metric))
            if number is None or number < 0.0:
                errors.append(
                    f"{path}.{module}.{metric} must be a finite non-negative number."
                )
            else:
                item[metric] = number
        if len(item) == 2:
            result[module] = item
    return result


def _reject_families(
    family_pvalues: Mapping[str, float], *, method: str, alpha: float, m: int
) -> set[str]:
    if method == "bonferroni":
        cutoff = alpha / m
        return {family for family, pvalue in family_pvalues.items() if pvalue <= cutoff}
    ordered = sorted(family_pvalues.items(), key=lambda item: item[1])
    max_rank = 0
    for rank, (_, pvalue) in enumerate(ordered, start=1):
        if pvalue <= (alpha * rank) / m:
            max_rank = rank
    if max_rank == 0:
        return set()
    cutoff = (alpha * max_rank) / m
    return {family for family, pvalue in ordered if pvalue <= cutoff}


def _violation_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(value.get("type") or ""),
        str(value.get("module") or ""),
        str(value.get("family") or ""),
    )


def _compare_violations(
    errors: list[str], observed: Any, expected: list[dict[str, Any]], path: str
) -> None:
    if not isinstance(observed, list):
        errors.append(f"{path} must be an array.")
        return
    observed_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for raw in observed:
        if not isinstance(raw, Mapping):
            errors.append(f"{path} entries must be objects.")
            continue
        key = _violation_key(raw)
        if key in observed_by_key:
            errors.append(f"{path} contains duplicate violation {key!r}.")
        observed_by_key[key] = raw
    expected_by_key = {_violation_key(item): item for item in expected}
    if set(observed_by_key) != set(expected_by_key):
        missing = sorted(set(expected_by_key) - set(observed_by_key))
        unexpected = sorted(set(observed_by_key) - set(expected_by_key))
        errors.append(
            f"{path} disagrees with replayed spectral violations "
            f"(missing={missing!r}, unexpected={unexpected!r})."
        )
        return
    for key, expected_item in expected_by_key.items():
        observed_item = observed_by_key[key]
        for field, expected_value in expected_item.items():
            if field == "message":
                continue
            _compare_tree(
                errors,
                observed_item.get(field),
                expected_value,
                f"{path}[{key!r}].{field}",
            )


__all__ = [
    "_close",
    "_compare_tree",
    "_compare_violations",
    "_degeneracy_map",
    "_family_map",
    "_family_stats",
    "_finite",
    "_mapping",
    "_measurement_close",
    "_nonnegative_int",
    "_numeric_map",
    "_policy_number",
    "_reject_families",
]
