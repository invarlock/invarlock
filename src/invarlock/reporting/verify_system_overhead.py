"""Strict consistency checks for reported system-overhead measurements."""

from __future__ import annotations

import math
from typing import Any


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate_system_overhead(report: dict[str, Any]) -> list[str]:
    """Verify each overhead delta and ratio against its source measurements."""

    overhead = report.get("system_overhead")
    if overhead is None:
        return []
    if not isinstance(overhead, dict):
        return ["system_overhead must be an object."]
    errors: list[str] = []
    for name, raw_entry in overhead.items():
        if not isinstance(raw_entry, dict):
            errors.append(f"system_overhead.{name} must be a structured entry.")
            continue
        edited = _coerce_finite_float(raw_entry.get("edited"))
        baseline = _coerce_finite_float(raw_entry.get("baseline"))
        delta = _coerce_finite_float(raw_entry.get("delta"))
        ratio = _coerce_finite_float(raw_entry.get("ratio"))
        if edited is None:
            errors.append(f"system_overhead.{name}.edited must be finite.")
            continue
        if baseline is None:
            if "delta" in raw_entry or "ratio" in raw_entry:
                errors.append(
                    f"system_overhead.{name} cannot declare delta or ratio without baseline."
                )
            continue
        expected_delta = edited - baseline
        if delta is None or not math.isclose(
            delta, expected_delta, rel_tol=1e-9, abs_tol=1e-9
        ):
            errors.append(
                f"system_overhead.{name}.delta does not match edited-baseline."
            )
        if baseline == 0.0:
            if "ratio" in raw_entry:
                errors.append(
                    f"system_overhead.{name}.ratio is undefined for a zero baseline."
                )
        else:
            expected_ratio = edited / baseline
            if ratio is None or not math.isclose(
                ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-9
            ):
                errors.append(
                    f"system_overhead.{name}.ratio does not match edited/baseline."
                )
    return errors


__all__ = ["validate_system_overhead"]
