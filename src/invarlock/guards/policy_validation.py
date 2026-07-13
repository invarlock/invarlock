"""Validation-gate resolution and enforcement for guard policies."""

from __future__ import annotations

import math
from typing import Any

from invarlock.core.exceptions import GuardError, PolicyViolationError


def _is_non_bool_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def get_validation_gate(
    name: str,
    *,
    gates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve a named validation gate from the supplied policy catalog."""
    if name not in gates:
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": list(gates)},
        )
    return gates[name].copy()


def enforce_validation_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    """Enforce validation gate thresholds."""
    violations: list[dict[str, Any]] = []

    try:
        caps_value = metrics.get("caps_applied", 0)
        total_value = metrics.get("total_layers", 0)
        if not (_is_non_bool_number(caps_value) and _is_non_bool_number(total_value)):
            raise TypeError("caps_applied and total_layers must be numeric")
        caps = float(caps_value)
        total = float(total_value)
        if total > 0:
            rate = caps / total
            limit = float(gate.get("max_capping_rate", 1.0))
            if rate > limit:
                violations.append(
                    {
                        "type": "capping_rate",
                        "actual": rate,
                        "limit": limit,
                    }
                )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    try:
        ratio = metrics.get("primary_metric_ratio")
        if (
            isinstance(ratio, int | float)
            and not isinstance(ratio, bool)
            and math.isfinite(float(ratio))
        ):
            ratio_f = float(ratio)
            limit = float(gate.get("max_ppl_degradation", 1.0))
            degradation = ratio_f - 1.0
            if degradation > limit:
                violations.append(
                    {
                        "type": "primary_metric_degradation",
                        "actual": degradation,
                        "limit": limit,
                    }
                )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    if isinstance(gate.get("require_branch_balance"), bool) and gate.get(
        "require_branch_balance"
    ):
        if metrics.get("branch_balance_ok") is False:
            violations.append(
                {"type": "branch_balance", "actual": False, "limit": True}
            )

    if violations:
        raise PolicyViolationError(
            code="E503",
            message="VALIDATION-GATE-FAILED",
            details={"violations": violations, "metrics": metrics, "gate": gate},
        )
