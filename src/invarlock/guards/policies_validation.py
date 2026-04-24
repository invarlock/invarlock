"""Validation gate helpers for guard policies."""

import math
from typing import Any

from invarlock.core.exceptions import GuardError, PolicyViolationError

from .policies_presets import _is_non_bool_number
from .tier_config import check_drift as check_tier_drift

# === Validation Gate Presets ===

VALIDATION_GATE_STRICT: dict[str, Any] = {
    "max_capping_rate": 0.3,  # Max 30% of layers can be capped
    "max_ppl_degradation": 0.01,  # Max 1% primary-metric degradation (ppl-like)
    "require_branch_balance": True,
}

VALIDATION_GATE_STANDARD: dict[str, Any] = {
    "max_capping_rate": 0.5,  # Max 50% of layers can be capped
    "max_ppl_degradation": 0.02,  # Max 2% primary-metric degradation (ppl-like)
    "require_branch_balance": True,
}

VALIDATION_GATE_PERMISSIVE: dict[str, Any] = {
    "max_capping_rate": 0.7,  # Max 70% of layers can be capped
    "max_ppl_degradation": 0.05,  # Max 5% primary-metric degradation (ppl-like)
    "require_branch_balance": False,
}

DEFAULT_VALIDATION_GATES: dict[str, dict[str, Any]] = {
    "strict": VALIDATION_GATE_STRICT,
    "standard": VALIDATION_GATE_STANDARD,
    "permissive": VALIDATION_GATE_PERMISSIVE,
}


def get_validation_gate(name: str = "standard") -> dict[str, Any]:
    """
    Get validation gate configuration by name.

    Args:
        name: Gate configuration name

    Returns:
        Validation gate configuration
    """
    if name not in DEFAULT_VALIDATION_GATES:
        available = list(DEFAULT_VALIDATION_GATES.keys())
        raise GuardError(
            code="E502",
            message="POLICY-NOT-FOUND",
            details={"name": name, "available": available},
        )

    return DEFAULT_VALIDATION_GATES[name].copy()


def enforce_validation_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    """Enforce validation gate thresholds.

    Raises PolicyViolationError(E503) with a 'violations' list in details
    when one or more constraints are exceeded.
    """
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
        # Ignore malformed metrics here; gating purely best-effort
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
            # ppl-like ratio: degradation ~ ratio-1; gate on allowed extra
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


def check_policy_drift(*, silent: bool = False) -> dict[str, list[str]]:
    """
    Check for drift between tiers.yaml and hardcoded policy fallbacks.

    This helps detect when tiers.yaml has been updated but hardcoded
    fallbacks in this module haven't been synchronized.

    Args:
        silent: If True, don't emit warnings (just return drift info)

    Returns:
        Dict of tier -> list of drift descriptions.
        Empty dict means no drift detected.

    Example:
        >>> drift = check_policy_drift()
        >>> if drift:
        ...     print("Policy drift detected:", drift)
        ...     print("Consider updating hardcoded defaults in policies.py")
    """
    return check_tier_drift(silent=silent)


__all__ = [
    "VALIDATION_GATE_STRICT",
    "VALIDATION_GATE_STANDARD",
    "VALIDATION_GATE_PERMISSIVE",
    "DEFAULT_VALIDATION_GATES",
    "enforce_validation_gate",
    "get_validation_gate",
    "check_policy_drift",
]
