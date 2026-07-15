"""Standalone whole-model invariant checks."""

from __future__ import annotations

from typing import Any

from invarlock.core.types import GuardOutcome

_INVARIANT_CAPTURE_ERRORS = (
    AttributeError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def check_all_invariants(model: Any, threshold: float = 1e-6) -> GuardOutcome:
    """
    Check all basic model invariants.

    Args:
        model: PyTorch model to check
        threshold: Numerical threshold for invariant checks

    Returns:
        GuardOutcome: Result of invariant checking
    """
    violations = []

    # Basic model structure checks
    if not hasattr(model, "named_parameters"):
        violations.append(
            {
                "type": "structure_violation",
                "message": "Model missing named_parameters method",
            }
        )
        return GuardOutcome(
            name="check_all_invariants",
            passed=False,
            decision="block",
            violations=violations,
            metrics={},
        )

    try:
        named_parameters = list(model.named_parameters())
    except _INVARIANT_CAPTURE_ERRORS as exc:
        violations.append(
            {
                "type": "structure_violation",
                "message": f"Could not iterate named_parameters: {exc}",
            }
        )
        return GuardOutcome(
            name="check_all_invariants",
            passed=False,
            decision="block",
            violations=violations,
            metrics={"parameters_checked": 0, "violations_found": len(violations)},
        )

    # Check for NaN/Inf in parameters
    for name, param in named_parameters:
        if hasattr(param.data, "isnan") and param.data.isnan().any():
            violations.append(
                {
                    "type": "nan_violation",
                    "parameter": name,
                    "message": f"NaN detected in parameter {name}",
                }
            )
        if hasattr(param.data, "isinf") and param.data.isinf().any():
            violations.append(
                {
                    "type": "inf_violation",
                    "parameter": name,
                    "message": f"Inf detected in parameter {name}",
                }
            )

    # Check parameter ranges are reasonable
    for name, param in named_parameters:
        if hasattr(param.data, "abs") and hasattr(param.data, "max"):
            max_val = param.data.abs().max()
            if hasattr(max_val, "item"):
                max_val = max_val.item()

            if max_val > 1000:
                violations.append(
                    {
                        "type": "range_violation",
                        "parameter": name,
                        "max_value": max_val,
                        "message": f"Parameter {name} has unusually large values (max: {max_val})",
                    }
                )
            if max_val < threshold:
                violations.append(
                    {
                        "type": "range_violation",
                        "parameter": name,
                        "max_value": max_val,
                        "message": f"Parameter {name} has unusually small values (max: {max_val})",
                    }
                )

    passed = len(violations) == 0
    decision = "allow" if passed else "block"

    return GuardOutcome(
        name="check_all_invariants",
        passed=passed,
        decision=decision,
        violations=violations,
        metrics={
            "parameters_checked": len(named_parameters),
            "violations_found": len(violations),
        },
    )


def assert_invariants(model: Any, threshold: float = 1e-6) -> None:
    """
    Assert that all model invariants hold, raising exception if not.

    Args:
        model: PyTorch model to check
        threshold: Numerical threshold for invariant checks

    Raises:
        AssertionError: If any invariants are violated
    """
    result = check_all_invariants(model, threshold)
    if not result.passed:
        violation_messages = [v.get("message", str(v)) for v in result.violations or []]
        raise AssertionError(
            f"Model invariants violated: {'; '.join(violation_messages)}"
        )
