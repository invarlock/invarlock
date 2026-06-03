from __future__ import annotations

from typing import Any

_INVARIANT_CAPTURE_ERRORS = (
    AttributeError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def check_adapter_aware_invariants(
    model: Any, verbose: bool = False
) -> tuple[bool, dict[str, Any]]:
    """
    Check model invariants with adapter awareness.

    Args:
        model: Model to check
        verbose: Whether to print detailed information

    Returns:
        (all_passed, results) tuple
    """
    results: dict[str, Any] = {"adapter_type": "none", "checks": {}, "violations": []}
    all_passed = True
    standard_checks: dict[str, dict[str, Any]] = check_standard_invariants(model)
    results["checks"].update(standard_checks)
    for check_name, check_result in standard_checks.items():
        if not check_result.get("passed", True):
            all_passed = False
            results["violations"].append(
                {
                    "type": "standard_violation",
                    "check": check_name,
                    "message": check_result.get("message", "Check failed"),
                }
            )
    return all_passed, results


def detect_adapter_type(model: Any) -> str:
    """Detect adapter type (disabled). Always returns 'none'."""
    return "none"


def check_standard_invariants(model: Any) -> dict[str, dict[str, Any]]:
    """Check standard model invariants."""
    checks: dict[str, dict[str, Any]] = {}

    try:
        param_count = sum(p.numel() for p in model.parameters())
        checks["parameter_count"] = {
            "passed": param_count > 0,
            "count": param_count,
            "message": f"Parameter count: {param_count}",
        }
    except _INVARIANT_CAPTURE_ERRORS as exc:
        checks["parameter_count"] = {
            "passed": False,
            "message": f"Could not count parameters: {exc}",
        }

    try:
        has_nan = False
        for param in model.parameters():
            if hasattr(param, "isnan") and param.isnan().any():
                has_nan = True
                break

        checks["no_nan_parameters"] = {
            "passed": not has_nan,
            "message": "NaN parameters detected" if has_nan else "No NaN parameters",
        }
    except _INVARIANT_CAPTURE_ERRORS as exc:
        checks["no_nan_parameters"] = {
            "passed": False,
            "message": f"Could not check for NaN: {exc}",
        }

    return checks


__all__ = [
    "check_adapter_aware_invariants",
    "check_standard_invariants",
    "detect_adapter_type",
]
