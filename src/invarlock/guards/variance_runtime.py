from __future__ import annotations

import time
from typing import Any

import torch.nn as nn

from invarlock.core.types import GuardDiagnostic, GuardValidationResult

from .variance_results import (
    build_finalize_metrics,
    build_finalize_result,
    evaluate_finalize_state,
)


def _build_diagnostics(
    *, warnings: list[str], errors: list[str]
) -> tuple[GuardDiagnostic, ...]:
    diagnostics = [
        GuardDiagnostic(
            kind="variance_warning",
            severity="warning",
            message=message,
        )
        for message in warnings
    ]
    diagnostics.extend(
        GuardDiagnostic(
            kind="variance_error",
            severity="error",
            message=message,
        )
        for message in errors
    )
    return tuple(diagnostics)


def before_edit_guard(guard: Any, model: nn.Module) -> None:
    """Execute before edit (no action needed beyond readiness logging)."""
    _ = model
    if guard._prepared:
        guard._log_event("before_edit", message="Variance guard ready for A/B testing")


def after_edit_guard(guard: Any, model: nn.Module) -> None:
    """Refresh post-edit metrics after an edit has completed."""
    if not guard._prepared:
        guard._log_event(
            "after_edit_skipped",
            level="WARN",
            message="Variance guard not prepared, skipping",
        )
        return

    guard._refresh_after_edit_metrics(model)
    guard._log_event(
        "after_edit",
        message="Variance guard refreshed post-edit metrics",
        evaluated=guard._post_edit_evaluated,
        proposed_scales=len(guard._scales),
    )


def validate_guard(
    guard: Any, model: Any, adapter: Any, context: dict[str, Any]
) -> GuardValidationResult:
    """Validate model state through the finalized variance result."""
    _ = adapter
    _ = context
    result = guard.finalize(model)
    details = result.get("details", {}) or {}
    errors = result.get("errors", []) or []
    warnings = result.get("warnings", []) or []
    passed = result.get("passed", False)

    if passed:
        decision = "monitor" if warnings else "allow"
    else:
        decision = "monitor" if guard._monitor_only else "block"

    violations = tuple(
        {
            "type": "variance_error",
            "severity": "error",
            "message": str(message),
        }
        for message in errors
    )
    reason = (
        "no_variance_targets"
        if any("no target modules found" in str(message).lower() for message in errors)
        else None
    )
    extras = (
        {
            "supported": False,
            "reason": reason,
            "assurance_blocking": True,
            "status": "unsupported",
        }
        if reason is not None
        else None
    )
    return GuardValidationResult(
        passed=bool(passed),
        decision=decision,
        metrics=dict(result.get("metrics", {})),
        diagnostics=_build_diagnostics(warnings=warnings, errors=errors),
        policy=dict(details.get("policy", guard._policy.copy()) or {}),
        details=dict(details),
        violations=violations,
        extras=extras,
    )


def finalize_guard(guard: Any, model: nn.Module) -> dict[str, Any]:
    """Finalize variance guard and return comprehensive results."""
    start_time = time.time()

    if not guard._prepared:
        guard._log_event(
            "finalize_failed",
            level="ERROR",
            message="Variance guard not properly prepared",
        )
        return {
            "passed": False,
            "metrics": {},
            "warnings": ["Variance guard not properly prepared"],
            "errors": ["Preparation failed or no target modules found"],
            "finalize_time": time.time() - start_time,
            "diagnostics": list(
                _build_diagnostics(
                    warnings=["Variance guard not properly prepared"],
                    errors=["Preparation failed or no target modules found"],
                )
            ),
        }

    if guard._monitor_only:
        guard._enabled = False
        guard._scales = {}

    if not guard._post_edit_evaluated:
        guard._refresh_after_edit_metrics(model)

    should_enable, gate_reason = guard._evaluate_ab_gate()
    enabled_after_ab = guard._enabled
    ab_gain = guard._ab_gain or 0.0

    if should_enable and not enabled_after_ab:
        enable_result = guard.enable(model)
        enabled_after_ab = enable_result or guard._enabled
    elif not should_enable and enabled_after_ab:
        guard.disable(model)
        enabled_after_ab = False

    guard._log_event(
        "ab_gate_evaluation",
        message=f"A/B gate decision: should_enable={should_enable}, reason={gate_reason}",
        should_enable=should_enable,
        reason=gate_reason,
        current_enabled=enabled_after_ab,
    )
    required_gain_with_deadband = guard._policy["min_gain"] + float(
        guard._policy.get("tie_breaker_deadband", guard.TIE_BREAKER_DEADBAND)
    )
    finalize_state = evaluate_finalize_state(
        should_enable=should_enable,
        enabled_after_ab=enabled_after_ab,
        gate_reason=gate_reason,
        ppl_no_ve=guard._ppl_no_ve,
        ppl_with_ve=guard._ppl_with_ve,
        final_ppl=getattr(guard, "_final_ppl", None),
        ab_windows_used=guard._ab_windows_used,
        ab_seed_used=guard._ab_seed_used,
        expected_seed=int(guard._policy.get("seed", 123)),
        enable_attempt_count=guard._enable_attempt_count,
        disable_attempt_count=guard._disable_attempt_count,
        checkpoint_depth=len(guard._checkpoint_stack),
        ab_gain=guard._ab_gain,
        required_gain_with_deadband=required_gain_with_deadband,
        absolute_floor=guard.ABSOLUTE_FLOOR,
        calibration_status=str(guard._calibration_stats.get("status")),
    )
    passed = bool(finalize_state["passed"])
    warnings = list(finalize_state["warnings"])
    errors = list(finalize_state["errors"])

    finalize_time = time.time() - start_time
    final_metrics = build_finalize_metrics(
        scales=guard._scales,
        target_modules=guard._target_modules,
        stats=guard._stats,
        focus_modules=guard._focus_modules,
        enabled_after_ab=enabled_after_ab,
        should_enable=should_enable,
        ab_gain=ab_gain,
        ab_windows_used=guard._ab_windows_used,
        ab_seed_used=guard._ab_seed_used,
        monitor_only=guard._monitor_only,
        policy=guard._policy,
        ppl_no_ve=guard._ppl_no_ve,
        ppl_with_ve=guard._ppl_with_ve,
        ratio_ci=guard._ratio_ci,
        calibration_stats=guard._calibration_stats,
        predictive_gate_state=guard._predictive_gate_state,
        raw_scales_pre_edit=guard._raw_scales_pre_edit,
        raw_scales_post_edit=guard._raw_scales_post_edit,
    )

    guard._log_event(
        "finalize_complete",
        message=f"Variance guard finalized - {'PASSED' if passed else 'FAILED'}",
        passed=passed,
        ve_enabled=enabled_after_ab,
        ab_gain=ab_gain,
        finalize_time=finalize_time,
    )

    result = build_finalize_result(
        passed=passed,
        metrics=final_metrics,
        warnings=warnings,
        errors=errors,
        finalize_time=finalize_time,
        enabled_after_ab=enabled_after_ab,
        ppl_no_ve=guard._ppl_no_ve,
        scales=guard._scales,
        stats=guard._stats,
        policy=guard._policy,
    )
    result["decision"] = (
        "allow"
        if passed and not warnings
        else ("monitor" if passed or guard._monitor_only else "block")
    )
    result["diagnostics"] = [
        {
            "kind": diagnostic.kind,
            "severity": diagnostic.severity,
            "message": diagnostic.message,
            "details": dict(diagnostic.details),
        }
        for diagnostic in _build_diagnostics(warnings=warnings, errors=errors)
    ]

    return result


__all__ = [
    "after_edit_guard",
    "before_edit_guard",
    "finalize_guard",
    "validate_guard",
]
