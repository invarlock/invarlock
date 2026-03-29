from __future__ import annotations

from typing import Any

import torch.nn as nn

from . import rmt_detection, rmt_result_contract
from .rmt_policy import apply_rmt_policy_overrides, compute_epsilon_violations

__all__ = [
    "apply_rmt_detection_and_correction",
    "before_edit_rmt_guard",
    "after_edit_rmt_guard",
    "finalize_rmt_guard",
    "prepare_rmt_guard",
    "validate_rmt_guard",
]


def apply_rmt_detection_and_correction(guard: Any, model: nn.Module) -> dict[str, Any]:
    modules_to_analyze = guard._get_linear_modules(model)
    guard._log_event(
        "rmt_correction",
        message=f"Applying Step 5 detection and correction to {len(modules_to_analyze)} modules",
    )
    result = rmt_detection.step5_detect_and_correct_modules(
        modules_to_analyze,
        baseline_sigmas=guard.baseline_sigmas,
        baseline_mp_stats=guard.baseline_mp_stats,
        deadband=guard.deadband,
        margin=guard.margin,
        correct=guard.correct,
        adapter=guard.adapter,
    )
    for event in result.pop("events", []):
        operation = str(event.get("operation", "rmt_event"))
        module_name = event.get("module_name")
        if operation == "rmt_correct":
            guard._log_event(
                operation,
                message=f"Applied correction to {module_name}",
                module_name=module_name,
                pre_ratio=event.get("pre_ratio"),
                threshold=event.get("threshold"),
            )
        elif operation == "rmt_correct_failed":
            guard._log_event(
                operation,
                level="ERROR",
                message=f"Correction failed for {module_name}: {event.get('error')}",
                module_name=module_name,
                error=event.get("error"),
            )
    return result


def prepare_rmt_guard(
    guard: Any,
    model: nn.Module,
    adapter=None,
    calib=None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import time

    start_time = time.time()
    guard._activation_required_failed = False
    guard._activation_required_reason = None
    guard.adapter = adapter

    apply_rmt_policy_overrides(guard, policy)

    guard._log_event(
        "prepare",
        message="Preparing RMT guard baseline activation edge-risk metrics",
    )

    try:
        windows_cfg = guard.activation_sampling.get("windows") or {}
        try:
            window_count = int(windows_cfg.get("count", 0) or 0)
        except (TypeError, ValueError):
            window_count = 0
        guard._calibration_batches = (
            guard._collect_calibration_batches(calib, window_count)
            if calib is not None and window_count > 0
            else []
        )

        guard.baseline_edge_risk_by_family = {}
        guard.baseline_edge_risk_by_module = {}
        guard.edge_risk_by_family = {}
        guard.edge_risk_by_module = {}
        guard.epsilon_violations = []

        if guard._require_activation and not guard._calibration_batches:
            guard._activation_required_failed = True
            guard._activation_required_reason = "activation_required"
            guard._activation_ready = False
            guard.prepared = False
            return rmt_result_contract.build_prepare_result(
                ready=False,
                baseline_metrics={},
                policy_applied=policy or {},
                preparation_time=time.time() - start_time,
                error="Activation batches required but unavailable",
            )

        baseline = (
            guard._compute_activation_edge_risk(model, guard._calibration_batches)
            if guard._calibration_batches
            else None
        )
        if baseline is None:
            if guard._require_activation:
                guard._activation_required_failed = True
                guard._activation_required_reason = "activation_baseline_unavailable"
                guard._activation_ready = False
                guard.prepared = False
                return rmt_result_contract.build_prepare_result(
                    ready=False,
                    baseline_metrics={},
                    policy_applied=policy or {},
                    preparation_time=time.time() - start_time,
                    error="Activation baseline unavailable",
                )
            guard._activation_ready = False
            guard.prepared = True
            return rmt_result_contract.build_prepare_result(
                ready=True,
                baseline_metrics={},
                policy_applied=policy or {},
                preparation_time=time.time() - start_time,
            )

        guard.baseline_edge_risk_by_module = dict(
            baseline.get("edge_risk_by_module") or {}
        )
        guard.baseline_edge_risk_by_family = dict(
            baseline.get("edge_risk_by_family") or {}
        )
        guard._activation_ready = True
        guard.prepared = True

        return rmt_result_contract.build_prepare_result(
            ready=True,
            baseline_metrics={
                "edge_risk_by_family": dict(guard.baseline_edge_risk_by_family),
                "measurement_contract": {
                    "kind": "activation_edge_risk",
                    "estimator": guard.estimator,
                    "activation_sampling": guard.activation_sampling,
                },
            },
            policy_applied=policy or {},
            preparation_time=time.time() - start_time,
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        guard.prepared = False
        guard._log_event(
            "prepare_failed",
            level="ERROR",
            message=f"Failed to prepare RMT guard: {str(exc)}",
            error=str(exc),
        )
        return rmt_result_contract.build_prepare_result(
            ready=False,
            baseline_metrics={},
            policy_applied=policy or {},
            preparation_time=time.time() - start_time,
            error=str(exc),
        )


def before_edit_rmt_guard(guard: Any, model: nn.Module) -> None:
    _ = model
    if guard.prepared:
        guard._log_event(
            "before_edit",
            message="RMT guard ready for post-edit detection and correction",
        )


def after_edit_rmt_guard(guard: Any, model: nn.Module) -> None:
    if not guard.prepared:
        guard._log_event(
            "after_edit_skipped",
            level="WARN",
            message="RMT guard not prepared, skipping post-edit detection",
        )
        return

    try:
        if guard._require_activation and not guard._calibration_batches:
            guard._activation_required_failed = True
            guard._activation_required_reason = "activation_unavailable"
            guard._last_result = rmt_result_contract.build_after_edit_result()
            return

        current = (
            guard._compute_activation_edge_risk(model, guard._calibration_batches)
            if guard._calibration_batches
            else None
        )
        if current is None:
            if guard._require_activation:
                guard._activation_required_failed = True
                guard._activation_required_reason = "activation_edge_risk_unavailable"
            guard._last_result = rmt_result_contract.build_after_edit_result()
            return

        guard.edge_risk_by_module = dict(current.get("edge_risk_by_module") or {})
        guard.edge_risk_by_family = dict(current.get("edge_risk_by_family") or {})
        guard._last_result = dict(current)
        guard.epsilon_violations = compute_epsilon_violations(guard)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        guard._log_event(
            "after_edit_failed",
            level="ERROR",
            message=f"RMT detection failed: {str(exc)}",
            error=str(exc),
        )
        guard._last_result = rmt_result_contract.build_after_edit_result()
        guard.epsilon_violations = []


def validate_rmt_guard(
    guard: Any, model: Any, adapter: Any, context: dict[str, Any]
) -> dict[str, Any]:
    _ = context
    result = guard.finalize(model, adapter)
    if (
        hasattr(result, "passed")
        and hasattr(result, "action")
        and hasattr(result, "metrics")
    ):
        violations_list: list[str] = []
        if hasattr(result, "violations") and result.violations:
            violations_list = [str(v) for v in result.violations]
        return {
            "passed": bool(result.passed),
            "action": str(result.action),
            "metrics": dict(result.metrics),
            "violations": violations_list,
            "message": "RMT guard validation completed",
        }
    return {
        "passed": result.get("passed", False),
        "action": "continue" if result.get("passed", False) else "warn",
        "metrics": result.get("metrics", {}),
        "violations": result.get("errors", []),
        "message": "RMT guard validation completed",
    }


def finalize_rmt_guard(
    guard: Any,
    model: nn.Module,
    adapter=None,
    *,
    has_guard_outcome: bool,
    guard_outcome_type: Any,
) -> Any:
    import time

    start_time = time.time()
    _ = adapter

    if not guard.prepared:
        if has_guard_outcome:
            return guard_outcome_type(
                name=guard.name,
                passed=False,
                action="abort",
                violations=[
                    {
                        "type": "preparation",
                        "severity": "error",
                        "message": "RMT guard not properly prepared",
                        "module_name": None,
                    }
                ],
                metrics={
                    "prepared": False,
                    "finalize_time": time.time() - start_time,
                },
            )
        return {
            "passed": False,
            "metrics": {
                "prepared": False,
                "finalize_time": time.time() - start_time,
            },
            "errors": ["RMT guard not properly prepared"],
        }

    if guard._require_activation and guard._activation_required_failed:
        reason = guard._activation_required_reason or "activation_required"
        finalize_time = time.time() - start_time
        if has_guard_outcome:
            return guard_outcome_type(
                name=guard.name,
                passed=False,
                action="abort",
                violations=[
                    {
                        "type": "activation_required",
                        "severity": "error",
                        "message": "Activation edge-risk analysis required but unavailable",
                        "module_name": None,
                        "reason": reason,
                    }
                ],
                metrics={
                    "prepared": True,
                    "activation_required": True,
                    "activation_ready": False,
                    "activation_reason": reason,
                    "finalize_time": finalize_time,
                },
            )
        return {
            "passed": False,
            "metrics": {
                "prepared": True,
                "activation_required": True,
                "activation_ready": False,
                "activation_reason": reason,
                "finalize_time": finalize_time,
            },
            "errors": ["Activation edge-risk analysis required but unavailable"],
        }

    if not guard.edge_risk_by_family and guard._calibration_batches:
        current = guard._compute_activation_edge_risk(model, guard._calibration_batches)
        if current is not None:
            guard.edge_risk_by_family = dict(current.get("edge_risk_by_family") or {})
            guard.edge_risk_by_module = dict(current.get("edge_risk_by_module") or {})
            guard._last_result = dict(current)

    guard.epsilon_violations = compute_epsilon_violations(guard)
    from ._contracts import guard_assert

    for fam, eps in guard.epsilon_by_family.items():
        guard_assert(eps >= 0.0, f"rmt.epsilon[{fam}] must be >= 0")

    stable = not guard.epsilon_violations
    action = "continue" if stable else "abort"
    finalize_time = time.time() - start_time

    metrics: dict[str, Any] = {
        "prepared": True,
        "stable": stable,
        "edge_risk_by_family_base": dict(guard.baseline_edge_risk_by_family),
        "edge_risk_by_family": dict(guard.edge_risk_by_family),
        "epsilon_by_family": dict(guard.epsilon_by_family),
        "epsilon_violations": list(guard.epsilon_violations),
        "measurement_contract": {
            "kind": "activation_edge_risk",
            "estimator": guard.estimator,
            "activation_sampling": guard.activation_sampling,
        },
        "finalize_time": finalize_time,
    }

    violations: list[dict[str, Any]] = []
    for v in guard.epsilon_violations:
        violations.append(
            {
                "type": "epsilon_band",
                "severity": "error",
                "family": v.get("family"),
                "edge_base": v.get("edge_base"),
                "edge_cur": v.get("edge_cur"),
                "allowed": v.get("allowed"),
                "epsilon": v.get("epsilon"),
                "delta": v.get("delta"),
                "message": f"ε-band violation in {v.get('family')}",
            }
        )

    if has_guard_outcome:
        return guard_outcome_type(
            name=guard.name,
            passed=stable,
            action=action,
            violations=violations,
            metrics=metrics,
        )
    return {
        "passed": stable,
        "action": action,
        "metrics": metrics,
        "violations": violations,
    }
