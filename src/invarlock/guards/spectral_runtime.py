from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from invarlock.core.types import GuardDiagnostic, GuardValidationResult

from ._estimators import frobenius_norm_sq, row_col_norm_extrema
from .policies import guard_assert
from .spectral_control import apply_spectral_control
from .spectral_detection import (
    classify_model_families,
    compute_family_stats,
    compute_z_scores,
    summarize_family_z_scores,
    summarize_sigmas,
)
from .spectral_policy import apply_policy_overrides, multiple_testing_alpha
from .spectral_results import (
    build_spectral_diagnostics,
    build_spectral_finalize_metrics,
    build_spectral_validation_metrics,
    categorize_spectral_messages,
    compute_family_observability,
    evaluate_spectral_outcome,
    partition_spectral_violations,
    spectral_validation_message,
)


def _typed_guard_diagnostics(
    diagnostics: list[dict[str, Any]],
) -> tuple[GuardDiagnostic, ...]:
    return tuple(
        GuardDiagnostic(
            kind=str(item.get("kind", "spectral_violation")),
            severity=str(item.get("severity", "warning")),
            message=str(item.get("message", "")),
            details={
                str(key): value
                for key, value in item.items()
                if key not in {"kind", "severity", "message"}
            },
        )
        for item in diagnostics
    )


def _raise_prepare_failure(message: str, *, error: Exception | None = None) -> None:
    if error is None:
        raise RuntimeError(message)
    raise RuntimeError(message) from error


def prepare_guard(
    guard: Any,
    model: Any,
    adapter: Any,
    calib: Any,
    policy: dict[str, Any],
    *,
    apply_policy_overrides_fn: Any = apply_policy_overrides,
    classify_model_families_fn: Any = classify_model_families,
    compute_family_stats_fn: Any = compute_family_stats,
    summarize_sigmas_fn: Any = summarize_sigmas,
    percentile_fn: Any = np.percentile,
) -> dict[str, Any]:
    """Prepare spectral guard by capturing baseline spectral properties."""
    _ = adapter
    _ = calib
    start_time = time.time()

    if policy:
        apply_policy_overrides_fn(guard, policy)

    guard._log_event(
        "prepare",
        message=(
            f"Preparing spectral guard with scope={guard.scope}, "
            f"sigma_quantile={guard.sigma_quantile}"
        ),
    )

    try:
        scoped_modules = guard._get_scoped_modules(model)
        guard.baseline_sigmas = guard._capture_sigmas(model, phase="prepare")
        guard.module_family_map = classify_model_families_fn(
            model,
            scope=guard.scope,
            existing=guard.module_family_map,
            modules=scoped_modules,
        )
        if not guard.baseline_family_stats:
            guard.baseline_family_stats = compute_family_stats_fn(
                guard.baseline_sigmas, guard.module_family_map
            )

        baseline_stats: dict[str, Any] = summarize_sigmas_fn(guard.baseline_sigmas)
        values = np.array(list(guard.baseline_sigmas.values()), dtype=float)
        if values.size:
            try:
                guard.target_sigma = float(
                    percentile_fn(values, float(guard.sigma_quantile) * 100.0)
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                _raise_prepare_failure(
                    "Spectral target-sigma computation failed.",
                    error=exc,
                )
        else:
            guard.target_sigma = float(guard.sigma_quantile)
        baseline_stats["target_sigma"] = guard.target_sigma

        guard.baseline_degeneracy = {}
        degeneracy_diagnostics: list[dict[str, Any]] = []
        if bool((guard.degeneracy or {}).get("enabled")):
            eps = 1e-12
            for name, module in scoped_modules:
                weight = getattr(module, "weight", None)
                if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                    continue
                sigma = guard.baseline_sigmas.get(name)
                if not (isinstance(sigma, int | float) and math.isfinite(float(sigma))):
                    continue
                try:
                    stable_rank = frobenius_norm_sq(weight) / max(
                        float(sigma) ** 2, eps
                    )
                    norms = row_col_norm_extrema(weight, eps=eps)
                    row_med = max(float(norms.get("row_median", 0.0)), eps)
                    col_med = max(float(norms.get("col_median", 0.0)), eps)
                    collapse = min(
                        float(norms.get("row_min", 0.0)) / row_med,
                        float(norms.get("col_min", 0.0)) / col_med,
                    )
                    guard.baseline_degeneracy[name] = {
                        "stable_rank": float(stable_rank),
                        "norm_collapse": float(collapse),
                    }
                except (RuntimeError, TypeError, ValueError) as exc:
                    degeneracy_diagnostics.append(
                        {
                            "kind": "spectral_degeneracy_unavailable",
                            "severity": "warning",
                            "message": f"Degeneracy metrics unavailable for {name}.",
                            "details": {"module": name, "error": str(exc)},
                        }
                    )
        baseline_stats["baseline_degeneracy"] = {
            name: values.copy() for name, values in guard.baseline_degeneracy.items()
        }
        if degeneracy_diagnostics:
            baseline_stats["degeneracy_diagnostics"] = degeneracy_diagnostics
        baseline_stats["family_stats"] = {
            family: stats.copy()
            for family, stats in guard.baseline_family_stats.items()
        }
        baseline_stats["family_caps"] = {
            family: caps.copy() for family, caps in guard.family_caps.items()
        }
        baseline_stats["module_sigmas"] = guard.baseline_sigmas.copy()
        baseline_stats["measurement_contract"] = {
            "estimator": guard.estimator,
            "degeneracy": guard.degeneracy,
        }

        guard.baseline_metrics = baseline_stats
        guard.prepared = True
        preparation_time = time.time() - start_time
        guard._log_event(
            "prepare_success",
            message=f"Prepared spectral guard with {len(guard.baseline_metrics)} baseline metrics",
            baseline_metrics_count=len(guard.baseline_metrics),
            target_sigma=guard.target_sigma,
            preparation_time=preparation_time,
        )
        return {
            "ready": True,
            "baseline_metrics": guard.baseline_metrics,
            "target_sigma": guard.target_sigma,
            "scope": guard.scope,
            "preparation_time": preparation_time,
        }
    except (RuntimeError, TypeError, ValueError) as error:
        guard.prepared = False
        guard._log_event(
            "prepare_failed",
            level="ERROR",
            message=f"Failed to prepare spectral guard: {str(error)}",
            error=str(error),
        )
        raise RuntimeError("Failed to prepare spectral guard.") from error


def before_edit_guard(
    guard: Any, model: Any, *, compute_z_scores_fn: Any = compute_z_scores
) -> None:
    """Capture pre-edit state for spectral comparison."""
    if not guard.prepared:
        guard._log_event(
            "before_edit_skipped",
            level="WARN",
            message="Spectral guard not prepared, skipping pre-edit capture",
        )
        return

    guard.pre_edit_metrics = guard._capture_sigmas(model, phase="before_edit")
    guard.pre_edit_z_scores = compute_z_scores_fn(
        guard.pre_edit_metrics,
        guard.baseline_family_stats,
        guard.module_family_map,
        guard.baseline_sigmas,
        deadband=guard.deadband,
    )
    guard._log_event("before_edit", message="Captured pre-edit spectral state")


def after_edit_guard(
    guard: Any,
    model: Any,
    *,
    apply_spectral_control_fn: Any = apply_spectral_control,
) -> None:
    """Capture post-edit state and apply control if needed."""
    if not guard.prepared:
        guard._log_event(
            "after_edit_skipped",
            level="WARN",
            message="Spectral guard not prepared, skipping post-edit analysis",
        )
        return

    try:
        guard.current_metrics = guard._capture_sigmas(model, phase="after_edit")
        violations = guard._detect_spectral_violations(
            model, guard.current_metrics, phase="after_edit"
        )
        guard.violations = violations
        if violations and guard.correction_enabled:
            control_result = apply_spectral_control_fn(
                model,
                policy={
                    "sigma_quantile": guard.sigma_quantile,
                    "scope": guard.scope,
                    "baseline_sigmas": guard.baseline_sigmas,
                    "target_sigma": guard.target_sigma,
                },
            )
            guard._log_event(
                "spectral_control_applied",
                message=f"Applied spectral control, violations: {len(violations)}",
                violations_count=len(violations),
                control_result=control_result,
            )

        guard._log_event(
            "after_edit",
            message=f"Post-edit analysis complete, {len(violations)} violations detected",
        )
    except (RuntimeError, TypeError, ValueError) as error:
        guard._log_event(
            "after_edit_failed",
            level="ERROR",
            message=f"Post-edit spectral analysis failed: {str(error)}",
            error=str(error),
        )
        raise RuntimeError("Post-edit spectral analysis failed.") from error


def validate_guard(
    guard: Any, model: Any, adapter: Any, context: dict[str, Any]
) -> GuardValidationResult:
    """Validate model spectral properties."""
    _ = context
    if not guard.prepared:
        guard.prepare(model, adapter, None, {})

    current_metrics = guard._capture_sigmas(model, phase="validate")
    violations = guard._detect_spectral_violations(
        model, current_metrics, phase="validate"
    )
    fatal_violations, budgeted_violations = partition_spectral_violations(violations)
    selected_budgeted, mt_selection = guard._select_budgeted_violations(
        budgeted_violations
    )
    outcome = evaluate_spectral_outcome(
        fatal_violations=fatal_violations,
        budgeted_violations=budgeted_violations,
        selected_budgeted=selected_budgeted,
        max_caps=int(guard.max_caps),
    )
    selected_violations = outcome["selected_violations"]
    candidate_budgeted = int(outcome["candidate_budgeted"])
    caps_applied = int(outcome["caps_applied"])
    caps_exceeded = bool(outcome["caps_exceeded"])
    passed = bool(outcome["passed"])
    decision = str(outcome["decision"])

    family_summary = summarize_family_z_scores(
        guard.latest_z_scores, guard.module_family_map, guard.family_caps
    )
    family_quantiles, top_z_scores = compute_family_observability(
        guard.latest_z_scores or {}, guard.module_family_map
    )
    metrics = build_spectral_validation_metrics(
        current_metrics=current_metrics,
        candidate_violations=violations,
        selected_violations=selected_violations,
        fatal_violations=fatal_violations,
        candidate_budgeted=candidate_budgeted,
        caps_applied=caps_applied,
        caps_exceeded=caps_exceeded,
        family_summary=family_summary,
        family_caps=guard.family_caps,
        sigma_quantile=guard.sigma_quantile,
        deadband=guard.deadband,
        max_caps=int(guard.max_caps),
        multiple_testing=guard.multiple_testing,
        multiple_testing_selection=mt_selection,
        estimator=guard.estimator,
        degeneracy=guard.degeneracy,
        family_quantiles=family_quantiles,
        top_z_scores=top_z_scores,
    )
    _ = spectral_validation_message(
        passed=passed,
        fatal_violations=fatal_violations,
        caps_applied=caps_applied,
        max_caps=int(guard.max_caps),
    )

    alpha = multiple_testing_alpha(guard.multiple_testing)
    guard_assert(guard.deadband >= 0.0, "spectral.deadband must be >= 0")
    guard_assert(0.0 < alpha <= 1.0, "spectral.multiple_testing.alpha out of range")
    guard_assert(guard.max_caps >= 0, "spectral.max_caps must be >= 0")

    diagnostics = [
        *getattr(guard, "_measurement_diagnostics", []),
        *build_spectral_diagnostics(selected_violations),
    ]
    return GuardValidationResult(
        passed=passed,
        decision=decision,
        metrics=metrics,
        diagnostics=_typed_guard_diagnostics(diagnostics),
        policy=guard._serialize_policy(),
        details={},
        violations=tuple(dict(item) for item in selected_violations),
        extras={
            "final_z_scores": guard.latest_z_scores.copy(),
            "module_family_map": dict(guard.module_family_map),
        },
    )


def finalize_guard(guard: Any, model: Any) -> dict[str, Any]:
    """Finalize spectral guard and return comprehensive results."""
    if not guard.prepared:
        return {
            "passed": False,
            "metrics": {},
            "warnings": ["Spectral guard not properly prepared"],
            "errors": ["Preparation failed or not called"],
            "diagnostics": [
                {
                    "kind": "spectral_preparation",
                    "severity": "error",
                    "message": "Preparation failed or not called",
                }
            ],
        }

    final_metrics = guard._capture_sigmas(model, phase="finalize")
    final_violations = guard._detect_spectral_violations(
        model, final_metrics, phase="finalize"
    )
    final_z_summary = summarize_family_z_scores(
        guard.latest_z_scores, guard.module_family_map, guard.family_caps
    )
    final_family_stats = compute_family_stats(final_metrics, guard.module_family_map)
    family_quantiles, top_z_scores = compute_family_observability(
        guard.latest_z_scores or {}, guard.module_family_map
    )

    fatal_violations, budgeted_violations = partition_spectral_violations(
        final_violations
    )
    selected_budgeted, mt_selection = guard._select_budgeted_violations(
        budgeted_violations
    )
    outcome = evaluate_spectral_outcome(
        fatal_violations=fatal_violations,
        budgeted_violations=budgeted_violations,
        selected_budgeted=selected_budgeted,
        max_caps=int(guard.max_caps),
    )
    selected_final_violations = outcome["selected_violations"]
    candidate_budgeted = int(outcome["candidate_budgeted"])
    caps_applied = int(outcome["caps_applied"])
    caps_exceeded = bool(outcome["caps_exceeded"])
    passed = bool(outcome["passed"])
    decision = str(outcome["decision"])

    metrics = build_spectral_finalize_metrics(
        final_metrics=final_metrics,
        selected_violations=selected_final_violations,
        candidate_violations=final_violations,
        fatal_violations=fatal_violations,
        candidate_budgeted=candidate_budgeted,
        caps_applied=caps_applied,
        caps_exceeded=caps_exceeded,
        baseline_metrics=guard.baseline_metrics,
        scope=guard.scope,
        correction_enabled=guard.correction_enabled,
        family_caps=guard.family_caps,
        final_z_summary=final_z_summary,
        final_family_stats=final_family_stats,
        sigma_quantile=guard.sigma_quantile,
        deadband=guard.deadband,
        max_caps=int(guard.max_caps),
        multiple_testing=guard.multiple_testing,
        multiple_testing_selection=mt_selection,
        estimator=guard.estimator,
        degeneracy=guard.degeneracy,
        family_quantiles=family_quantiles,
        top_z_scores=top_z_scores,
    )
    metrics["target_sigma"] = guard.target_sigma
    warnings, errors = categorize_spectral_messages(selected_final_violations)

    result = {
        "passed": passed,
        "decision": decision,
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
        "violations": selected_final_violations,
        "diagnostics": [
            *getattr(guard, "_measurement_diagnostics", []),
            *build_spectral_diagnostics(selected_final_violations),
        ],
        "baseline_metrics": guard.baseline_metrics,
        "final_metrics": final_metrics,
        "final_z_scores": guard.latest_z_scores,
        "module_family_map": dict(guard.module_family_map),
        "policy": guard._serialize_policy(),
    }

    return result


__all__ = [
    "after_edit_guard",
    "before_edit_guard",
    "finalize_guard",
    "prepare_guard",
    "validate_guard",
]
