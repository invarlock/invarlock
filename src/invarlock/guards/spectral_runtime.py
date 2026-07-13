from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from invarlock.core.types import GuardDiagnostic, GuardValidationResult

from . import spectral_correction
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


def _spectral_measurement_contract(guard: Any) -> dict[str, Any]:
    return {
        "estimator": dict(guard.estimator),
        "degeneracy": dict(guard.degeneracy),
    }


def _identity_changes(guard: Any) -> list[str]:
    changed: set[str] = set()
    for inventory in (getattr(guard, "measurement_inventory", {}) or {}).values():
        if not isinstance(inventory, dict):
            continue
        raw = inventory.get("identity_changed_modules")
        if isinstance(raw, list):
            changed.update(str(name) for name in raw if isinstance(name, str))
    return sorted(changed)


def _measurement_exclusions(guard: Any) -> list[dict[str, str]]:
    exclusions: dict[tuple[str, str], dict[str, str]] = {}
    for phase, inventory in (getattr(guard, "measurement_inventory", {}) or {}).items():
        if not isinstance(inventory, dict):
            continue
        raw_entries = inventory.get("excluded_modules")
        if not isinstance(raw_entries, list):
            continue
        for raw in raw_entries:
            if not isinstance(raw, dict) or raw.get("stage") != "measurement":
                continue
            module = str(raw.get("module") or "")
            reason = str(raw.get("reason") or "measurement_unavailable")
            exclusions[(str(phase), module)] = {
                "phase": str(phase),
                "module": module,
                "reason": reason,
            }
    return [exclusions[key] for key in sorted(exclusions)]


def _discovery_errors(guard: Any) -> list[dict[str, str]]:
    errors: set[tuple[str, str]] = set()
    for phase, inventory in (getattr(guard, "measurement_inventory", {}) or {}).items():
        if not isinstance(inventory, dict):
            continue
        raw = inventory.get("discovery_errors")
        if isinstance(raw, list):
            errors.update(
                (str(phase), reason) for reason in raw if isinstance(reason, str)
            )
    return [{"phase": phase, "reason": reason} for phase, reason in sorted(errors)]


def _unsupported_spectral_result(
    guard: Any,
    *,
    reason: str,
    modules_checked: int,
) -> GuardValidationResult:
    message = f"Spectral assurance unavailable: {reason}"
    return GuardValidationResult(
        passed=False,
        decision="block",
        metrics={
            "modules_checked": int(modules_checked),
            "measurement_contract": _spectral_measurement_contract(guard),
            "external_baseline_required": bool(guard._external_baseline_required),
            "external_baseline_ready": bool(guard._external_baseline_ready),
        },
        diagnostics=(
            GuardDiagnostic(
                kind="spectral_unsupported",
                severity="error",
                message=message,
                details={"reason": reason, "modules_checked": int(modules_checked)},
            ),
        ),
        policy=guard._serialize_policy(),
        violations=(
            {
                "type": "spectral_unsupported",
                "severity": "error",
                "reason": reason,
                "message": message,
            },
        ),
        extras={
            "supported": False,
            "reason": reason,
            "assurance_blocking": True,
            "status": "unsupported",
            "measurement_inventory": {
                phase: dict(inventory)
                for phase, inventory in (
                    getattr(guard, "measurement_inventory", {}) or {}
                ).items()
            },
        },
    )


def load_external_baseline_evidence(guard: Any) -> dict[str, Any]:
    """Load baseline-run measurements into a subject Spectral guard.

    ``prepare`` still measures the subject so that module and measurement
    coverage can be checked.  This function then replaces only the comparison
    reference.  Invalid evidence is retained as an explicit blocking reason;
    it never falls back to the subject-local reference.
    """

    guard._external_baseline_ready = False
    guard._external_baseline_reason = None
    if not guard._external_baseline_required:
        return {"ready": False, "required": False, "reason": "not_required"}

    evidence = guard._external_baseline_evidence
    if not isinstance(evidence, dict):
        guard._external_baseline_reason = "baseline_spectral_evidence_missing"
        return {
            "ready": False,
            "required": True,
            "reason": guard._external_baseline_reason,
        }

    metrics = evidence.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    baseline_metrics = evidence.get("baseline_metrics")
    baseline_metrics = baseline_metrics if isinstance(baseline_metrics, dict) else {}
    final_metrics = evidence.get("final_metrics")
    final_metrics = final_metrics if isinstance(final_metrics, dict) else {}
    raw_sigmas = baseline_metrics.get("module_sigmas") or final_metrics
    if not isinstance(raw_sigmas, dict) or not raw_sigmas:
        guard._external_baseline_reason = (
            "baseline_spectral_module_measurements_missing"
        )
        return {
            "ready": False,
            "required": True,
            "reason": guard._external_baseline_reason,
        }

    external_sigmas: dict[str, float] = {}
    for name, value in raw_sigmas.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric >= 0.0:
            external_sigmas[str(name)] = numeric
    local_names = set(guard.baseline_sigmas)
    external_names = set(external_sigmas)
    if not local_names or local_names != external_names:
        guard._external_baseline_reason = "baseline_spectral_module_coverage_mismatch"
        return {
            "ready": False,
            "required": True,
            "reason": guard._external_baseline_reason,
            "missing": sorted(local_names - external_names),
            "unexpected": sorted(external_names - local_names),
        }

    external_contract = baseline_metrics.get("measurement_contract") or metrics.get(
        "measurement_contract"
    )
    if external_contract != _spectral_measurement_contract(guard):
        guard._external_baseline_reason = (
            "baseline_spectral_measurement_contract_mismatch"
        )
        return {
            "ready": False,
            "required": True,
            "reason": guard._external_baseline_reason,
        }

    raw_family_map = evidence.get("module_family_map")
    if not isinstance(raw_family_map, dict):
        guard._external_baseline_reason = "baseline_spectral_family_map_missing"
        return {
            "ready": False,
            "required": True,
            "reason": guard._external_baseline_reason,
        }
    external_family_map = {
        str(name): str(family)
        for name, family in raw_family_map.items()
        if str(name) in external_names and isinstance(family, str) and family
    }
    if set(external_family_map) != external_names:
        guard._external_baseline_reason = "baseline_spectral_family_coverage_mismatch"
        return {
            "ready": False,
            "required": True,
            "reason": guard._external_baseline_reason,
        }

    guard.baseline_sigmas = external_sigmas
    guard.module_family_map = external_family_map
    guard.baseline_family_stats = compute_family_stats(
        guard.baseline_sigmas, guard.module_family_map
    )
    raw_degeneracy = baseline_metrics.get("baseline_degeneracy")
    guard.baseline_degeneracy = (
        {
            str(name): dict(values)
            for name, values in raw_degeneracy.items()
            if isinstance(values, dict)
        }
        if isinstance(raw_degeneracy, dict)
        else {}
    )
    guard.baseline_metrics = dict(baseline_metrics)
    guard.baseline_metrics["module_sigmas"] = dict(external_sigmas)
    guard.baseline_metrics["family_stats"] = {
        family: dict(values) for family, values in guard.baseline_family_stats.items()
    }
    guard._external_baseline_ready = True
    return {
        "ready": True,
        "required": True,
        "modules": len(external_sigmas),
        "measurement_contract": _spectral_measurement_contract(guard),
    }


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
    _ = calib
    start_time = time.time()
    guard._adapter_ref = adapter
    guard._scoped_modules_model_id = None

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
        fatal_violations, budgeted_violations = partition_spectral_violations(
            violations
        )
        selected_budgeted, mt_selection = guard._select_budgeted_violations(
            budgeted_violations
        )
        selected_violations = [*fatal_violations, *selected_budgeted]
        guard.current_metrics, correction_ledger = (
            spectral_correction.run_correction_lifecycle(
                guard,
                model,
                phase="after_edit",
                pre_correction_metrics=guard.current_metrics,
                selected_violations=selected_violations,
                multiple_testing_selection=mt_selection,
                apply_spectral_control_fn=apply_spectral_control_fn,
            )
        )
        guard.correction_ledger = correction_ledger
        if selected_violations and guard.correction_enabled:
            guard._log_event(
                "spectral_control_attempted",
                message=(
                    "Spectral control lifecycle completed for "
                    f"{len(selected_violations)} selected findings"
                ),
                selected_findings_count=len(selected_violations),
                correction_ledger=correction_ledger,
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
    if not current_metrics:
        return _unsupported_spectral_result(
            guard,
            reason="no_eligible_modules_measured",
            modules_checked=0,
        )
    if guard._external_baseline_required and not guard._external_baseline_ready:
        return _unsupported_spectral_result(
            guard,
            reason=(
                guard._external_baseline_reason
                or "baseline_spectral_evidence_unavailable"
            ),
            modules_checked=len(current_metrics),
        )
    if guard._external_baseline_required and set(current_metrics) != set(
        guard.baseline_sigmas
    ):
        guard._external_baseline_ready = False
        guard._external_baseline_reason = "subject_spectral_module_coverage_mismatch"
        return _unsupported_spectral_result(
            guard,
            reason=guard._external_baseline_reason,
            modules_checked=len(current_metrics),
        )
    violations = guard._detect_spectral_violations(
        model, current_metrics, phase="validate"
    )
    fatal_violations, budgeted_violations = partition_spectral_violations(violations)
    selected_budgeted, mt_selection = guard._select_budgeted_violations(
        budgeted_violations
    )
    pre_correction_selected = [*fatal_violations, *selected_budgeted]
    current_metrics, correction_ledger = spectral_correction.run_correction_lifecycle(
        guard,
        model,
        phase="validate",
        pre_correction_metrics=current_metrics,
        selected_violations=pre_correction_selected,
        multiple_testing_selection=mt_selection,
    )
    guard.correction_ledger = correction_ledger
    if bool(correction_ledger.get("correction_enabled")) and pre_correction_selected:
        if not current_metrics:
            return _unsupported_spectral_result(
                guard,
                reason="post_correction_spectral_measurements_missing",
                modules_checked=0,
            )
        violations = guard._detect_spectral_violations(
            model, current_metrics, phase="validate_post_correction"
        )
        fatal_violations, budgeted_violations = partition_spectral_violations(
            violations
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
    selected_violations = outcome["selected_violations"]
    candidate_budgeted = int(outcome["candidate_budgeted"])
    caps_applied = int(outcome["caps_applied"])
    caps_exceeded = bool(outcome["caps_exceeded"])
    passed = bool(outcome["passed"])
    decision = str(outcome["decision"])
    if correction_ledger.get("policy_result") in {
        "correction_failed",
        "evidence_incomplete",
    }:
        passed = False
        decision = "block"
    identity_changes = _identity_changes(guard)
    measurement_exclusions = _measurement_exclusions(guard)
    discovery_errors = _discovery_errors(guard)
    if identity_changes or measurement_exclusions or discovery_errors:
        passed = False
        decision = "block"
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
    metrics["baseline_source"] = (
        "external_run" if guard._external_baseline_required else "run_local_prepare"
    )
    metrics["external_baseline_required"] = bool(guard._external_baseline_required)
    metrics["external_baseline_ready"] = bool(guard._external_baseline_ready)
    metrics["baseline_modules"] = len(guard.baseline_sigmas)
    spectral_correction.attach_correction_metrics(metrics, correction_ledger)
    metrics["identity_changed_modules"] = identity_changes
    metrics["measurement_exclusions"] = measurement_exclusions
    metrics["discovery_errors"] = discovery_errors
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
            "baseline_metrics": dict(guard.baseline_metrics),
            "final_metrics": dict(current_metrics),
            "final_degeneracy": {
                name: dict(values)
                for name, values in (
                    getattr(guard, "latest_degeneracy", {}) or {}
                ).items()
            },
            "measurement_inventory": {
                phase: dict(inventory)
                for phase, inventory in (
                    getattr(guard, "measurement_inventory", {}) or {}
                ).items()
            },
            "correction_ledger": dict(correction_ledger),
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
    fatal_violations, budgeted_violations = partition_spectral_violations(
        final_violations
    )
    selected_budgeted, mt_selection = guard._select_budgeted_violations(
        budgeted_violations
    )
    pre_correction_selected = [*fatal_violations, *selected_budgeted]
    final_metrics, correction_ledger = spectral_correction.run_correction_lifecycle(
        guard,
        model,
        phase="finalize",
        pre_correction_metrics=final_metrics,
        selected_violations=pre_correction_selected,
        multiple_testing_selection=mt_selection,
    )
    guard.correction_ledger = correction_ledger
    if bool(correction_ledger.get("correction_enabled")) and pre_correction_selected:
        final_violations = guard._detect_spectral_violations(
            model, final_metrics, phase="finalize_post_correction"
        )
        fatal_violations, budgeted_violations = partition_spectral_violations(
            final_violations
        )
        selected_budgeted, mt_selection = guard._select_budgeted_violations(
            budgeted_violations
        )
    final_z_summary = summarize_family_z_scores(
        guard.latest_z_scores, guard.module_family_map, guard.family_caps
    )
    final_family_stats = compute_family_stats(final_metrics, guard.module_family_map)
    family_quantiles, top_z_scores = compute_family_observability(
        guard.latest_z_scores or {}, guard.module_family_map
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
    if correction_ledger.get("policy_result") in {
        "correction_failed",
        "evidence_incomplete",
    }:
        passed = False
        decision = "block"
    identity_changes = _identity_changes(guard)
    measurement_exclusions = _measurement_exclusions(guard)
    discovery_errors = _discovery_errors(guard)
    if identity_changes or measurement_exclusions or discovery_errors:
        passed = False
        decision = "block"

    metrics = build_spectral_finalize_metrics(
        final_metrics=final_metrics,
        selected_violations=selected_final_violations,
        candidate_violations=final_violations,
        fatal_violations=fatal_violations,
        candidate_budgeted=candidate_budgeted,
        caps_applied=caps_applied,
        caps_exceeded=caps_exceeded,
        baseline_metrics=guard.baseline_sigmas,
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
    spectral_correction.attach_correction_metrics(metrics, correction_ledger)
    metrics["identity_changed_modules"] = identity_changes
    metrics["measurement_exclusions"] = measurement_exclusions
    metrics["discovery_errors"] = discovery_errors
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
        "final_degeneracy": {
            name: dict(values) for name, values in guard.latest_degeneracy.items()
        },
        "measurement_inventory": {
            phase: dict(inventory)
            for phase, inventory in guard.measurement_inventory.items()
        },
        "correction_ledger": dict(correction_ledger),
        "module_family_map": dict(guard.module_family_map),
        "policy": guard._serialize_policy(),
    }

    return result


__all__ = [
    "after_edit_guard",
    "before_edit_guard",
    "finalize_guard",
    "load_external_baseline_evidence",
    "prepare_guard",
    "validate_guard",
]
