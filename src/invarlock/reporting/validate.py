"""
InvarLock Validation Framework
=========================

Validation utilities for checking pruning results against baseline metrics.
Supports both automated CI testing and flexible user validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from invarlock.eval.guard_metric_impact import (
    compute_guard_metric_impact,
    degradation_within_limit,
)

_VALIDATION_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)

__all__ = [
    "validate_against_baseline",
    "validate_drift_gate",
    "validate_guard_metric_impact",
    "ValidationResult",
    "load_baseline",
    "save_baseline",
    "create_baseline_from_report",
]


class ValidationResult:
    """Container for validation results."""

    def __init__(
        self,
        passed: bool,
        checks: dict[str, bool],
        metrics: dict[str, Any],
        messages: list[str],
        warnings: list[str] | None = None,
        errors: list[str] | None = None,
    ):
        self.passed = passed
        self.checks = checks
        self.metrics = metrics
        self.messages = messages
        self.warnings = warnings or []
        self.errors = errors or []

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        """Return typed diagnostics for the result's messages, warnings, and errors."""
        diagnostics: list[dict[str, Any]] = []
        for message in self.messages:
            diagnostics.append(
                {
                    "kind": "validation_info",
                    "severity": "info",
                    "message": str(message),
                    "details": {},
                }
            )
        for warning in self.warnings:
            diagnostics.append(
                {
                    "kind": "validation_warning",
                    "severity": "warning",
                    "message": str(warning),
                    "details": {},
                }
            )
        for error in self.errors:
            diagnostics.append(
                {
                    "kind": "validation_error",
                    "severity": "error",
                    "message": str(error),
                    "details": {},
                }
            )
        return diagnostics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "checks": self.checks,
            "metrics": self.metrics,
            "messages": self.messages,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        passed_count = sum(1 for check in self.checks.values() if check)
        total_count = len(self.checks)

        lines = [
            f"Validation {status} ({passed_count}/{total_count} checks passed)",
            "",
        ]

        # Show individual check results
        for check_name, passed in self.checks.items():
            symbol = "✓" if passed else "✗"
            lines.append(f"  {symbol} {check_name}")

        # Show messages
        if self.messages:
            lines.append("")
            lines.extend(f"  {msg}" for msg in self.messages)

        # Show warnings and errors
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  ⚠️ {warning}" for warning in self.warnings)

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            lines.extend(f"  ❌ {error}" for error in self.errors)

        return "\n".join(lines)


def validate_against_baseline(
    run_report: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tol_ratio: float = 0.02,
    tol_param_ratio: float = 0.02,
    ratio_bounds: tuple[float, float] = (1.25, 1.32),
    delta_bounds_pp: tuple[float, float] | None = None,
    structural_exact: bool = True,
) -> ValidationResult:
    """
    Validate pruning results against baseline metrics (PM-only API).

    Args:
        run_report: Report from pruning run (dict with metrics)
        baseline: Baseline metrics to compare against
        tol_ratio: Tolerance for primary metric ratio deviation (±2% = 0.02) for lower-is-better families
        tol_param_ratio: Tolerance for parameter reduction ratio deviation
        ratio_bounds: Acceptable ratio bounds for lower-is-better families (min, max)
        delta_bounds_pp: Acceptable delta bounds in percentage points for higher-is-better families (min, max)
        structural_exact: Whether structural counts must match exactly

    Returns:
        ValidationResult with detailed check results
    """
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    messages: list[str] = []
    warnings_list: list[str] = []
    errors: list[str] = []

    try:
        # Extract the kind-specific baseline comparison.
        current_ratio = None
        current_delta_pp = None
        pm_kind = None
        pm = (
            (run_report.get("metrics") or {}).get("primary_metric")
            if isinstance(run_report.get("metrics"), dict)
            else None
        )
        if isinstance(pm, dict) and pm:
            try:
                pm_kind = str(pm.get("kind") or "").lower()
            except _VALIDATION_EXCEPTIONS:
                pm_kind = None
            comparison_field = (
                "delta_vs_baseline_pp" if pm_kind == "accuracy" else "ratio_vs_baseline"
            )
            val = pm.get(comparison_field)
            if isinstance(val, int | float):
                if pm_kind == "accuracy":
                    current_delta_pp = float(val)
                else:
                    current_ratio = float(val)
        if pm_kind == "accuracy" and current_delta_pp is None:
            errors.append("Cannot extract delta_vs_baseline_pp from run report")
        elif pm_kind != "accuracy" and current_ratio is None:
            errors.append("Cannot extract ratio_vs_baseline from run report")

        if "param_reduction_ratio" in run_report:
            current_param_ratio = run_report["param_reduction_ratio"]
        elif "parameters_removed" in run_report and "original_params" in run_report:
            current_param_ratio = (
                run_report["parameters_removed"] / run_report["original_params"]
            )
        else:
            current_param_ratio = None
            errors.append("Cannot extract parameter reduction ratio from run report")

        # Extract baseline metrics
        baseline_ratio = baseline.get("ratio_vs_baseline")
        baseline_param_ratio = baseline.get("param_reduction_ratio")

        if pm_kind != "accuracy" and baseline_ratio is None:
            errors.append("Baseline missing ratio_vs_baseline")
        if baseline_param_ratio is None:
            errors.append("Baseline missing param_reduction_ratio")

        # Primary metric tolerance (lower-is-better families)
        if pm_kind in {"ppl_causal", "ppl_mlm", "ppl_seq2seq", None}:
            if current_ratio is not None and baseline_ratio is not None:
                rel_diff = abs(current_ratio - float(baseline_ratio)) / float(
                    baseline_ratio
                )
                checks["ratio_tolerance"] = rel_diff <= tol_ratio
                metrics["ratio_diff"] = rel_diff
                metrics["current_ratio"] = current_ratio
                metrics["baseline_ratio"] = float(baseline_ratio)

                if not checks["ratio_tolerance"]:
                    msg = f"Primary metric ratio deviation {rel_diff:.3f} exceeds tolerance {tol_ratio:.3f}"
                    messages.append(msg)
                else:
                    messages.append(
                        f"Primary metric ratio within tolerance: {current_ratio:.3f} vs baseline {float(baseline_ratio):.3f}"
                    )
            else:
                checks["ratio_tolerance"] = False

        # Parameter ratio validation
        if current_param_ratio is not None and baseline_param_ratio is not None:
            param_relative_diff = (
                abs(current_param_ratio - baseline_param_ratio) / baseline_param_ratio
            )
            checks["param_ratio_tolerance"] = param_relative_diff <= tol_param_ratio
            metrics["param_ratio_diff"] = param_relative_diff
            metrics["current_param_ratio"] = current_param_ratio
            metrics["baseline_param_ratio"] = baseline_param_ratio

            if not checks["param_ratio_tolerance"]:
                messages.append(
                    f"Parameter ratio deviation {param_relative_diff:.3f} exceeds tolerance {tol_param_ratio:.3f}"
                )
            else:
                messages.append(
                    f"Parameter ratio within tolerance: {current_param_ratio:.3f} vs baseline {baseline_param_ratio:.3f}"
                )
        else:
            checks["param_ratio_tolerance"] = False

        # Bounds check
        if pm_kind == "accuracy":
            if current_delta_pp is not None:
                if isinstance(delta_bounds_pp, tuple) and len(delta_bounds_pp) == 2:
                    delta_pp = current_delta_pp
                    lo_pp, hi_pp = float(delta_bounds_pp[0]), float(delta_bounds_pp[1])
                    checks["delta_bounds_pp"] = lo_pp <= delta_pp <= hi_pp
                    if not checks["delta_bounds_pp"]:
                        messages.append(
                            f"Δpp {delta_pp:+.2f} outside acceptable bounds {delta_bounds_pp}"
                        )
                    else:
                        messages.append(
                            f"Δpp {delta_pp:+.2f} within acceptable bounds {delta_bounds_pp}"
                        )
            else:
                checks["delta_bounds_pp"] = False
        elif current_ratio is not None:
            if pm_kind != "accuracy":
                checks["ratio_bounds"] = (
                    ratio_bounds[0] <= current_ratio <= ratio_bounds[1]
                )
                if not checks["ratio_bounds"]:
                    messages.append(
                        f"Ratio {current_ratio:.3f} outside acceptable bounds {ratio_bounds}"
                    )
                else:
                    messages.append(
                        f"Ratio {current_ratio:.3f} within acceptable bounds {ratio_bounds}"
                    )
        else:
            checks["ratio_bounds"] = False

        # Structural count validation
        if structural_exact:
            structural_checks = _validate_structural_counts(run_report, baseline)
            checks.update(structural_checks["checks"])
            messages.extend(structural_checks["messages"])
            warnings_list.extend(structural_checks["warnings"])
        # An explicitly disabled structural comparison is omitted rather than
        # represented as a successful evidence-backed check.

        # Invariants validation (if present in report)
        invariants_passed = _validate_invariants(run_report)
        checks["invariants"] = invariants_passed
        if not invariants_passed:
            errors.append("Model invariants evidence is missing or failed")

        # Overall pass/fail
        passed = all(checks.values()) and len(errors) == 0

        return ValidationResult(
            passed=passed,
            checks=checks,
            metrics=metrics,
            messages=messages,
            warnings=warnings_list,
            errors=errors,
        )

    except _VALIDATION_EXCEPTIONS as e:
        return ValidationResult(
            passed=False,
            checks={"validation_error": False},
            metrics={},
            messages=[],
            warnings=[],
            errors=[f"Validation failed with exception: {str(e)}"],
        )


def validate_drift_gate(
    run_report: dict[str, Any], drift_bounds: tuple[float, float] = (0.95, 1.05)
) -> ValidationResult:
    """
    Validate hard drift gate: 0.95 ≤ final/preview ≤ 1.05.

    Args:
        run_report: Report from run with metrics.primary_metric preview/final
        drift_bounds: Acceptable drift bounds (min, max) - default (0.95, 1.05)

    Returns:
        ValidationResult with drift gate check
    """
    checks = {}
    metrics = {}
    messages = []
    warnings: list[str] = []
    errors = []

    try:
        # Extract preview and final from primary_metric
        pm = (
            (run_report.get("metrics") or {}).get("primary_metric")
            if isinstance(run_report.get("metrics"), dict)
            else None
        )
        pm_preview = pm.get("preview") if isinstance(pm, dict) else None
        pm_final = pm.get("final") if isinstance(pm, dict) else None

        # Calculate drift ratio (final/preview) for lower-is-better families
        if (
            isinstance(pm_preview, (int | float))
            and isinstance(pm_final, (int | float))
            and pm_preview > 0
        ):
            drift_ratio = float(pm_final) / float(pm_preview)
            metrics["drift_ratio"] = drift_ratio
            metrics["preview"] = float(pm_preview)
            metrics["final"] = float(pm_final)

            # Apply hard gate
            checks["drift_gate"] = drift_bounds[0] <= drift_ratio <= drift_bounds[1]

            if checks["drift_gate"]:
                messages.append(
                    f"Drift gate PASSED: {drift_ratio:.3f} within bounds {drift_bounds}"
                )
            else:
                errors.append(
                    f"Drift gate FAILED: {drift_ratio:.3f} outside bounds {drift_bounds} "
                    f"(±5% drift limit exceeded)"
                )
        else:
            errors.append(
                "Cannot calculate drift: missing primary_metric preview/final"
            )
            checks["drift_gate"] = False

        # Overall pass/fail
        passed = all(checks.values()) and len(errors) == 0

        return ValidationResult(
            passed=passed,
            checks=checks,
            metrics=metrics,
            messages=messages,
            warnings=warnings,
            errors=errors,
        )

    except _VALIDATION_EXCEPTIONS as e:
        return ValidationResult(
            passed=False,
            checks={"drift_gate_error": False},
            metrics={},
            messages=[],
            warnings=[],
            errors=[f"Drift gate validation failed: {str(e)}"],
        )


def validate_guard_metric_impact(
    bare_report: dict[str, Any],
    guarded_report: dict[str, Any],
    degradation_limit: float = 0.01,
) -> ValidationResult:
    """
    Validate guard impact for a supported direction-aware primary metric.

    Args:
        bare_report: Report from bare (no guards) run (expects metrics.primary_metric)
        guarded_report: Report from guarded run (expects metrics.primary_metric)
        degradation_limit: Maximum allowed degradation in the metric-owned basis.

    Returns:
        ValidationResult with guard metric impact check
    """
    checks = {}
    metrics = {}
    messages = []
    warnings: list[str] = []
    errors = []

    try:
        if (
            isinstance(degradation_limit, bool)
            or not isinstance(degradation_limit, int | float)
            or not math.isfinite(float(degradation_limit))
            or float(degradation_limit) < 0.0
        ):
            return ValidationResult(
                passed=False,
                checks={"guard_metric_impact": False},
                metrics={},
                messages=[],
                warnings=[],
                errors=[
                    "Cannot calculate guard metric impact: invalid degradation_limit"
                ],
            )
        threshold = float(degradation_limit)

        # Extract matching primary metric finals from both reports.
        bare_pm = (
            (bare_report.get("metrics") or {}).get("primary_metric")
            if isinstance(bare_report.get("metrics"), dict)
            else bare_report.get("primary_metric")
        )
        guarded_pm = (
            (guarded_report.get("metrics") or {}).get("primary_metric")
            if isinstance(guarded_report.get("metrics"), dict)
            else guarded_report.get("primary_metric")
        )

        bare_value = None
        guarded_value = None
        bare_kind = None
        guarded_kind = None
        if isinstance(bare_pm, dict):
            bare_value = bare_pm.get("final")
            bare_kind = bare_pm.get("kind")
        if isinstance(guarded_pm, dict):
            guarded_value = guarded_pm.get("final")
            guarded_kind = guarded_pm.get("kind")

        measurement = (
            compute_guard_metric_impact(bare_kind, bare_value, guarded_value)
            if isinstance(bare_kind, str) and bare_kind == guarded_kind
            else None
        )

        if measurement is not None:
            metrics.update(measurement.to_metrics())
            checks["metric_kind_matches"] = True
            checks["measurements_valid"] = True
            checks["guard_metric_impact"] = degradation_within_limit(
                degradation=measurement.degradation,
                degradation_limit=threshold,
            )

            if checks["guard_metric_impact"]:
                messages.append("Guard metric impact PASSED for retained measurements")
            else:
                errors.append(
                    "Guard metric impact FAILED: retained primary-metric degradation "
                    "exceeds the configured threshold"
                )
        else:
            errors.append(
                "Cannot calculate guard metric impact: expected matching finite "
                "supported primary metrics"
            )
            checks["metric_kind_matches"] = bool(
                isinstance(bare_kind, str) and bare_kind == guarded_kind
            )
            checks["measurements_valid"] = False
            checks["guard_metric_impact"] = False

        # Overall pass/fail
        passed = all(checks.values()) and len(errors) == 0

        return ValidationResult(
            passed=passed,
            checks=checks,
            metrics=metrics,
            messages=messages,
            warnings=warnings,
            errors=errors,
        )

    except _VALIDATION_EXCEPTIONS as e:
        return ValidationResult(
            passed=False,
            checks={"guard_metric_impact_error": False},
            metrics={},
            messages=[],
            warnings=[],
            errors=[f"Guard metric impact validation failed: {str(e)}"],
        )


def _validate_structural_counts(
    run_report: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Validate that structural counts match exactly."""
    checks = {}
    messages = []
    warnings = []

    # Heads/neurons counts removed from simplified schema; only validate layers

    # Check layers modified
    current_layers = run_report.get(
        "layers_modified", run_report.get("metrics", {}).get("layers_modified")
    )
    baseline_layers = baseline.get("layers_modified")

    if current_layers is not None and baseline_layers is not None:
        checks["layers_count_exact"] = current_layers == baseline_layers
        if checks["layers_count_exact"]:
            messages.append(f"Modified layers count matches: {current_layers}")
        else:
            messages.append(
                f"Modified layers mismatch: {current_layers} vs baseline {baseline_layers}"
            )
    else:
        warnings.append("Cannot validate layers count - missing data")
        checks["layers_count_exact"] = False

    return {"checks": checks, "messages": messages, "warnings": warnings}


def _validate_invariants(run_report: dict[str, Any]) -> bool:
    """Check if model invariants passed."""
    # Look for invariants check in guard reports
    guard_reports = run_report.get("guard_reports")

    if isinstance(guard_reports, dict):
        invariant_reports = [
            guard_report
            for guard_name, guard_report in guard_reports.items()
            if isinstance(guard_name, str) and "invariants" in guard_name.lower()
        ]
        if invariant_reports:
            return all(
                type(guard_report) is dict and guard_report.get("passed") is True
                for guard_report in invariant_reports
            )

    # Look for validation results in metrics
    metrics = run_report.get("metrics")
    if isinstance(metrics, dict) and "invariants_passed" in metrics:
        return metrics["invariants_passed"] is True

    # No invariants check found
    return False


def load_baseline(baseline_path: Path) -> dict[str, Any]:
    """Load baseline metrics from JSON file."""
    try:
        with open(baseline_path) as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(
                    f"Baseline file must contain a JSON object, got {type(data)}"
                )
            return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in baseline file: {e}") from e


def save_baseline(baseline: dict[str, Any], baseline_path: Path) -> None:
    """Save baseline metrics to JSON file."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2, allow_nan=False)


def create_baseline_from_report(run_report: dict[str, Any]) -> dict[str, Any]:
    """Create a baseline structure from a run report."""
    baseline: dict[str, Any] = {}

    # Extract core metrics (PM-only)
    try:
        pm = (
            run_report.get("metrics", {}).get("primary_metric")
            if isinstance(run_report.get("metrics"), dict)
            else None
        )
        if isinstance(pm, dict):
            kind = str(pm.get("kind") or "").strip().lower()
            field = (
                "delta_vs_baseline_pp" if kind == "accuracy" else "ratio_vs_baseline"
            )
            if pm.get(field) is not None:
                baseline[field] = float(pm[field])
    except _VALIDATION_EXCEPTIONS:
        pass

    if "param_reduction_ratio" in run_report:
        baseline["param_reduction_ratio"] = run_report["param_reduction_ratio"]
    elif "parameters_removed" in run_report and "original_params" in run_report:
        baseline["param_reduction_ratio"] = (
            run_report["parameters_removed"] / run_report["original_params"]
        )

    # Extract structural counts
    metrics = run_report.get("metrics", {})
    for key in ["heads_pruned", "neurons_pruned", "layers_modified"]:
        if key in run_report:
            baseline[key] = run_report[key]
        elif key in metrics:
            baseline[key] = metrics[key]

    # Extract sparsity metrics
    sparsity = run_report.get("actual_sparsity", {})
    for key in ["head_sparsity", "neuron_sparsity", "weight_sparsity"]:
        if key in sparsity:
            baseline[key] = sparsity[key]

    # Add metadata
    baseline["baseline_created"] = True
    baseline["source"] = "run_report"

    return baseline
