"""
InvarLock Validation Framework
=========================

Validation utilities for checking pruning results against baseline metrics.
Supports both automated CI testing and flexible user validation.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, cast

__all__ = [
    "validate_against_baseline",
    "validate_drift_gate",
    "validate_guard_overhead",
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
        metrics: dict[str, float],
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
    **kwargs,
) -> ValidationResult:
    # Backward-compatible kwargs (deprecated): enable via INVARLOCK_VALIDATE_LEGACY=1
    legacy = str(os.environ.get("INVARLOCK_VALIDATE_LEGACY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if legacy:
        if "tol_ppl_ratio" in kwargs and isinstance(
            kwargs["tol_ppl_ratio"], int | float
        ):
            tol_ratio = float(kwargs["tol_ppl_ratio"])
        if "ppl_bounds" in kwargs and isinstance(kwargs["ppl_bounds"], tuple):
            # Coerce after runtime guard
            ratio_bounds = cast(tuple[float, float], kwargs["ppl_bounds"])
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
        # Extract primary metric ratio (canonical)
        current_ratio = None
        pm_kind = None
        pm = (
            (run_report.get("metrics") or {}).get("primary_metric")
            if isinstance(run_report.get("metrics"), dict)
            else None
        )
        if isinstance(pm, dict) and pm:
            val = pm.get("ratio_vs_baseline")
            if isinstance(val, int | float):
                current_ratio = float(val)
            try:
                pm_kind = str(pm.get("kind") or "").lower()
            except Exception:
                pm_kind = None
        if current_ratio is None:
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

        if baseline_ratio is None:
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
        if current_ratio is not None:
            if pm_kind in {"accuracy", "vqa_accuracy"}:
                # Interpret current_ratio as delta proportion; compare in pp when bounds provided
                if isinstance(delta_bounds_pp, tuple) and len(delta_bounds_pp) == 2:
                    delta_pp = 100.0 * float(current_ratio)
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
            if pm_kind in {"accuracy", "vqa_accuracy"}:
                checks["delta_bounds_pp"] = False
            else:
                checks["ratio_bounds"] = False

        # Structural count validation
        if structural_exact:
            structural_checks = _validate_structural_counts(run_report, baseline)
            checks.update(structural_checks["checks"])
            messages.extend(structural_checks["messages"])
            warnings_list.extend(structural_checks["warnings"])
        else:
            checks["structural_counts"] = True  # Skip structural validation

        # Invariants validation (if present in report)
        invariants_passed = _validate_invariants(run_report)
        if invariants_passed is not None:
            checks["invariants"] = invariants_passed
            if not invariants_passed:
                errors.append("Model invariants validation failed")

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

    except Exception as e:
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

    except Exception as e:
        return ValidationResult(
            passed=False,
            checks={"drift_gate_error": False},
            metrics={},
            messages=[],
            warnings=[],
            errors=[f"Drift gate validation failed: {str(e)}"],
        )


def validate_guard_overhead(
    bare_report: dict[str, Any],
    guarded_report: dict[str, Any],
    overhead_threshold: float = 0.01,
) -> ValidationResult:
    """
    Validate guard overhead using primary_metric: final(guarded)/final(bare) ≤ 1%.

    Args:
        bare_report: Report from bare (no guards) run (expects metrics.primary_metric)
        guarded_report: Report from guarded run (expects metrics.primary_metric)
        overhead_threshold: Maximum allowed overhead (default 0.01 = 1%)

    Returns:
        ValidationResult with guard overhead check
    """
    checks = {}
    metrics = {}
    messages = []
    warnings: list[str] = []
    errors = []

    try:
        # Extract primary metric final from both reports
        bare_pm = (
            (bare_report.get("metrics") or {}).get("primary_metric")
            if isinstance(bare_report.get("metrics"), dict)
            else None
        )
        guarded_pm = (
            (guarded_report.get("metrics") or {}).get("primary_metric")
            if isinstance(guarded_report.get("metrics"), dict)
            else None
        )

        bare_final = None
        guarded_final = None
        if isinstance(bare_pm, dict):
            bare_final = bare_pm.get("final")
        if isinstance(guarded_pm, dict):
            guarded_final = guarded_pm.get("final")

        if (
            isinstance(bare_final, (int | float))
            and bare_final > 0
            and isinstance(guarded_final, (int | float))
        ):
            overhead_ratio = float(guarded_final) / float(bare_final)
            overhead_percent = (overhead_ratio - 1.0) * 100

            metrics["overhead_ratio"] = overhead_ratio
            metrics["overhead_percent"] = overhead_percent
            metrics["bare_final"] = float(bare_final)
            metrics["guarded_final"] = float(guarded_final)

            # Apply overhead gate
            checks["guard_overhead"] = overhead_ratio <= (1.0 + overhead_threshold)

            if checks["guard_overhead"]:
                messages.append(
                    f"Guard overhead PASSED: {overhead_percent:+.2f}% ≤ {overhead_threshold * 100:.1f}%"
                )
            else:
                errors.append(
                    f"Guard overhead FAILED: {overhead_percent:+.2f}% > {overhead_threshold * 100:.1f}% "
                    f"(guards add too much primary-metric overhead)"
                )
        else:
            errors.append(
                "Cannot calculate guard overhead: missing primary_metric data"
            )
            checks["guard_overhead"] = False

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

    except Exception as e:
        return ValidationResult(
            passed=False,
            checks={"guard_overhead_error": False},
            metrics={},
            messages=[],
            warnings=[],
            errors=[f"Guard overhead validation failed: {str(e)}"],
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
        checks["layers_count_exact"] = True  # Don't fail on missing data

    return {"checks": checks, "messages": messages, "warnings": warnings}


def _validate_invariants(run_report: dict[str, Any]) -> bool | None:
    """Check if model invariants passed."""
    # Look for invariants check in guard reports
    guard_reports = run_report.get("guard_reports", {})

    for guard_name, guard_report in guard_reports.items():
        if "invariants" in guard_name.lower():
            passed = guard_report.get("passed", True)
            return bool(passed) if passed is not None else True

    # Look for validation results in metrics
    metrics = run_report.get("metrics", {})
    if "invariants_passed" in metrics:
        passed = metrics["invariants_passed"]
        return bool(passed) if passed is not None else None

    # No invariants check found
    return None


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
        json.dump(baseline, f, indent=2)


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
        if isinstance(pm, dict) and pm.get("ratio_vs_baseline") is not None:
            baseline["ratio_vs_baseline"] = float(pm["ratio_vs_baseline"])
    except Exception:
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


def validate_gpt2_small_wt2_baseline(
    run_report: dict[str, Any], baseline_path: Path | None = None
) -> ValidationResult:
    """
    Validate against the canonical GPT-2 small + WikiText-2 baseline.

    This is the CI validation function that uses the pinned baseline.
    """
    if baseline_path is None:
        # Use default baseline path
        baseline_path = (
            Path(__file__).parent.parent.parent
            / "benchmarks"
            / "baselines"
            / "gpt2_small_wt2.json"
        )

    try:
        baseline = load_baseline(baseline_path)
    except FileNotFoundError:
        # Create a default baseline if file doesn't exist
        warnings.warn(
            f"Baseline file not found: {baseline_path}. Using default values.",
            stacklevel=2,
        )
        baseline = {
            "ratio_vs_baseline": 1.285,  # Target: ~1.25-1.32
            "param_reduction_ratio": 0.022,  # Target: ~2.2%
            "heads_pruned": 16,  # Example values
            "neurons_pruned": 1024,
            "layers_modified": 8,
            "head_sparsity": 0.1,
            "neuron_sparsity": 0.1,
        }

    return validate_against_baseline(
        run_report,
        baseline,
        tol_ratio=0.02,
        tol_param_ratio=0.02,
        ratio_bounds=(1.25, 1.32),
        structural_exact=True,
    )
