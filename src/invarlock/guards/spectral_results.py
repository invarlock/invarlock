from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def _quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    fraction = position - lower
    return sorted_values[lower] + (
        (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def compute_family_observability(
    latest_z_scores: dict[str, float],
    module_family_map: dict[str, str],
    *,
    top_k: int = 3,
) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, Any]]]]:
    family_scores: dict[str, list[float]] = defaultdict(list)
    family_modules: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for module_name, z_value in latest_z_scores.items():
        family = module_family_map.get(module_name)
        if family is None:
            continue
        try:
            z_abs = abs(float(z_value))
        except (TypeError, ValueError):
            continue
        family_scores.setdefault(family, []).append(z_abs)
        family_modules.setdefault(family, []).append((z_abs, module_name))

    family_quantiles: dict[str, dict[str, float]] = {}
    for family, scores in family_scores.items():
        sorted_scores = sorted(scores)
        family_quantiles[family] = {
            "q95": _quantile(sorted_scores, 0.95),
            "q99": _quantile(sorted_scores, 0.99),
            "max": sorted_scores[-1] if sorted_scores else 0.0,
            "count": len(sorted_scores),
        }

    top_z_scores: dict[str, list[dict[str, Any]]] = {}
    for family, module_entries in family_modules.items():
        module_entries.sort(key=lambda item: item[0], reverse=True)
        top_z_scores[family] = [
            {"module": module_name, "z": float(z_abs), "family": family}
            for z_abs, module_name in module_entries[:top_k]
        ]

    return family_quantiles, top_z_scores


def partition_spectral_violations(
    violations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fatal_violations = [
        violation
        for violation in violations
        if (violation.get("severity") == "fatal")
        or (violation.get("type") == "max_spectral_norm")
    ]
    budgeted_violations = [
        violation for violation in violations if violation not in fatal_violations
    ]
    return fatal_violations, budgeted_violations


def evaluate_spectral_outcome(
    *,
    fatal_violations: list[dict[str, Any]],
    budgeted_violations: list[dict[str, Any]],
    selected_budgeted: list[dict[str, Any]],
    max_caps: int,
) -> dict[str, Any]:
    caps_applied = len(selected_budgeted)
    caps_exceeded = caps_applied > int(max_caps)
    passed = not fatal_violations and not caps_exceeded
    if fatal_violations or caps_exceeded:
        decision = "block"
        action = "abort"
    elif caps_applied > 0:
        decision = "monitor"
        action = "warn"
    else:
        decision = "allow"
        action = "continue"
    return {
        "selected_violations": [*fatal_violations, *selected_budgeted],
        "candidate_budgeted": len(budgeted_violations),
        "caps_applied": caps_applied,
        "caps_exceeded": caps_exceeded,
        "passed": passed,
        "decision": decision,
        "action": action,
    }


def build_spectral_diagnostics(
    selected_violations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for violation in selected_violations:
        severity = "error" if (
            violation.get("severity") == "fatal"
            or violation.get("type") == "max_spectral_norm"
        ) else "warning"
        diagnostics.append(
            {
                "kind": str(violation.get("type", "spectral_violation")),
                "severity": severity,
                "message": str(violation.get("message", "")),
                "family": violation.get("family"),
                "module": violation.get("module"),
            }
        )
    return diagnostics


def build_spectral_validation_metrics(
    *,
    current_metrics: dict[str, float],
    candidate_violations: list[dict[str, Any]],
    selected_violations: list[dict[str, Any]],
    fatal_violations: list[dict[str, Any]],
    candidate_budgeted: int,
    caps_applied: int,
    caps_exceeded: bool,
    family_summary: dict[str, Any],
    family_caps: dict[str, Any],
    sigma_quantile: float,
    deadband: float,
    max_caps: int,
    multiple_testing: dict[str, Any],
    multiple_testing_selection: dict[str, Any],
    estimator: dict[str, Any],
    degeneracy: dict[str, Any],
    family_quantiles: dict[str, dict[str, float]],
    top_z_scores: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metrics = {
        "modules_checked": len(current_metrics),
        "violations_found": len(selected_violations),
        "budgeted_violations": caps_applied,
        "candidate_budgeted_violations": candidate_budgeted,
        "fatal_violations": len(fatal_violations),
        "max_spectral_norm": max(current_metrics.values()) if current_metrics else 0.0,
        "mean_spectral_norm": np.mean(list(current_metrics.values()))
        if current_metrics
        else 0.0,
        "stability_score": 1.0
        - min(len(candidate_violations) / max(len(current_metrics), 1), 1.0),
        "family_z_summary": family_summary,
        "family_caps": family_caps,
        "sigma_quantile": float(sigma_quantile),
        "deadband": float(deadband),
        "max_caps": int(max_caps),
        "caps_applied": caps_applied,
        "caps_exceeded": caps_exceeded,
        "multiple_testing": multiple_testing,
        "multiple_testing_selection": multiple_testing_selection,
        "measurement_contract": {
            "estimator": estimator,
            "degeneracy": degeneracy,
        },
    }
    if family_quantiles:
        metrics["family_z_quantiles"] = family_quantiles
    if top_z_scores:
        metrics["top_z_scores"] = top_z_scores
    return metrics


def build_spectral_finalize_metrics(
    *,
    final_metrics: dict[str, float],
    selected_violations: list[dict[str, Any]],
    candidate_violations: list[dict[str, Any]],
    fatal_violations: list[dict[str, Any]],
    candidate_budgeted: int,
    caps_applied: int,
    caps_exceeded: bool,
    baseline_metrics: dict[str, float],
    scope: str,
    correction_enabled: bool,
    family_caps: dict[str, Any],
    final_z_summary: dict[str, Any],
    final_family_stats: dict[str, Any],
    sigma_quantile: float,
    deadband: float,
    max_caps: int,
    multiple_testing: dict[str, Any],
    multiple_testing_selection: dict[str, Any],
    estimator: dict[str, Any],
    degeneracy: dict[str, Any],
    family_quantiles: dict[str, dict[str, float]],
    top_z_scores: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "modules_analyzed": len(final_metrics),
        "violations_detected": len(selected_violations),
        "budgeted_violations": caps_applied,
        "candidate_violations_detected": len(candidate_violations),
        "candidate_budgeted_violations": candidate_budgeted,
        "fatal_violations": len(fatal_violations),
        "baseline_modules": len(baseline_metrics),
        "scope": scope,
        "max_spectral_norm_final": max(final_metrics.values())
        if final_metrics
        else 0.0,
        "mean_spectral_norm_final": np.mean(list(final_metrics.values()))
        if final_metrics
        else 0.0,
        "spectral_stability_score": 1.0
        - min(len(candidate_violations) / max(len(final_metrics), 1), 1.0),
        "correction_applied": len(selected_violations) > 0 and correction_enabled,
        "family_caps": family_caps,
        "family_z_summary": final_z_summary,
        "family_stats": final_family_stats,
        "sigma_quantile": float(sigma_quantile),
        "deadband": float(deadband),
        "max_caps": int(max_caps),
        "caps_applied": caps_applied,
        "caps_exceeded": caps_exceeded,
        "multiple_testing": multiple_testing,
        "multiple_testing_selection": multiple_testing_selection,
        "family_z_quantiles": family_quantiles,
        "top_z_scores": top_z_scores,
        "measurement_contract": {
            "estimator": estimator,
            "degeneracy": degeneracy,
        },
    }


def spectral_validation_message(
    *,
    passed: bool,
    fatal_violations: list[dict[str, Any]],
    caps_applied: int,
    max_caps: int,
) -> str:
    if passed:
        return (
            "Spectral validation passed with "
            f"{len(fatal_violations) + caps_applied} violations "
            f"(caps_applied={caps_applied}, max_caps={max_caps})"
        )
    reason = (
        "fatal spectral violation detected"
        if fatal_violations
        else "cap budget exceeded"
    )
    return (
        f"Spectral validation failed: {reason} "
        f"(caps_applied={caps_applied}, max_caps={max_caps})"
    )


def categorize_spectral_messages(
    selected_violations: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    for violation in selected_violations:
        message = str(violation.get("message", ""))
        if (violation.get("severity") == "fatal") or (
            violation.get("type") == "max_spectral_norm"
        ):
            errors.append(message)
        else:
            warnings.append(message)
    return warnings, errors


__all__ = [
    "build_spectral_finalize_metrics",
    "build_spectral_diagnostics",
    "build_spectral_validation_metrics",
    "categorize_spectral_messages",
    "compute_family_observability",
    "evaluate_spectral_outcome",
    "partition_spectral_violations",
    "spectral_validation_message",
]
