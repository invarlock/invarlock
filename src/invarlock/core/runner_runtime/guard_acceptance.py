from __future__ import annotations

import math
from typing import Any

from invarlock.guards.authority import guard_is_enforced


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _numeric_map(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or not value:
        return None
    result: dict[str, float] = {}
    for key, raw in value.items():
        numeric = _finite_number(raw)
        if not isinstance(key, str) or not key or numeric is None:
            return None
        result[key] = numeric
    return result


def _same_number(left: Any, right: Any) -> bool:
    left_number = _finite_number(left)
    right_number = _finite_number(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-12)
    )


def _spectral_z_finding_complete(
    finding: dict[str, Any],
    *,
    module: str,
    family: str,
    policy_caps: dict[str, Any],
    selected_families: list[Any],
    family_pvalues: dict[str, Any],
    final_z_scores: dict[str, float],
    final_metrics: dict[str, float],
    baseline: dict[str, float],
) -> float | None:
    z_score = _finite_number(finding.get("z_score"))
    kappa = _finite_number(finding.get("kappa"))
    p_value = _finite_number(finding.get("p_value"))
    expected_p_value = (
        math.erfc(abs(z_score) / math.sqrt(2.0)) if z_score is not None else None
    )
    family_policy = policy_caps.get(family) or policy_caps.get("other")
    if (
        z_score is None
        or kappa is None
        or kappa < 0.0
        or abs(z_score) <= kappa
        or p_value is None
        or not 0.0 <= p_value <= 1.0
        or expected_p_value is None
        or not math.isclose(p_value, expected_p_value, rel_tol=1e-9, abs_tol=1e-12)
        or family not in selected_families
        or _finite_number(family_pvalues.get(family)) is None
        or not isinstance(family_policy, dict)
        or not _same_number(family_policy.get("kappa"), kappa)
        or not _same_number(final_z_scores[module], z_score)
        or not _same_number(final_metrics[module], finding.get("sigma"))
        or not _same_number(baseline[module], finding.get("baseline_sigma"))
    ):
        return None
    return p_value


def _spectral_degeneracy_finding_complete(
    finding: dict[str, Any],
    *,
    finding_type: str,
    module: str,
    baseline_degeneracy: Any,
    final_degeneracy: Any,
) -> bool:
    metric = (
        "stable_rank"
        if finding_type == "degeneracy_stable_rank_drop"
        else "norm_collapse"
    )
    base_value = _finite_number(finding.get(f"{metric}_base"))
    current_value = _finite_number(finding.get(f"{metric}_cur"))
    ratio = _finite_number(finding.get("ratio"))
    warn_ratio = _finite_number(finding.get("warn_ratio"))
    fatal_ratio = _finite_number(finding.get("fatal_ratio"))
    if (
        base_value is None
        or base_value <= 0.0
        or current_value is None
        or ratio is None
        or warn_ratio is None
        or fatal_ratio is None
        or not fatal_ratio <= ratio < warn_ratio
        or not math.isclose(
            ratio,
            current_value / base_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or not isinstance(baseline_degeneracy, dict)
        or not isinstance(baseline_degeneracy.get(module), dict)
        or not _same_number(baseline_degeneracy[module].get(metric), base_value)
        or not isinstance(final_degeneracy, dict)
        or not isinstance(final_degeneracy.get(module), dict)
        or not _same_number(final_degeneracy[module].get(metric), current_value)
    ):
        return False
    return True


def _spectral_findings_complete(
    violations: list[Any],
    *,
    modules: set[str],
    module_families: dict[Any, Any],
    policy_caps: dict[str, Any],
    selected_families: list[Any],
    family_pvalues: dict[str, Any],
    final_z_scores: dict[str, float],
    final_metrics: dict[str, float],
    baseline: dict[str, float],
    baseline_degeneracy: Any,
    final_degeneracy: Any,
) -> bool:
    observed_pvalues: dict[str, list[float]] = {}
    degeneracy_types = {
        "degeneracy_stable_rank_drop",
        "degeneracy_norm_collapse",
    }
    for finding in violations:
        if not isinstance(finding, dict) or finding.get("severity") != "budgeted":
            return False
        finding_type = finding.get("type")
        module = finding.get("module")
        family = finding.get("family")
        if (
            not isinstance(module, str)
            or module not in modules
            or not isinstance(family, str)
            or module_families.get(module) != family
            or finding.get("selected") is not True
        ):
            return False
        if finding_type == "family_z_cap":
            p_value = _spectral_z_finding_complete(
                finding,
                module=module,
                family=family,
                policy_caps=policy_caps,
                selected_families=selected_families,
                family_pvalues=family_pvalues,
                final_z_scores=final_z_scores,
                final_metrics=final_metrics,
                baseline=baseline,
            )
            if p_value is None:
                return False
            observed_pvalues.setdefault(family, []).append(p_value)
        elif finding_type in degeneracy_types:
            if not _spectral_degeneracy_finding_complete(
                finding,
                finding_type=str(finding_type),
                module=module,
                baseline_degeneracy=baseline_degeneracy,
                final_degeneracy=final_degeneracy,
            ):
                return False
        else:
            return False
    return not any(
        not _same_number(family_pvalues.get(family), min(pvalues))
        for family, pvalues in observed_pvalues.items()
    )


def _complete_observed_spectral_finding(result: dict[str, Any]) -> bool:
    """Require the raw facts needed to replay a non-fatal spectral finding."""

    metrics = result.get("metrics")
    policy = result.get("policy")
    violations = result.get("violations")
    baseline_metrics = result.get("baseline_metrics")
    final_metrics = _numeric_map(result.get("final_metrics"))
    final_z_scores = _numeric_map(result.get("final_z_scores"))
    module_families = result.get("module_family_map")
    if (
        not isinstance(metrics, dict)
        or not metrics
        or not isinstance(policy, dict)
        or not policy
        or not isinstance(violations, list)
        or not violations
        or not isinstance(baseline_metrics, dict)
        or not baseline_metrics
        or final_metrics is None
        or final_z_scores is None
        or not isinstance(module_families, dict)
        or not module_families
        or not isinstance(result.get("measurement_inventory"), dict)
        or not result["measurement_inventory"]
        or not isinstance(result.get("correction_ledger"), dict)
        or not result["correction_ledger"]
    ):
        return False
    baseline = _numeric_map(baseline_metrics.get("module_sigmas"))
    modules = set(final_metrics)
    if (
        baseline is None
        or set(baseline) != modules
        or set(final_z_scores) != modules
        or set(module_families) != modules
        or any(
            not isinstance(family, str) or not family
            for family in module_families.values()
        )
    ):
        return False
    modules_checked = _nonnegative_int(metrics.get("modules_checked"))
    caps_applied = _nonnegative_int(metrics.get("caps_applied"))
    max_caps = _nonnegative_int(metrics.get("max_caps"))
    fatal_violations = _nonnegative_int(metrics.get("fatal_violations"))
    selected_findings = _nonnegative_int(metrics.get("selected_budgeted_findings"))
    if (
        modules_checked != len(modules)
        or caps_applied is None
        or max_caps is None
        or caps_applied <= max_caps
        or caps_applied != len(violations)
        or selected_findings != caps_applied
        or fatal_violations != 0
        or metrics.get("caps_exceeded") is not True
        or metrics.get("cap_budget_exceeded") is not True
        or policy.get("max_caps") != max_caps
        or not isinstance(metrics.get("measurement_contract"), dict)
        or not metrics["measurement_contract"]
        or any(
            metrics.get(field) not in ([], None)
            for field in (
                "identity_changed_modules",
                "measurement_exclusions",
                "discovery_errors",
            )
        )
    ):
        return False
    policy_caps = policy.get("family_caps")
    selection = metrics.get("multiple_testing_selection")
    if (
        not isinstance(policy_caps, dict)
        or not isinstance(selection, dict)
        or metrics.get("family_caps") != policy_caps
        or metrics.get("multiple_testing") != policy.get("multiple_testing")
    ):
        return False
    selected_families = selection.get("families_selected")
    family_pvalues = selection.get("family_pvalues")
    if not isinstance(selected_families, list) or not isinstance(family_pvalues, dict):
        return False
    return _spectral_findings_complete(
        violations,
        modules=modules,
        module_families=module_families,
        policy_caps=policy_caps,
        selected_families=selected_families,
        family_pvalues=family_pvalues,
        final_z_scores=final_z_scores,
        final_metrics=final_metrics,
        baseline=baseline,
        baseline_degeneracy=baseline_metrics.get("baseline_degeneracy"),
        final_degeneracy=result.get("final_degeneracy"),
    )


def _replay_rmt_family_maps(
    baseline_modules: dict[str, float],
    current_modules: dict[str, float],
    module_families: dict[Any, Any],
) -> tuple[dict[str, float], dict[str, float]] | None:
    replay_baseline: dict[str, float] = {}
    replay_current: dict[str, float] = {}
    for module, family in module_families.items():
        if not isinstance(module, str) or not isinstance(family, str) or not family:
            return None
        replay_baseline[family] = max(
            replay_baseline.get(family, 0.0), baseline_modules[module]
        )
        replay_current[family] = max(
            replay_current.get(family, 0.0), current_modules[module]
        )
    return replay_baseline, replay_current


def _expected_rmt_violation_families(
    baseline_families: dict[str, float],
    current_families: dict[str, float],
    epsilon_by_family: dict[Any, Any],
    policy_epsilons: dict[Any, Any],
    epsilon_default: float,
) -> set[str] | None:
    expected: set[str] = set()
    for family, base in baseline_families.items():
        epsilon = _finite_number(epsilon_by_family.get(family, epsilon_default))
        policy_epsilon = _finite_number(policy_epsilons.get(family, epsilon_default))
        if (
            epsilon is None
            or epsilon < 0.0
            or policy_epsilon is None
            or not math.isclose(epsilon, policy_epsilon, rel_tol=1e-9, abs_tol=1e-12)
        ):
            return None
        if base > 0.0 and current_families[family] > (1.0 + epsilon) * base:
            expected.add(family)
    return expected


def _rmt_finding_family(
    finding: dict[str, Any],
    metric_finding: dict[str, Any],
    *,
    baseline_families: dict[str, float],
    current_families: dict[str, float],
    epsilon_by_family: dict[Any, Any],
    policy_epsilons: dict[Any, Any],
    epsilon_default: float,
    observed_families: set[str],
) -> str | None:
    family = finding.get("family")
    finding_base = _finite_number(finding.get("edge_base"))
    finding_current = _finite_number(finding.get("edge_cur"))
    allowed = _finite_number(finding.get("allowed"))
    epsilon = _finite_number(finding.get("epsilon"))
    delta = _finite_number(finding.get("delta"))
    if (
        finding.get("type") != "epsilon_band"
        or finding.get("severity") != "error"
        or not isinstance(family, str)
        or family in observed_families
        or family not in baseline_families
        or finding_base is None
        or finding_base <= 0.0
        or finding_current is None
        or allowed is None
        or epsilon is None
        or epsilon < 0.0
        or delta is None
        or finding_current <= allowed
        or not _same_number(finding_base, baseline_families[family])
        or not _same_number(finding_current, current_families[family])
        or not math.isclose(
            allowed,
            (1.0 + epsilon) * finding_base,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or not math.isclose(
            delta,
            (finding_current / finding_base) - 1.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or not _same_number(epsilon_by_family.get(family, epsilon_default), epsilon)
        or not _same_number(policy_epsilons.get(family, epsilon_default), epsilon)
        or any(
            not _same_number(metric_finding.get(field), finding.get(field))
            for field in ("edge_base", "edge_cur", "allowed", "epsilon", "delta")
        )
        or metric_finding.get("family") != family
    ):
        return None
    return family


def _rmt_findings_complete(
    violations: list[Any],
    metric_violations: list[Any],
    *,
    baseline_families: dict[str, float],
    current_families: dict[str, float],
    epsilon_by_family: dict[Any, Any],
    policy_epsilons: dict[Any, Any],
    epsilon_default: float,
    expected_families: set[str],
) -> bool:
    observed_families: set[str] = set()
    for finding, metric_finding in zip(violations, metric_violations, strict=True):
        if not isinstance(finding, dict) or not isinstance(metric_finding, dict):
            return False
        family = _rmt_finding_family(
            finding,
            metric_finding,
            baseline_families=baseline_families,
            current_families=current_families,
            epsilon_by_family=epsilon_by_family,
            policy_epsilons=policy_epsilons,
            epsilon_default=epsilon_default,
            observed_families=observed_families,
        )
        if family is None:
            return False
        observed_families.add(family)
    return bool(observed_families) and observed_families == expected_families


def _complete_observed_rmt_finding(result: dict[str, Any]) -> bool:
    """Require an exact, replayable epsilon-band finding."""

    metrics = result.get("metrics")
    policy = result.get("policy")
    violations = result.get("violations")
    if (
        not isinstance(metrics, dict)
        or not metrics
        or not isinstance(policy, dict)
        or not policy
        or not isinstance(violations, list)
        or not violations
        or metrics.get("prepared") is not True
        or metrics.get("stable") is not False
        or not isinstance(metrics.get("measurement_contract"), dict)
        or not metrics["measurement_contract"]
    ):
        return False
    baseline_families = _numeric_map(metrics.get("edge_risk_by_family_base"))
    current_families = _numeric_map(metrics.get("edge_risk_by_family"))
    baseline_modules = _numeric_map(metrics.get("edge_risk_by_module_base"))
    current_modules = _numeric_map(metrics.get("edge_risk_by_module"))
    module_families = metrics.get("module_family_map")
    epsilon_by_family = metrics.get("epsilon_by_family")
    metric_violations = metrics.get("epsilon_violations")
    if (
        baseline_families is None
        or current_families is None
        or set(baseline_families) != set(current_families)
        or baseline_modules is None
        or current_modules is None
        or set(baseline_modules) != set(current_modules)
        or not isinstance(module_families, dict)
        or set(module_families) != set(baseline_modules)
        or not isinstance(epsilon_by_family, dict)
        or not isinstance(metric_violations, list)
        or len(metric_violations) != len(violations)
    ):
        return False
    replayed = _replay_rmt_family_maps(
        baseline_modules,
        current_modules,
        module_families,
    )
    if replayed is None:
        return False
    replay_baseline, replay_current = replayed
    if set(replay_baseline) != set(baseline_families) or any(
        not _same_number(replay_baseline[family], baseline_families[family])
        or not _same_number(replay_current[family], current_families[family])
        for family in replay_baseline
    ):
        return False
    epsilon_default = _finite_number(policy.get("epsilon_default"))
    policy_epsilons = policy.get("epsilon_by_family")
    if (
        epsilon_default is None
        or epsilon_default < 0.0
        or not isinstance(policy_epsilons, dict)
    ):
        return False
    expected_families = _expected_rmt_violation_families(
        baseline_families,
        current_families,
        epsilon_by_family,
        policy_epsilons,
        epsilon_default,
    )
    return expected_families is not None and _rmt_findings_complete(
        violations,
        metric_violations,
        baseline_families=baseline_families,
        current_families=current_families,
        epsilon_by_family=epsilon_by_family,
        policy_epsilons=policy_epsilons,
        epsilon_default=epsilon_default,
        expected_families=expected_families,
    )


def guard_result_is_acceptable(
    name: str,
    result: Any,
    authority: dict[str, str],
) -> bool:
    """Apply acceptance authority without accepting degraded guard evidence."""

    if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
        return False
    base_name = "invariants" if name == "invariants_post" else name
    status = str(result.get("status") or "").strip().lower().replace("_", "-")
    if (
        result.get("supported") is False
        or result.get("assurance_blocking") is True
        or status in {"degraded", "error", "monitor-only", "unsupported"}
        or bool(result.get("errors"))
    ):
        return False
    metrics = result.get("metrics")
    if (
        base_name == "variance"
        and isinstance(metrics, dict)
        and metrics.get("monitor_only") is True
    ):
        return False
    if result["passed"] is True:
        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, list) and any(
            isinstance(item, dict)
            and str(item.get("severity") or "").strip().lower()
            in {"critical", "error", "fatal"}
            for item in diagnostics
        ):
            return False
        return True
    if guard_is_enforced(authority, base_name):
        return False
    decision = str(result.get("decision") or "").strip().lower()
    if decision not in {"block", "blocked", "deny", "fail", "failed", "reject"}:
        return False
    if base_name == "spectral" and not _complete_observed_spectral_finding(result):
        return False
    if base_name == "rmt" and not _complete_observed_rmt_finding(result):
        return False
    if base_name == "variance":
        if not isinstance(metrics, dict):
            return False
        predictive = metrics.get("predictive_gate")
        calibration = metrics.get("calibration")
        if not isinstance(predictive, dict) or not isinstance(calibration, dict):
            return False
        reason = str(predictive.get("reason") or "").strip().lower().replace("_", "-")
        if (
            predictive.get("evaluated") is not True
            or predictive.get("passed") is not False
            or reason
            not in {
                "ci-contains-zero",
                "gain-below-threshold",
                "mean-not-negative",
                "regression-detected",
            }
            or calibration.get("status") != "complete"
        ):
            return False
        coverage = calibration.get("coverage")
        minimum = calibration.get("min_coverage")
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, int)
            or isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or minimum <= 0
            or coverage < minimum
        ):
            return False
    return True


__all__ = ["guard_result_is_acceptable"]
