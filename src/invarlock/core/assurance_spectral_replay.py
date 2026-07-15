"""Independent strict replay of retained SpectralGuard observations.

This public orchestrator intentionally depends only on verifier-owned modules.
It does not import guard-side spectral decision helpers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .assurance_spectral_replay_common import (
    _close,
    _compare_tree,
    _compare_violations,
    _degeneracy_map,
    _family_map,
    _family_stats,
    _finite,
    _mapping,
    _nonnegative_int,
    _numeric_map,
    _policy_number,
    _reject_families,
)
from .assurance_spectral_replay_correction import (
    replay_correction_ledger as _replay_correction_ledger,
)
from .assurance_spectral_replay_inventory import (
    replay_measurement_inventory as _replay_measurement_inventory,
)


@dataclass(frozen=True)
class _ReplayPolicy:
    deadband: float
    max_caps: int
    max_norm: float | None
    family_caps: dict[str, float]
    caps_raw: Mapping[str, Any]
    multiple_testing: Mapping[str, Any]
    method: str
    alpha: float
    configured_m: int
    degeneracy_enabled: bool
    thresholds: dict[str, tuple[float, float]]
    correction_enabled: bool
    correction_cap_ratio: float


def _validated_policy(
    errors: list[str],
    policy: Mapping[str, Any],
    families: Mapping[str, str],
    source: str,
) -> _ReplayPolicy | None:
    initial_error_count = len(errors)
    deadband = _policy_number(
        errors, policy, "deadband", f"{source}.policy", minimum=0.0
    )
    max_caps = _nonnegative_int(policy.get("max_caps"))
    if max_caps is None:
        errors.append(f"{source}.policy.max_caps must be a non-negative integer.")
    max_norm_raw = policy.get("max_spectral_norm")
    max_norm = None
    if max_norm_raw is not None:
        max_norm = _finite(max_norm_raw)
        if max_norm is None or max_norm < 0.0:
            errors.append(
                f"{source}.policy.max_spectral_norm must be null or a finite non-negative number."
            )

    caps_raw = _mapping(policy.get("family_caps"))
    family_caps: dict[str, float] = {}
    if caps_raw is None or not caps_raw:
        errors.append(f"{source}.policy.family_caps must be a non-empty object.")
    else:
        for family in sorted(set(families.values())):
            cap_entry = _mapping(caps_raw.get(family)) or _mapping(
                caps_raw.get("other")
            )
            cap = _finite(cap_entry.get("kappa")) if cap_entry is not None else None
            if cap is None or cap < 0.0:
                errors.append(
                    f"{source}.policy.family_caps lacks a finite non-negative kappa for {family!r}."
                )
            else:
                family_caps[family] = cap

    multiple_testing = _mapping(policy.get("multiple_testing"))
    method = (
        str(multiple_testing.get("method") or "").strip().lower()
        if multiple_testing is not None
        else ""
    )
    alpha = (
        _finite(multiple_testing.get("alpha")) if multiple_testing is not None else None
    )
    configured_m = (
        _nonnegative_int(multiple_testing.get("m"))
        if multiple_testing is not None
        else None
    )
    if method not in {"bh", "bonferroni"}:
        errors.append(
            f"{source}.policy.multiple_testing.method must be 'bh' or 'bonferroni'."
        )
    if alpha is None or not 0.0 < alpha <= 1.0:
        errors.append(
            f"{source}.policy.multiple_testing.alpha must be in the interval (0, 1]."
        )
    if configured_m is None or configured_m < 1:
        errors.append(f"{source}.policy.multiple_testing.m must be a positive integer.")

    degeneracy = _mapping(policy.get("degeneracy"))
    enabled = degeneracy.get("enabled") if degeneracy is not None else None
    if not isinstance(enabled, bool):
        errors.append(f"{source}.policy.degeneracy.enabled must be a boolean.")
    thresholds: dict[str, tuple[float, float]] = {}
    if degeneracy is not None:
        for metric in ("stable_rank", "norm_collapse"):
            config = _mapping(degeneracy.get(metric))
            warn = _finite(config.get("warn_ratio")) if config is not None else None
            fatal = _finite(config.get("fatal_ratio")) if config is not None else None
            if warn is None or fatal is None or warn < 0.0 or fatal < 0.0:
                errors.append(
                    f"{source}.policy.degeneracy.{metric} thresholds must be finite and non-negative."
                )
            else:
                thresholds[metric] = (warn, fatal)

    correction_enabled = policy.get("correction_enabled")
    if not isinstance(correction_enabled, bool):
        errors.append(f"{source}.policy.correction_enabled must be a boolean.")
    correction_cap_ratio = _finite(policy.get("correction_cap_ratio"))
    if correction_cap_ratio is None or correction_cap_ratio <= 0.0:
        errors.append(
            f"{source}.policy.correction_cap_ratio must be finite and greater than zero."
        )
    if len(errors) != initial_error_count:
        return None
    assert deadband is not None
    assert max_caps is not None
    assert caps_raw is not None
    assert multiple_testing is not None
    assert alpha is not None
    assert configured_m is not None
    assert isinstance(enabled, bool)
    assert isinstance(correction_enabled, bool)
    assert correction_cap_ratio is not None
    return _ReplayPolicy(
        deadband=deadband,
        max_caps=max_caps,
        max_norm=max_norm,
        family_caps=family_caps,
        caps_raw=caps_raw,
        multiple_testing=multiple_testing,
        method=method,
        alpha=alpha,
        configured_m=configured_m,
        degeneracy_enabled=enabled,
        thresholds=thresholds,
        correction_enabled=correction_enabled,
        correction_cap_ratio=correction_cap_ratio,
    )


def _replayed_z_scores(
    errors: list[str],
    *,
    source: str,
    modules: set[str],
    baseline: Mapping[str, float],
    final: Mapping[str, float],
    z_scores: Mapping[str, float],
    families: Mapping[str, str],
    family_stats: Mapping[str, Mapping[str, float | int]],
    deadband: float,
) -> dict[str, float]:
    replayed_z: dict[str, float] = {}
    for module in sorted(modules):
        stats = family_stats[families[module]]
        std = float(stats["std"])
        if std > 0.0:
            z_value = (final[module] - float(stats["mean"])) / std
        else:
            denominator = baseline[module] if baseline[module] > 0.0 else 1.0
            relative_change = (final[module] / denominator) - 1.0
            if abs(relative_change) <= deadband:
                z_value = 0.0
            else:
                z_value = relative_change / (deadband if deadband > 0.0 else 1.0)
        replayed_z[module] = z_value
        if not _close(z_scores[module], z_value):
            errors.append(
                f"{source}.final_z_scores.{module} disagrees with baseline/final measurements."
            )
    return replayed_z


def _initial_violations(
    *,
    modules: set[str],
    baseline: Mapping[str, float],
    final: Mapping[str, float],
    replayed_z: Mapping[str, float],
    families: Mapping[str, str],
    family_caps: Mapping[str, float],
    max_norm: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    budgeted: list[dict[str, Any]] = []
    fatal: list[dict[str, Any]] = []
    for module in sorted(modules):
        family = families[module]
        cap = family_caps[family]
        if abs(replayed_z[module]) > cap:
            budgeted.append(
                {
                    "type": "family_z_cap",
                    "severity": "budgeted",
                    "module": module,
                    "family": family,
                    "z_score": replayed_z[module],
                    "kappa": cap,
                    "sigma": final[module],
                    "baseline_sigma": baseline[module],
                }
            )
        if max_norm is not None and final[module] > max_norm:
            fatal.append(
                {
                    "type": "max_spectral_norm",
                    "severity": "fatal",
                    "module": module,
                    "family": family,
                    "current_sigma": final[module],
                    "threshold": max_norm,
                }
            )
    return budgeted, fatal


def _append_degeneracy_violations(
    budgeted: list[dict[str, Any]],
    fatal: list[dict[str, Any]],
    *,
    modules: set[str],
    families: Mapping[str, str],
    baseline_degeneracy: Mapping[str, Mapping[str, float]],
    final_degeneracy: Mapping[str, Mapping[str, float]],
    thresholds: Mapping[str, tuple[float, float]],
) -> None:
    type_by_metric = {
        "stable_rank": "degeneracy_stable_rank_drop",
        "norm_collapse": "degeneracy_norm_collapse",
    }
    field_by_metric = {
        "stable_rank": ("stable_rank_base", "stable_rank_cur"),
        "norm_collapse": ("norm_collapse_base", "norm_collapse_cur"),
    }
    for module in sorted(modules):
        for metric in ("stable_rank", "norm_collapse"):
            base_value = baseline_degeneracy[module][metric]
            current_value = final_degeneracy[module][metric]
            if base_value <= 0.0:
                continue
            ratio = current_value / max(base_value, 1e-12)
            warn_ratio, fatal_ratio = thresholds[metric]
            if math.isfinite(ratio) and ratio < warn_ratio:
                base_field, current_field = field_by_metric[metric]
                finding = {
                    "type": type_by_metric[metric],
                    "severity": "fatal" if ratio < fatal_ratio else "budgeted",
                    "module": module,
                    "family": families[module],
                    base_field: base_value,
                    current_field: current_value,
                    "ratio": ratio,
                    "warn_ratio": warn_ratio,
                    "fatal_ratio": fatal_ratio,
                }
                if finding["severity"] == "fatal":
                    fatal.append(finding)
                else:
                    budgeted.append(finding)


def _selected_budgeted_violations(
    candidate_budgeted: list[dict[str, Any]],
    *,
    method: str,
    alpha: float,
    configured_m: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_pvalues: dict[str, float] = {}
    family_max_abs_z: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    for finding in candidate_budgeted:
        if finding["type"] != "family_z_cap":
            continue
        family = str(finding["family"])
        z_value = float(finding["z_score"])
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
        family_counts[family] = family_counts.get(family, 0) + 1
        if family not in family_pvalues or p_value < family_pvalues[family]:
            family_pvalues[family] = p_value
            family_max_abs_z[family] = abs(z_value)
    families_tested = sorted(family_pvalues)
    m_effective = max(configured_m, len(families_tested), 1)
    selected_families = _reject_families(
        family_pvalues, method=method, alpha=alpha, m=m_effective
    )
    selected_budgeted: list[dict[str, Any]] = []
    default_selected = 0
    for finding in candidate_budgeted:
        item: dict[str, Any] = dict(finding)
        if item["type"] == "family_z_cap":
            item["p_value"] = math.erfc(abs(float(item["z_score"])) / math.sqrt(2.0))
            item["selected"] = item["family"] in selected_families
        else:
            item["p_value"] = None
            item["selected"] = True
            default_selected += 1
        if item["selected"]:
            selected_budgeted.append(item)
    selection = {
        "method": method,
        "alpha": alpha,
        "m": m_effective,
        "families_tested": families_tested,
        "families_selected": sorted(selected_families),
        "family_pvalues": {
            family: family_pvalues[family] for family in families_tested
        },
        "family_max_abs_z": {
            family: family_max_abs_z[family] for family in families_tested
        },
        "family_violation_counts": family_counts,
        "default_selected_without_pvalue": default_selected,
    }
    return selected_budgeted, selection


def _validate_entry_metrics(
    errors: list[str],
    *,
    entry: Mapping[str, Any],
    metrics: Mapping[str, Any],
    source: str,
    modules: set[str],
    candidate_budgeted: list[dict[str, Any]],
    fatal_violations: list[dict[str, Any]],
    selected_violations: list[dict[str, Any]],
    caps_applied: int,
    max_caps: int,
    caps_exceeded: bool,
    final: Mapping[str, float],
    expected_pass: bool,
    expected_decision: str,
) -> None:
    count_expectations = {
        "baseline_modules": len(modules),
        "candidate_budgeted_violations": len(candidate_budgeted),
        "fatal_violations": len(fatal_violations),
        "budgeted_violations": caps_applied,
        "caps_applied": caps_applied,
        "max_caps": max_caps,
    }
    for field, expected in count_expectations.items():
        if _nonnegative_int(metrics.get(field)) != expected:
            errors.append(f"{source}.metrics.{field} disagrees with replayed evidence.")
    module_fields = [
        field for field in ("modules_checked", "modules_analyzed") if field in metrics
    ]
    if not module_fields:
        errors.append(f"{source}.metrics must retain a measured module count.")
    for field in module_fields:
        if _nonnegative_int(metrics.get(field)) != len(modules):
            errors.append(f"{source}.metrics.{field} disagrees with module evidence.")
    for field in ("violations_found", "violations_detected"):
        if field in metrics and _nonnegative_int(metrics.get(field)) != len(
            selected_violations
        ):
            errors.append(f"{source}.metrics.{field} disagrees with replayed evidence.")
    if "candidate_violations_detected" in metrics and _nonnegative_int(
        metrics.get("candidate_violations_detected")
    ) != len(candidate_budgeted) + len(fatal_violations):
        errors.append(
            f"{source}.metrics.candidate_violations_detected disagrees with replayed evidence."
        )
    if metrics.get("caps_exceeded") is not caps_exceeded:
        errors.append(
            f"{source}.metrics.caps_exceeded disagrees with replayed evidence."
        )
    for field in ("max_spectral_norm", "max_spectral_norm_final"):
        if field in metrics:
            observed = _finite(metrics.get(field))
            if observed is None or not _close(observed, max(final.values())):
                errors.append(f"{source}.metrics.{field} disagrees with final_metrics.")
    for field in ("mean_spectral_norm", "mean_spectral_norm_final"):
        if field in metrics:
            observed = _finite(metrics.get(field))
            expected_mean = sum(final.values()) / len(final)
            if observed is None or not _close(observed, expected_mean):
                errors.append(f"{source}.metrics.{field} disagrees with final_metrics.")

    if entry.get("passed") is not expected_pass:
        errors.append(f"{source}.passed disagrees with replayed spectral evidence.")
    if str(entry.get("decision") or "").strip().lower() != expected_decision:
        errors.append(f"{source}.decision disagrees with replayed spectral evidence.")
    _compare_violations(
        errors, entry.get("violations"), selected_violations, f"{source}.violations"
    )


def _validate_external_baseline(
    errors: list[str],
    *,
    report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    source: str,
) -> None:
    external_required = metrics.get("external_baseline_required")
    external_ready = metrics.get("external_baseline_ready")
    if "external_baseline_required" in metrics and not isinstance(
        external_required, bool
    ):
        errors.append(f"{source}.metrics.external_baseline_required must be a boolean.")
    if "external_baseline_ready" in metrics and not isinstance(external_ready, bool):
        errors.append(f"{source}.metrics.external_baseline_ready must be a boolean.")
    context = _mapping(report.get("context"))
    report_requires_external = (
        context.get("baseline_guard_evidence_required") is True
        if context is not None
        else False
    )
    if report_requires_external and external_required is not True:
        errors.append(
            f"{source} does not bind the externally required spectral baseline."
        )
    if external_required is True and external_ready is not True:
        errors.append(f"{source} required external spectral baseline is not ready.")
    if external_required is True and metrics.get("baseline_source") != "external_run":
        errors.append(f"{source}.metrics.baseline_source must be external_run.")


def _validate_public_spectral_summary(
    errors: list[str],
    *,
    report: Mapping[str, Any],
    selected_violations: list[dict[str, Any]],
    modules: set[str],
    caps_applied: int,
    max_caps: int,
    caps_exceeded: bool,
    expected_pass: bool,
    expected_decision: str,
    accepted_statuses: set[str],
) -> None:
    spectral = _mapping(report.get("spectral"))
    if spectral is None:
        errors.append("spectral summary is required for strict assurance.")
        return
    mirror_expectations: tuple[tuple[str, Any], ...] = (
        ("passed", expected_pass),
        ("decision", expected_decision),
        ("caps_applied", caps_applied),
        ("max_caps", max_caps),
        ("caps_exceeded", caps_exceeded),
    )
    for field, mirror_expected in mirror_expectations:
        observed_value = spectral.get(field)
        if field == "decision":
            observed_value = str(observed_value or "").strip().lower()
        if observed_value != mirror_expected:
            errors.append(f"spectral.{field} disagrees with the raw spectral record.")
    if (
        "status" in spectral
        and str(spectral.get("status") or "").strip().lower() not in accepted_statuses
    ):
        errors.append("spectral.status disagrees with replayed spectral evidence.")
    _compare_violations(
        errors,
        spectral.get("violations"),
        selected_violations,
        "spectral.violations",
    )
    summary = _mapping(spectral.get("summary"))
    if summary is None:
        errors.append("spectral.summary is required for strict assurance.")
        return
    summary_expected = {
        "modules_checked": len(modules),
        "caps_applied": caps_applied,
        "max_caps": max_caps,
        "caps_exceeded": caps_exceeded,
        "status": "capped"
        if caps_applied and expected_pass
        else ("stable" if expected_pass else "fail"),
    }
    for field, summary_expected_value in summary_expected.items():
        observed_value = summary.get(field)
        if field == "status":
            observed_value = str(observed_value or "").strip().lower()
        if observed_value != summary_expected_value:
            errors.append(
                f"spectral.summary.{field} disagrees with the raw spectral record."
            )


def replay_spectral_guard(
    report: Mapping[str, Any],
    entry: Mapping[str, Any],
    source: str,
    *,
    enforce_outcome: bool = True,
) -> list[str]:
    """Replay one retained spectral record and its public summary mirrors.

    ``enforce_outcome`` changes only whether a complete cap-budget finding is
    acceptance-blocking. Fatal findings and incomplete or inconsistent evidence
    always fail replay.
    """

    errors: list[str] = []
    metrics = _mapping(entry.get("metrics"))
    policy = _mapping(entry.get("policy"))
    baseline_metrics = _mapping(entry.get("baseline_metrics"))
    if metrics is None or not metrics:
        return [f"{source}.metrics must be a non-empty object for strict assurance."]
    if policy is None or not policy:
        return [f"{source}.policy must be a non-empty object for strict assurance."]
    if baseline_metrics is None or not baseline_metrics:
        return [
            f"{source}.baseline_metrics must be a non-empty object for strict assurance."
        ]

    baseline = _numeric_map(
        errors,
        baseline_metrics.get("module_sigmas"),
        f"{source}.baseline_metrics.module_sigmas",
        nonnegative=True,
    )
    final = _numeric_map(
        errors, entry.get("final_metrics"), f"{source}.final_metrics", nonnegative=True
    )
    z_scores = _numeric_map(
        errors, entry.get("final_z_scores"), f"{source}.final_z_scores"
    )
    families = _family_map(
        errors, entry.get("module_family_map"), f"{source}.module_family_map"
    )
    if baseline is None or final is None or z_scores is None or families is None:
        return errors
    modules = set(baseline)
    if (
        not modules
        or set(final) != modules
        or set(z_scores) != modules
        or set(families) != modules
    ):
        errors.append(
            f"{source} spectral module inventories must match across baseline sigmas, "
            "final sigmas, z-scores, and family identities."
        )
        return errors
    _replay_measurement_inventory(
        errors,
        entry,
        source,
        baseline_modules=modules,
        final_modules=set(final),
    )

    replay_policy = _validated_policy(errors, policy, families, source)
    if errors or replay_policy is None:
        return errors

    replayed_stats = _family_stats(baseline, families)
    retained_stats = _mapping(baseline_metrics.get("family_stats"))
    if retained_stats is None:
        errors.append(f"{source}.baseline_metrics.family_stats must be an object.")
    else:
        _compare_tree(
            errors,
            retained_stats,
            replayed_stats,
            f"{source}.baseline_metrics.family_stats",
        )

    replayed_z = _replayed_z_scores(
        errors,
        source=source,
        modules=modules,
        baseline=baseline,
        final=final,
        z_scores=z_scores,
        families=families,
        family_stats=replayed_stats,
        deadband=replay_policy.deadband,
    )
    candidate_budgeted, fatal_violations = _initial_violations(
        modules=modules,
        baseline=baseline,
        final=final,
        replayed_z=replayed_z,
        families=families,
        family_caps=replay_policy.family_caps,
        max_norm=replay_policy.max_norm,
    )

    baseline_degeneracy: Mapping[str, Mapping[str, float]] = {}
    final_degeneracy: Mapping[str, Mapping[str, float]] = {}
    if replay_policy.degeneracy_enabled:
        parsed_baseline_degeneracy = _degeneracy_map(
            errors,
            baseline_metrics.get("baseline_degeneracy"),
            f"{source}.baseline_metrics.baseline_degeneracy",
            modules,
        )
        parsed_final_degeneracy = _degeneracy_map(
            errors,
            entry.get("final_degeneracy"),
            f"{source}.final_degeneracy",
            modules,
        )
        if (
            parsed_baseline_degeneracy is not None
            and parsed_final_degeneracy is not None
        ):
            baseline_degeneracy = parsed_baseline_degeneracy
            final_degeneracy = parsed_final_degeneracy
            _append_degeneracy_violations(
                candidate_budgeted,
                fatal_violations,
                modules=modules,
                families=families,
                baseline_degeneracy=baseline_degeneracy,
                final_degeneracy=final_degeneracy,
                thresholds=replay_policy.thresholds,
            )

    selected_budgeted, selection = _selected_budgeted_violations(
        candidate_budgeted,
        method=replay_policy.method,
        alpha=replay_policy.alpha,
        configured_m=replay_policy.configured_m,
    )
    _compare_tree(
        errors,
        metrics.get("multiple_testing_selection"),
        selection,
        f"{source}.metrics.multiple_testing_selection",
    )
    _compare_tree(
        errors,
        metrics.get("multiple_testing"),
        dict(replay_policy.multiple_testing),
        f"{source}.metrics.multiple_testing",
    )
    _compare_tree(
        errors,
        metrics.get("family_caps"),
        dict(replay_policy.caps_raw),
        f"{source}.metrics.family_caps",
    )

    selected_violations = [*fatal_violations, *selected_budgeted]
    caps_applied = len(selected_budgeted)
    caps_exceeded = caps_applied > replay_policy.max_caps
    expected_pass = not fatal_violations and not caps_exceeded
    expected_decision = (
        "block" if not expected_pass else ("monitor" if caps_applied else "allow")
    )
    _replay_correction_ledger(
        errors,
        entry=entry,
        source=source,
        metrics=metrics,
        baseline=baseline,
        final=final,
        families=families,
        family_stats=replayed_stats,
        family_caps=replay_policy.family_caps,
        deadband=replay_policy.deadband,
        max_norm=replay_policy.max_norm,
        method=replay_policy.method,
        alpha=replay_policy.alpha,
        configured_m=replay_policy.configured_m,
        degeneracy_enabled=replay_policy.degeneracy_enabled,
        baseline_degeneracy=baseline_degeneracy,
        thresholds=replay_policy.thresholds,
        correction_enabled=replay_policy.correction_enabled,
        correction_cap_ratio=replay_policy.correction_cap_ratio,
        final_caps_applied=caps_applied,
        final_caps_exceeded=caps_exceeded,
    )
    accepted_statuses = (
        {"fail", "failed", "block"}
        if not expected_pass
        else ({"capped"} if caps_applied else {"pass", "stable"})
    )

    _validate_entry_metrics(
        errors,
        entry=entry,
        metrics=metrics,
        source=source,
        modules=modules,
        candidate_budgeted=candidate_budgeted,
        fatal_violations=fatal_violations,
        selected_violations=selected_violations,
        caps_applied=caps_applied,
        max_caps=replay_policy.max_caps,
        caps_exceeded=caps_exceeded,
        final=final,
        expected_pass=expected_pass,
        expected_decision=expected_decision,
    )
    _validate_external_baseline(
        errors,
        report=report,
        metrics=metrics,
        source=source,
    )
    _validate_public_spectral_summary(
        errors,
        report=report,
        selected_violations=selected_violations,
        modules=modules,
        caps_applied=caps_applied,
        max_caps=replay_policy.max_caps,
        caps_exceeded=caps_exceeded,
        expected_pass=expected_pass,
        expected_decision=expected_decision,
        accepted_statuses=accepted_statuses,
    )

    if fatal_violations:
        errors.append("replayed spectral evidence contains fatal violations.")
    if caps_exceeded and enforce_outcome:
        errors.append("replayed spectral evidence exceeds the selected-cap budget.")
    return errors


__all__ = ["replay_spectral_guard"]
