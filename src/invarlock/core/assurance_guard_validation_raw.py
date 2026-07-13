"""Strict reconciliation of retained guard observations.

Display summaries are projections.  This module requires the ordered guard
inventory to retain the observations and applied policy needed to justify a
strict guard decision, then reconciles those observations with the projections.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from invarlock.eval.guard_metric_impact import (
    arm_facts_match_measurements,
    compute_guard_metric_impact,
    guard_metric_impact_payload_errors,
)

from .assurance_guard_validation_common import (
    _finite_number,
    _mapping,
    _nonnegative_int,
)
from .assurance_spectral_replay import replay_spectral_guard


def _entries(
    inventory: Sequence[tuple[str, dict[str, Any], str]], name: str
) -> list[tuple[dict[str, Any], str]]:
    return [(entry, source) for guard, entry, source in inventory if guard == name]


def _required_mapping(
    errors: list[str], entry: Mapping[str, Any], source: str, field: str
) -> dict[str, Any] | None:
    value = _mapping(entry.get(field))
    if value is None or not value:
        errors.append(
            f"{source}.{field} must be a non-empty object for strict assurance."
        )
        return None
    return value


def _spectral_raw_errors(
    report: Mapping[str, Any],
    inventory: Sequence[tuple[str, dict[str, Any], str]],
) -> list[str]:
    entries = _entries(inventory, "spectral")
    if len(entries) != 1:
        return ["strict assurance requires exactly one raw spectral guard record."]
    entry, source = entries[0]
    return replay_spectral_guard(report, entry, source)


def _rmt_raw_errors(
    report: Mapping[str, Any],
    inventory: Sequence[tuple[str, dict[str, Any], str]],
) -> list[str]:
    errors: list[str] = []
    entries = _entries(inventory, "rmt")
    if len(entries) != 1:
        return ["strict assurance requires exactly one raw rmt guard record."]
    entry, source = entries[0]
    metrics = _required_mapping(errors, entry, source, "metrics")
    policy = _required_mapping(errors, entry, source, "policy")
    if metrics is None:
        return errors
    contract = _required_mapping(
        errors, metrics, f"{source}.metrics", "measurement_contract"
    )
    _ = contract

    baseline = _mapping(metrics.get("edge_risk_by_family_base"))
    current = _mapping(metrics.get("edge_risk_by_family"))
    baseline_modules = _mapping(metrics.get("edge_risk_by_module_base"))
    current_modules = _mapping(metrics.get("edge_risk_by_module"))
    family_map = _mapping(metrics.get("module_family_map"))
    epsilon_map = _mapping(metrics.get("epsilon_by_family"))
    if not baseline or not current:
        errors.append(
            f"{source}.metrics requires non-empty baseline/current family maps."
        )
        return errors
    if not baseline_modules or not current_modules or not family_map:
        errors.append(
            f"{source}.metrics requires module-level baseline/current risks "
            "and family identities."
        )
        return errors
    if set(baseline_modules) != set(current_modules) or set(baseline_modules) != set(
        family_map
    ):
        errors.append(f"{source}.metrics module-level RMT inventories must match.")
        return errors
    replay_base: dict[str, float] = {}
    replay_current: dict[str, float] = {}
    for module_name in sorted(baseline_modules):
        family = family_map.get(module_name)
        base_value = _finite_number(baseline_modules.get(module_name))
        current_value = _finite_number(current_modules.get(module_name))
        if (
            not isinstance(family, str)
            or not family
            or base_value is None
            or current_value is None
        ):
            errors.append(
                f"{source}.metrics has invalid module evidence for {module_name!r}."
            )
            continue
        replay_base[family] = max(replay_base.get(family, 0.0), base_value)
        replay_current[family] = max(replay_current.get(family, 0.0), current_value)
    for family in sorted(set(replay_base) | set(replay_current)):
        observed_base = _finite_number(baseline.get(family))
        observed_current = _finite_number(current.get(family))
        if observed_base is None or not math.isclose(
            observed_base,
            replay_base.get(family, 0.0),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            errors.append(
                f"{source}.metrics family baseline disagrees with module evidence "
                f"for {family!r}."
            )
        if observed_current is None or not math.isclose(
            observed_current,
            replay_current.get(family, 0.0),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            errors.append(
                f"{source}.metrics family current value disagrees with module "
                f"evidence for {family!r}."
            )
    if set(baseline) != set(current):
        errors.append(f"{source}.metrics baseline/current family sets must match.")
    default = _finite_number(
        (policy or {}).get("epsilon_default", metrics.get("epsilon_default"))
    )
    if default is None or default < 0.0:
        errors.append(f"{source} requires a non-negative epsilon_default policy.")
    replay_violations: list[str] = []
    for family in sorted(set(baseline) & set(current)):
        base = _finite_number(baseline.get(family))
        observed = _finite_number(current.get(family))
        epsilon = _finite_number((epsilon_map or {}).get(family, default))
        if base is None or observed is None or epsilon is None or base < 0.0:
            errors.append(f"{source}.metrics has invalid RMT values for {family!r}.")
            continue
        if base == 0.0:
            if observed != 0.0:
                replay_violations.append(family)
            continue
        if observed > (1.0 + epsilon) * base:
            replay_violations.append(family)
    stable = metrics.get("stable")
    expected_stable = not replay_violations
    if not isinstance(stable, bool) or stable is not expected_stable:
        errors.append(f"{source}.metrics.stable disagrees with replayed RMT evidence.")
    if (
        isinstance(entry.get("passed"), bool)
        and entry.get("passed") is not expected_stable
    ):
        errors.append(f"{source}.passed disagrees with replayed RMT evidence.")

    rmt = _mapping(report.get("rmt"))
    if rmt is not None:
        for field, raw in (
            ("edge_risk_by_family_base", baseline),
            ("edge_risk_by_family", current),
            ("epsilon_by_family", epsilon_map),
        ):
            if raw is not None and rmt.get(field) != raw:
                errors.append(f"rmt.{field} disagrees with the raw rmt record.")
    return errors


def _invariants_raw_errors(
    inventory: Sequence[tuple[str, dict[str, Any], str]],
) -> list[str]:
    errors: list[str] = []
    entries = _entries(inventory, "invariants")
    if len(entries) != 2:
        return ["strict assurance requires pre and post raw invariant records."]
    for entry, source in entries:
        metrics = _required_mapping(errors, entry, source, "metrics")
        policy = _required_mapping(errors, entry, source, "policy")
        details = _required_mapping(errors, entry, source, "details")
        if metrics is None or details is None:
            continue
        baseline = _mapping(details.get("baseline_checks"))
        current = _mapping(details.get("current_checks"))
        if not baseline or not current:
            errors.append(
                f"{source}.details requires non-empty baseline_checks/current_checks."
            )
            continue
        if baseline != current:
            errors.append(f"{source} invariant observations changed from baseline.")
        checks = _nonnegative_int(metrics.get("checks_performed"))
        if checks != len(baseline):
            errors.append(f"{source}.metrics.checks_performed disagrees with details.")
        violations = entry.get("violations")
        observed_violations = len(violations) if isinstance(violations, list) else None
        if metrics.get("violations_found") != observed_violations:
            errors.append(
                f"{source}.metrics.violations_found disagrees with violations."
            )
        if policy is not None:
            if (
                policy.get("strict_mode") is not True
                or policy.get("on_fail") != "block"
            ):
                errors.append(
                    f"{source}.policy is not fail-closed for strict assurance."
                )
    return errors


def _guard_metric_impact_raw_errors(report: Mapping[str, Any]) -> list[str]:
    metric_impact = _mapping(report.get("guard_metric_impact"))
    if metric_impact is None:
        return []
    errors: list[str] = []
    stale_fields = {
        "bare_ppl",
        "guarded_ppl",
        "impact_ratio",
        "impact_percent",
        "impact_threshold",
        "metric_direction",
        "impact_basis",
        "impact_value",
    }
    for field in sorted(stale_fields & set(metric_impact)):
        errors.append(
            f"guard_metric_impact.{field} is not part of the guard metric impact contract."
        )
    measurement = compute_guard_metric_impact(
        metric_impact.get("metric_kind"),
        metric_impact.get("bare_value"),
        metric_impact.get("guarded_value"),
    )
    if measurement is None:
        errors.append(
            "guard_metric_impact requires valid retained primary-metric measurements."
        )
    else:
        if not arm_facts_match_measurements(
            metric_impact.get("metric_kind"),
            metric_impact.get("bare_facts"),
            metric_impact.get("guarded_facts"),
            metric_impact.get("bare_value"),
            metric_impact.get("guarded_value"),
        ):
            errors.append(
                "guard_metric_impact arm facts do not replay the paired measurements."
            )
        if metric_impact.get("direction") != measurement.direction:
            errors.append("guard_metric_impact direction disagrees with metric_kind.")
        if metric_impact.get("degradation_basis") != measurement.degradation_basis:
            errors.append(
                "guard_metric_impact degradation_basis disagrees with metric_kind."
            )
        degradation = _finite_number(metric_impact.get("degradation"))
        if degradation is None or not math.isclose(
            degradation,
            measurement.degradation,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append(
                "guard_metric_impact degradation disagrees with retained measurements."
            )
        display_value = _finite_number(metric_impact.get("display_value"))
        if display_value is None or not math.isclose(
            display_value,
            measurement.display_value,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            errors.append(
                "guard_metric_impact display_value disagrees with retained measurements."
            )
        if metric_impact.get("display_unit") != measurement.display_unit:
            errors.append(
                "guard_metric_impact display_unit disagrees with metric_kind."
            )
    checks = _mapping(metric_impact.get("checks"))
    if not checks:
        errors.append(
            "guard_metric_impact.checks must retain measured consistency checks."
        )
    elif any(value is not True for value in checks.values()):
        errors.append("guard_metric_impact.checks must contain only passing booleans.")
    errors.extend(
        guard_metric_impact_payload_errors(
            metric_impact,
            subject_report=report,
            require_bare_report=True,
        )
    )
    return errors


def raw_guard_evidence_errors(
    report: Mapping[str, Any],
    inventory: Sequence[tuple[str, dict[str, Any], str]],
    *,
    require_complete: bool,
) -> list[str]:
    """Validate raw evidence needed to replay strict guard outcomes."""

    if not require_complete:
        return []
    return [
        *_spectral_raw_errors(report, inventory),
        *_rmt_raw_errors(report, inventory),
        *_invariants_raw_errors(inventory),
        *_guard_metric_impact_raw_errors(report),
    ]


__all__ = ["raw_guard_evidence_errors"]
