"""Baseline-relative guard warning extraction."""

from __future__ import annotations

import math
from typing import Any

_PARSE_EXCEPTIONS = (AttributeError, KeyError, OverflowError, TypeError, ValueError)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except _PARSE_EXCEPTIONS:
        return None
    return out if math.isfinite(out) else None


def _finite_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        out = int(value)
    except _PARSE_EXCEPTIONS:
        return None
    return out if out >= 0 else None


def _guard_section(value: dict[str, Any], guard_name: str) -> dict[str, Any]:
    section = _as_dict(value.get(guard_name))

    metrics = _as_dict(value.get("metrics"))
    metric_section = _as_dict(metrics.get(guard_name))

    matching_entries: list[dict[str, Any]] = []
    for entry_raw in _as_list(value.get("guards")):
        entry = _as_dict(entry_raw)
        entry_name = str(entry.get("name") or "").strip().lower()
        entry_stage = str(entry.get("stage") or "").strip().lower()
        is_match = entry_name == guard_name
        if guard_name == "invariants":
            is_match = (
                is_match or entry_name == "invariants_post" or entry_stage == "post"
            )
        if not is_match:
            continue
        matching_entries.append(entry)

    if matching_entries:
        # A staged post-edit result supersedes its pre-edit result for warning
        # attribution. Select one representation; never add both copies.
        entry = next(
            (
                candidate
                for candidate in matching_entries
                if str(candidate.get("name") or "").strip().lower() == "invariants_post"
                or str(candidate.get("stage") or "").strip().lower() == "post"
            ),
            matching_entries[0],
        )
        merged = dict(section)
        merged.update(metric_section)
        merged.update(_as_dict(entry.get("metrics")))
        for key in (
            "violations",
            "top_violations",
            "final_z_scores",
            "module_family_map",
            "epsilon_violations",
            "policy",
            "baseline_metrics",
            "final_metrics",
            "final_degeneracy",
            "measurement_inventory",
            "correction_ledger",
        ):
            if key not in entry:
                continue
            existing = merged.get(key)
            candidate = entry[key]
            if (
                key not in merged
                or (isinstance(existing, list) and not existing)
                or (isinstance(existing, dict) and not existing)
                or existing is None
            ):
                merged[key] = candidate
            elif key in {"violations", "top_violations", "epsilon_violations"} and (
                not isinstance(existing, list) or not existing
            ):
                merged[key] = entry[key]
        return merged

    if section:
        return section
    return metric_section


def _policy_gate(validation: dict[str, Any], key: str) -> str:
    value = validation.get(key)
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return "unknown"


def _warning(
    *,
    guard: str,
    kind: str,
    message: str,
    policy_gate: str,
    family: str | None = None,
    module: str | None = None,
    baseline: dict[str, Any] | None = None,
    subject: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "guard": guard,
        "kind": kind,
        "severity": "warning",
        "policy_gate": policy_gate,
        "message": message,
    }
    if family:
        item["family"] = family
    if module:
        item["module"] = module
    if baseline is not None:
        item["baseline"] = baseline
    if subject is not None:
        item["subject"] = subject
    return item


def _family_kappa(spectral: dict[str, Any], family: str | None) -> float | None:
    if not family:
        return None
    family_caps = _as_dict(spectral.get("family_caps"))
    cap_entry = _as_dict(family_caps.get(family))
    kappa = _finite_float(cap_entry.get("kappa"))
    if kappa is not None:
        return kappa
    families = _as_dict(spectral.get("families"))
    family_entry = _as_dict(families.get(family))
    return _finite_float(family_entry.get("kappa"))


def _spectral_capped_modules(
    spectral: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    modules: dict[tuple[str, str], dict[str, Any]] = {}
    raw_violations = list(_as_list(spectral.get("top_violations")))
    raw_violations.extend(_as_list(spectral.get("violations")))
    for entry_raw in raw_violations:
        entry = _as_dict(entry_raw)
        module = str(entry.get("module") or "").strip()
        family = str(entry.get("family") or "").strip()
        if not module:
            continue
        violation_type = str(entry.get("type") or "family_z_cap").strip()
        if violation_type and violation_type not in {"family_z_cap", "spectral_cap"}:
            continue
        if not family:
            family = "unknown"
        z_score = _finite_float(entry.get("z_score"))
        if z_score is None:
            z_score = _finite_float(entry.get("z"))
        kappa = _finite_float(entry.get("kappa")) or _family_kappa(spectral, family)
        modules[(family, module)] = {
            "capped": True,
            "family": family,
            "module": module,
            "z_score": z_score,
            "kappa": kappa,
        }

    top_z_scores = _as_dict(spectral.get("top_z_scores"))
    for family_raw, entries_raw in top_z_scores.items():
        family = str(family_raw or "").strip() or "unknown"
        kappa = _family_kappa(spectral, family)
        if kappa is None:
            continue
        for entry_raw in _as_list(entries_raw):
            entry = _as_dict(entry_raw)
            module = str(entry.get("module") or "").strip()
            z_score = _finite_float(entry.get("z"))
            if not module or z_score is None or abs(z_score) <= kappa:
                continue
            modules.setdefault(
                (family, module),
                {
                    "capped": True,
                    "family": family,
                    "module": module,
                    "z_score": z_score,
                    "kappa": kappa,
                },
            )
    return modules


def _spectral_deadband(spectral: dict[str, Any]) -> float:
    for source in (
        spectral,
        _as_dict(spectral.get("policy")),
        _as_dict(spectral.get("summary")),
    ):
        value = _finite_float(source.get("deadband"))
        if value is not None:
            return max(value, 0.0)
    return 0.1


def _spectral_warnings(
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    subject_spectral = _guard_section(subject, "spectral")
    baseline_spectral = _guard_section(baseline, "spectral")
    if not subject_spectral:
        return []

    subject_modules = _spectral_capped_modules(subject_spectral)
    baseline_modules = _spectral_capped_modules(baseline_spectral)
    policy_gate = _policy_gate(validation, "spectral_stable")
    warnings: list[dict[str, Any]] = []

    for family, module in sorted(set(subject_modules) - set(baseline_modules)):
        subject_entry = dict(subject_modules[(family, module)])
        baseline_entry = {"capped": False}
        warnings.append(
            _warning(
                guard="spectral",
                kind="new_capped_module",
                family=family,
                module=module,
                baseline=baseline_entry,
                subject=subject_entry,
                policy_gate=policy_gate,
                message=(
                    "Policy passes, but subject has a new capped module versus baseline."
                    if policy_gate == "pass"
                    else "Subject has a new capped module versus baseline."
                ),
            )
        )

    baseline_count = _finite_int(baseline_spectral.get("caps_applied")) or 0
    subject_count = _finite_int(subject_spectral.get("caps_applied")) or 0
    if subject_count > baseline_count and not warnings:
        warnings.append(
            _warning(
                guard="spectral",
                kind="cap_count_increase",
                baseline={"caps_applied": baseline_count},
                subject={"caps_applied": subject_count},
                policy_gate=policy_gate,
                message=(
                    "Policy passes, but subject applies more spectral caps than baseline."
                    if policy_gate == "pass"
                    else "Subject applies more spectral caps than baseline."
                ),
            )
        )

    deadband = _spectral_deadband(subject_spectral)
    for family, module in sorted(set(subject_modules) & set(baseline_modules)):
        subject_z = _finite_float(subject_modules[(family, module)].get("z_score"))
        baseline_z = _finite_float(baseline_modules[(family, module)].get("z_score"))
        if subject_z is None or baseline_z is None:
            continue
        delta = abs(subject_z) - abs(baseline_z)
        if delta <= deadband:
            continue
        warnings.append(
            _warning(
                guard="spectral",
                kind="capped_module_z_score_increase",
                family=family,
                module=module,
                baseline={"z_score": baseline_z},
                subject={"z_score": subject_z, "delta_abs_z": delta},
                policy_gate=policy_gate,
                message=(
                    "Policy passes, but a capped module moved farther beyond the baseline z-score."
                    if policy_gate == "pass"
                    else "A capped module moved farther beyond the baseline z-score."
                ),
            )
        )
    return warnings


def _epsilon_violation_keys(
    rmt: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry_raw in _as_list(rmt.get("epsilon_violations")):
        entry = _as_dict(entry_raw)
        family = str(entry.get("family") or "").strip() or "unknown"
        module = str(entry.get("module") or family).strip() or family
        out[(family, module)] = {
            "family": family,
            "module": module,
            "edge_base": _finite_float(entry.get("edge_base")),
            "edge_cur": _finite_float(entry.get("edge_cur")),
            "delta": _finite_float(entry.get("delta")),
            "epsilon": _finite_float(entry.get("epsilon")),
        }
    return out


def _rmt_warnings(
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    subject_rmt = _guard_section(subject, "rmt")
    baseline_rmt = _guard_section(baseline, "rmt")
    if not subject_rmt:
        return []
    subject_violations = _epsilon_violation_keys(subject_rmt)
    baseline_violations = _epsilon_violation_keys(baseline_rmt)
    policy_gate = _policy_gate(validation, "rmt_stable")
    warnings: list[dict[str, Any]] = []

    for family, module in sorted(set(subject_violations) - set(baseline_violations)):
        warnings.append(
            _warning(
                guard="rmt",
                kind="new_epsilon_violation",
                family=family,
                module=None if module == family else module,
                baseline={"epsilon_violation": False},
                subject=subject_violations[(family, module)],
                policy_gate=policy_gate,
                message=(
                    "Policy passes, but subject has a new RMT epsilon violation versus baseline."
                    if policy_gate == "pass"
                    else "Subject has a new RMT epsilon violation versus baseline."
                ),
            )
        )
    return warnings


def _variance_signal(value: dict[str, Any]) -> dict[str, Any]:
    variance = _guard_section(value, "variance")
    predictive_gate = _as_dict(variance.get("predictive_gate"))
    ab_test = _as_dict(variance.get("ab_test"))
    enabled = bool(variance.get("enabled", False))
    evaluated = bool(predictive_gate.get("evaluated", False))
    mean_delta = _finite_float(predictive_gate.get("mean_delta"))
    delta_ci = predictive_gate.get("delta_ci")
    has_ci = isinstance(delta_ci, list | tuple) and len(delta_ci) == 2
    has_ab = bool(ab_test)
    return {
        "enabled": enabled,
        "evaluated": evaluated,
        "mean_delta": mean_delta,
        "has_delta_ci": has_ci,
        "has_ab_test": has_ab,
        "active": bool(
            enabled and (evaluated or mean_delta is not None or has_ci or has_ab)
        ),
    }


def _variance_warnings(
    *, subject: dict[str, Any], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    subject_signal = _variance_signal(subject)
    baseline_signal = _variance_signal(baseline)
    if not subject_signal["active"] or baseline_signal["active"]:
        return []
    return [
        _warning(
            guard="variance",
            kind="new_predictive_signal",
            baseline=baseline_signal,
            subject=subject_signal,
            policy_gate="pass",
            message=(
                "Policy passes, but subject has a new variance/VE signal versus baseline."
            ),
        )
    ]


def _invariant_warning_count(value: dict[str, Any]) -> int:
    invariants = _guard_section(value, "invariants")
    summary = _as_dict(invariants.get("summary"))
    # Canonical evaluation reports place the count under ``summary`` while
    # raw run reports expose it directly from ``guards[].metrics``. Prefer one
    # representation instead of adding them because assembled reports may
    # retain both views of the same observation.
    for candidate in (
        summary.get("warning_violations"),
        summary.get("warnings"),
        invariants.get("warning_violations"),
        invariants.get("warnings"),
    ):
        count = _finite_int(candidate)
        if count is not None:
            return count
    return 0


def _invariant_warnings(
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any],
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    subject_count = _invariant_warning_count(subject)
    baseline_count = _invariant_warning_count(baseline)
    if subject_count <= baseline_count:
        return []
    return [
        _warning(
            guard="invariants",
            kind="warning_count_increase",
            baseline={"warning_violations": baseline_count},
            subject={"warning_violations": subject_count},
            policy_gate=_policy_gate(validation, "invariants_pass"),
            message=(
                "Policy passes, but subject has more non-fatal invariant warnings than baseline."
                if validation.get("invariants_pass") is True
                else "Subject has more non-fatal invariant warnings than baseline."
            ),
        )
    ]


def build_guard_warnings(
    *,
    subject: dict[str, Any],
    baseline: dict[str, Any] | None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build additive baseline-relative guard warnings for an evaluation report."""

    subject_map = _as_dict(subject)
    baseline_map = _as_dict(baseline)
    validation_map = _as_dict(validation)
    warnings: list[dict[str, Any]] = []
    warnings.extend(
        _spectral_warnings(
            subject=subject_map,
            baseline=baseline_map,
            validation=validation_map,
        )
    )
    warnings.extend(
        _rmt_warnings(
            subject=subject_map,
            baseline=baseline_map,
            validation=validation_map,
        )
    )
    warnings.extend(_variance_warnings(subject=subject_map, baseline=baseline_map))
    warnings.extend(
        _invariant_warnings(
            subject=subject_map,
            baseline=baseline_map,
            validation=validation_map,
        )
    )
    return {
        "present": bool(warnings),
        "warning_count": len(warnings),
        "warnings": warnings,
    }


__all__ = ["build_guard_warnings"]
