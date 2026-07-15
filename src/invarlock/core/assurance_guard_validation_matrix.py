"""Spectral and random-matrix assurance guard reconciliation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .assurance_guard_validation_common import (
    _consistent_bool,
    _consistent_nonnegative_int,
    _field_values,
    _finite_number,
    _mapping,
    _nonnegative_int,
    _normalized_token,
)


def _spectral_errors(
    report: Mapping[str, Any],
    inventory: Sequence[tuple[str, dict[str, Any], str]],
    *,
    require_complete: bool,
    enforce_outcome: bool = True,
) -> list[str]:
    spectral = _mapping(report.get("spectral"))
    if spectral is None:
        return []
    errors: list[str] = []
    evaluated = spectral.get("evaluated")
    if "evaluated" not in spectral:
        if require_complete:
            errors.append("spectral.evaluated is required for strict assurance.")
    elif not isinstance(evaluated, bool):
        errors.append("spectral.evaluated must be a boolean.")
    elif require_complete and evaluated is not True:
        errors.append("spectral.evaluated must be true for strict assurance.")

    summary = _mapping(spectral.get("summary"))
    policy = _mapping(spectral.get("policy"))
    sources: list[tuple[str, Mapping[str, Any]]] = [("spectral", spectral)]
    if summary is not None:
        sources.append(("spectral.summary", summary))
    elif require_complete:
        errors.append("spectral.summary is required for strict assurance.")
    if policy is not None:
        sources.append(("spectral.policy", policy))
    for guard_name, entry, source in inventory:
        if guard_name != "spectral":
            continue
        metrics = _mapping(entry.get("metrics"))
        guard_policy = _mapping(entry.get("policy"))
        if metrics is not None:
            sources.append((f"{source}.metrics", metrics))
        if guard_policy is not None:
            sources.append((f"{source}.policy", guard_policy))

    caps_applied = _consistent_nonnegative_int(
        errors,
        _field_values(sources, "caps_applied"),
        required_path="spectral.caps_applied" if require_complete else None,
    )
    max_caps = _consistent_nonnegative_int(
        errors,
        _field_values(sources, "max_caps"),
        required_path="spectral.max_caps" if require_complete else None,
    )
    caps_exceeded = _consistent_bool(
        errors,
        _field_values(sources, "caps_exceeded"),
        required_path="spectral.caps_exceeded" if require_complete else None,
    )
    if caps_exceeded is True and enforce_outcome:
        errors.append("spectral.caps_exceeded is true.")
    submitted_caps = _nonnegative_int(spectral.get("caps_applied"))
    submitted_max_caps = _nonnegative_int(spectral.get("max_caps"))
    if (
        submitted_caps is not None
        and submitted_max_caps is not None
        and submitted_caps > submitted_max_caps
        and enforce_outcome
    ):
        errors.append(
            "spectral.caps_applied exceeds spectral.max_caps "
            f"({submitted_caps} > {submitted_max_caps})."
        )
    if (
        caps_applied is not None
        and max_caps is not None
        and caps_applied > max_caps
        and enforce_outcome
    ):
        errors.append(
            "spectral.caps_applied exceeds spectral.max_caps "
            f"({caps_applied} > {max_caps})."
        )
    if caps_applied is not None and caps_applied > 0 and max_caps is None:
        errors.append(
            "spectral caps cannot be accepted without a typed max_caps limit."
        )

    if summary is not None:
        modules_checked = summary.get("modules_checked")
        if "modules_checked" not in summary:
            if require_complete:
                errors.append(
                    "spectral.summary.modules_checked is required for strict assurance."
                )
        else:
            count = _nonnegative_int(modules_checked)
            if count is None:
                errors.append("spectral.summary.modules_checked must be an integer.")
            elif require_complete and count == 0:
                errors.append(
                    "spectral.summary.modules_checked must be positive for strict assurance."
                )
        summary_status = summary.get("status")
        if "status" in summary:
            if not isinstance(summary_status, str):
                errors.append("spectral.summary.status must be a string.")
            else:
                normalized = _normalized_token(summary_status)
                if normalized not in {"stable", "capped"} and not (
                    not enforce_outcome
                    and normalized in {"block", "fail", "failed", "unstable"}
                ):
                    errors.append("spectral.summary.status is not a passing state.")
                elif normalized == "stable" and caps_applied not in {None, 0}:
                    errors.append(
                        "spectral.summary.status=stable contradicts caps_applied > 0."
                    )

    return errors


def _numeric_map(
    errors: list[str], value: Any, *, path: str, require_nonempty: bool
) -> dict[str, float] | None:
    if not isinstance(value, dict):
        if require_nonempty:
            errors.append(f"{path} must be a non-empty object.")
        return None
    if not value:
        if require_nonempty:
            errors.append(f"{path} must be a non-empty object.")
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{path} keys must be non-empty strings.")
            continue
        numeric = _finite_number(raw)
        if numeric is None:
            errors.append(f"{path}.{key} must be a finite number.")
            continue
        result[key] = numeric
    return result


def _rmt_sources(
    rmt: Mapping[str, Any],
    inventory: Sequence[tuple[str, dict[str, Any], str]],
) -> list[tuple[str, Mapping[str, Any]]]:
    sources: list[tuple[str, Mapping[str, Any]]] = [("rmt", rmt)]
    for guard_name, entry, source in inventory:
        if guard_name != "rmt":
            continue
        metrics = _mapping(entry.get("metrics"))
        if metrics is not None:
            sources.append((f"{source}.metrics", metrics))
    return sources


def _rmt_evaluation_errors(
    rmt: Mapping[str, Any],
    sources: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    require_complete: bool,
    enforce_outcome: bool,
) -> list[str]:
    errors: list[str] = []
    evaluated = rmt.get("evaluated")
    if "evaluated" not in rmt:
        if require_complete:
            errors.append("rmt.evaluated is required for strict assurance.")
    elif not isinstance(evaluated, bool):
        errors.append("rmt.evaluated must be a boolean.")
    elif require_complete and evaluated is not True:
        errors.append("rmt.evaluated must be true for strict assurance.")

    stable = _consistent_bool(
        errors,
        _field_values(sources, "stable"),
        required_path="rmt.stable" if require_complete else None,
    )
    if stable is False and enforce_outcome:
        errors.append("rmt.stable is false.")

    violation_values = _field_values(sources, "epsilon_violations")
    if not violation_values and require_complete:
        errors.append("rmt.epsilon_violations is required for strict assurance.")
    for path, violations in violation_values:
        if not isinstance(violations, list):
            errors.append(f"{path} must be an array.")
        elif violations and enforce_outcome:
            errors.append(f"{path} must be empty; epsilon violations were recorded.")
    return errors


def _rmt_acceptance_errors(
    edge_base: Mapping[str, float],
    edge_cur: Mapping[str, float],
    epsilon_map: Mapping[str, float] | None,
    epsilon_default: float | None,
    *,
    require_complete: bool,
    enforce_outcome: bool,
) -> list[str]:
    errors: list[str] = []
    if set(edge_base) != set(edge_cur):
        errors.append("rmt edge-risk baseline/current family sets must match exactly.")
    positive_baselines = 0
    for family in sorted(set(edge_base) & set(edge_cur)):
        base = edge_base[family]
        current = edge_cur[family]
        if base <= 0.0:
            continue
        positive_baselines += 1
        epsilon = (
            epsilon_map.get(family)
            if epsilon_map is not None and family in epsilon_map
            else epsilon_default
        )
        if epsilon is None or epsilon < 0.0:
            errors.append(f"rmt epsilon for family {family!r} is unavailable.")
            continue
        allowed = (1.0 + epsilon) * base
        if current > allowed and enforce_outcome:
            errors.append(
                f"rmt acceptance inequality failed for {family}: {current} > {allowed}."
            )
    if require_complete and positive_baselines == 0:
        errors.append(
            "rmt requires at least one positive baseline edge-risk measurement."
        )
    return errors


def _rmt_family_detail_errors(
    families: Mapping[str, Any],
    edge_base: Mapping[str, float],
    edge_cur: Mapping[str, float],
    epsilon_map: Mapping[str, float] | None,
    epsilon_default: float | None,
) -> list[str]:
    errors: list[str] = []
    for family, details in families.items():
        if not isinstance(details, dict):
            errors.append(f"rmt.families.{family} must be an object.")
            continue
        if family not in edge_base or family not in edge_cur:
            errors.append(f"rmt.families.{family} is absent from the edge-risk maps.")
            continue
        base = edge_base[family]
        current = edge_cur[family]
        epsilon = (
            epsilon_map.get(family)
            if epsilon_map is not None and family in epsilon_map
            else epsilon_default
        )
        expected_values = {
            "edge_base": base,
            "edge_cur": current,
            "epsilon": epsilon,
            "allowed": (
                (1.0 + epsilon) * base if epsilon is not None and base > 0 else None
            ),
            "ratio": (current / base if base > 0 else None),
            "delta": ((current / base) - 1.0 if base > 0 else None),
        }
        for key, expected in expected_values.items():
            if key not in details or expected is None:
                continue
            observed = _finite_number(details.get(key))
            if observed is None:
                errors.append(f"rmt.families.{family}.{key} must be finite.")
            elif not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-12):
                errors.append(
                    f"rmt.families.{family}.{key} disagrees with recomputation."
                )
    return errors


def _rmt_errors(
    report: Mapping[str, Any],
    inventory: Sequence[tuple[str, dict[str, Any], str]],
    *,
    require_complete: bool,
    enforce_outcome: bool = True,
) -> list[str]:
    rmt = _mapping(report.get("rmt"))
    if rmt is None:
        return []
    errors = _rmt_evaluation_errors(
        rmt,
        _rmt_sources(rmt, inventory),
        require_complete=require_complete,
        enforce_outcome=enforce_outcome,
    )

    edge_base = _numeric_map(
        errors,
        rmt.get("edge_risk_by_family_base"),
        path="rmt.edge_risk_by_family_base",
        require_nonempty=require_complete,
    )
    edge_cur = _numeric_map(
        errors,
        rmt.get("edge_risk_by_family"),
        path="rmt.edge_risk_by_family",
        require_nonempty=require_complete,
    )
    epsilon_map = _numeric_map(
        errors,
        rmt.get("epsilon_by_family"),
        path="rmt.epsilon_by_family",
        require_nonempty=require_complete,
    )
    epsilon_default = _finite_number(rmt.get("epsilon_default"))
    if "epsilon_default" not in rmt:
        if require_complete:
            errors.append("rmt.epsilon_default is required for strict assurance.")
    elif epsilon_default is None or epsilon_default < 0.0:
        errors.append("rmt.epsilon_default must be a finite non-negative number.")
        epsilon_default = None

    if edge_base is not None and edge_cur is not None:
        errors.extend(
            _rmt_acceptance_errors(
                edge_base,
                edge_cur,
                epsilon_map,
                epsilon_default,
                require_complete=require_complete,
                enforce_outcome=enforce_outcome,
            )
        )

    families = rmt.get("families")
    if require_complete and not isinstance(families, dict):
        errors.append("rmt.families is required for strict assurance.")
    elif isinstance(families, dict) and edge_base is not None and edge_cur is not None:
        errors.extend(
            _rmt_family_detail_errors(
                families, edge_base, edge_cur, epsilon_map, epsilon_default
            )
        )
    return errors
