"""Fail-closed reconciliation facade for assurance-critical guard evidence.

The report contains both derived guard summaries and the original ordered guard
inventory. Strict assurance validates and reconciles those facts instead of
accepting producer-supplied outcome booleans at face value. Guard-family logic
lives in focused sibling modules; this facade preserves the public API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .assurance_guard_validation_common import (
    _GUARD_NAMES,
    _dedupe,
    _guard_base_name,
    _guard_inventory,
    _mapping,
    _normalized_token,
    _validate_guard_outcome,
)
from .assurance_guard_validation_matrix import _rmt_errors, _spectral_errors
from .assurance_guard_validation_raw import raw_guard_evidence_errors
from .assurance_guard_validation_runtime import (
    _guard_metric_impact_errors,
    _invariants_errors,
    _variance_errors,
)
from .assurance_guard_validation_variance_evidence import _variance_inventory_errors

_VALIDATION_GUARD_KEYS = (
    "invariants_pass",
    "spectral_stable",
    "rmt_stable",
    "guard_metric_impact_acceptable",
)
_STRICT_VALIDATION_KEYS = (
    "preview_final_drift_acceptable",
    "primary_metric_acceptable",
    *_VALIDATION_GUARD_KEYS,
)


def _validation_errors(
    report: Mapping[str, Any], *, require_complete: bool
) -> list[str]:
    validation = report.get("validation")
    if not isinstance(validation, dict):
        return (
            ["strict assurance requires a validation object."]
            if require_complete
            else []
        )
    errors: list[str] = []
    keys = _STRICT_VALIDATION_KEYS if require_complete else _VALIDATION_GUARD_KEYS
    for key in keys:
        if key not in validation:
            if require_complete:
                errors.append(f"validation.{key} is required for strict assurance.")
            continue
        value = validation.get(key)
        if not isinstance(value, bool):
            errors.append(f"validation.{key} must be a boolean.")
        elif value is not True:
            errors.append(f"validation.{key} is false.")
    for key in ("guard_warning_policy_acceptable", "primary_metric_tail_acceptable"):
        if key in validation:
            value = validation.get(key)
            if not isinstance(value, bool):
                errors.append(f"validation.{key} must be a boolean.")
            elif value is False:
                errors.append(f"validation.{key} is false.")
    return errors


def guard_evidence_policy_errors(
    report: Mapping[str, Any], *, require_complete: bool
) -> list[str]:
    """Validate raw guard facts and their submitted outcome mirrors.

    ``require_complete`` selects the strict-assurance contract. CI/release
    verification uses the same function with ``False`` to reject explicit or
    contradictory failures without retroactively requiring optional evidence.
    """

    errors: list[str] = []
    inventory = _guard_inventory(report)
    for guard_name in _GUARD_NAMES:
        errors.extend(
            _validate_guard_outcome(
                guard_name,
                report.get(guard_name),
                source=guard_name,
                require_complete=require_complete,
            )
        )
    for guard_name, entry, source in inventory:
        errors.extend(
            _validate_guard_outcome(
                guard_name,
                entry,
                source=source,
                require_complete=require_complete,
            )
        )

    errors.extend(
        _spectral_errors(report, inventory, require_complete=require_complete)
    )
    errors.extend(_rmt_errors(report, inventory, require_complete=require_complete))
    errors.extend(_invariants_errors(report, require_complete=require_complete))
    errors.extend(_variance_errors(report, require_complete=require_complete))
    variance = _mapping(report.get("variance"))
    if variance is not None:
        errors.extend(
            _variance_inventory_errors(
                report,
                variance,
                inventory,
                require_complete=require_complete,
            )
        )
    errors.extend(
        _guard_metric_impact_errors(report, require_complete=require_complete)
    )
    errors.extend(
        raw_guard_evidence_errors(
            report,
            inventory,
            require_complete=require_complete,
        )
    )
    errors.extend(_validation_errors(report, require_complete=require_complete))
    return _dedupe(errors)


def _chain_from_sequence(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    names: list[str] = []
    for item in value:
        raw_name = item.get("name") if isinstance(item, dict) else item
        guard_name = _guard_base_name(raw_name)
        if guard_name is None:
            return None
        names.append(guard_name)
    return tuple(names)


def strict_guard_chain_errors(
    report: Mapping[str, Any],
    *,
    canonical_chain: Sequence[str],
    require_assurance: bool = True,
) -> list[str]:
    """Require and cross-check every in-report guard-chain representation."""

    expected = tuple(canonical_chain)
    errors: list[str] = []
    assurance = _mapping(report.get("assurance"))
    representations: list[tuple[str, Any]] = []
    if assurance is None:
        if require_assurance:
            errors.append("strict assurance report missing assurance section.")
    else:
        representations.extend(
            [
                (
                    "assurance.canonical_guard_chain",
                    assurance.get("canonical_guard_chain"),
                ),
                (
                    "assurance.guard_chain_observed",
                    assurance.get("guard_chain_observed"),
                ),
            ]
        )
    plugins = _mapping(report.get("plugins"))
    representations.append(
        ("plugins.guards", plugins.get("guards") if plugins is not None else None)
    )
    representations.append(("guards", report.get("guards")))
    context = _mapping(report.get("context"))
    representations.append(
        (
            "context.guard_chain_observed",
            context.get("guard_chain_observed") if context is not None else None,
        )
    )

    for path, raw in representations:
        if raw is None:
            errors.append(f"strict assurance requires {path} guard chain evidence.")
            continue
        observed = _chain_from_sequence(raw)
        if observed is None:
            errors.append(f"{path} must be an ordered guard chain array.")
        elif observed != expected:
            errors.append(
                f"{path} guard chain must exactly match the canonical guard chain."
            )

    inventory = report.get("guards")
    if isinstance(inventory, list) and len(inventory) == len(expected):
        first = inventory[0] if isinstance(inventory[0], dict) else {}
        last = inventory[-1] if isinstance(inventory[-1], dict) else {}
        if _normalized_token(str(first.get("stage", ""))) != "pre":
            errors.append("guards[0].stage must be pre for strict assurance.")
        if _normalized_token(str(last.get("stage", ""))) != "post":
            errors.append("the final invariants guard stage must be post.")
    return _dedupe(errors)


__all__ = [
    "guard_evidence_policy_errors",
    "strict_guard_chain_errors",
]
