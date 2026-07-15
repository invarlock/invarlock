"""Strict reconciliation of retained spectral module inventories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .assurance_spectral_replay_common import (
    _mapping,
    _nonnegative_int,
)

_EXCLUSION_REASONS = {
    "include_pattern_miss",
    "exclude_pattern_match",
    "scope_mismatch",
    "missing_weight",
    "non_tensor_weight",
    "non_matrix_weight",
    "not_selected_by_adapter",
    "parameter_alias",
    "quantized_weight_without_dense_view",
    "non_finite_weight",
    "estimator_error",
    "non_finite_estimate",
}


def _module_list(errors: list[str], value: Any, path: str) -> list[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{path} must be an array of module-name strings.")
        return None
    if len(value) != len(set(value)) or value != sorted(value):
        errors.append(f"{path} must be sorted and contain unique module names.")
        return None
    return list(value)


def _excluded_module_name(
    errors: list[str], item_raw: Any, item_path: str
) -> str | None:
    item = _mapping(item_raw)
    if item is None:
        errors.append(f"{item_path} must be an object.")
        return None
    module = item.get("module")
    stage = item.get("stage")
    reason = item.get("reason")
    if not isinstance(module, str):
        errors.append(f"{item_path}.module must be a string.")
        return None
    if stage not in {"selection", "measurement"}:
        errors.append(f"{item_path}.stage is not a recognized exclusion stage.")
    elif stage == "measurement":
        errors.append(
            f"{item_path} records an eligible module without a valid measurement."
        )
    if reason not in _EXCLUSION_REASONS:
        errors.append(f"{item_path}.reason is not a recognized typed reason.")
    if reason == "parameter_alias" and not isinstance(item.get("alias_of"), str):
        errors.append(f"{item_path}.alias_of must bind the primary module.")
    return module


def replay_measurement_inventory(
    errors: list[str],
    entry: Mapping[str, Any],
    source: str,
    *,
    baseline_modules: set[str],
    final_modules: set[str],
) -> dict[str, dict[str, Any]] | None:
    raw_inventory = _mapping(entry.get("measurement_inventory"))
    if raw_inventory is None or not raw_inventory:
        errors.append(f"{source}.measurement_inventory must be a non-empty object.")
        return None
    inventories: dict[str, dict[str, Any]] = {}
    reference_enumerated: set[str] | None = None
    reference_eligible: set[str] | None = None
    for phase, raw in raw_inventory.items():
        path = f"{source}.measurement_inventory.{phase}"
        if not isinstance(phase, str) or not phase:
            errors.append(
                f"{source}.measurement_inventory phase names must be non-empty."
            )
            continue
        inventory = _mapping(raw)
        if inventory is None:
            errors.append(f"{path} must be an object.")
            continue
        if inventory.get("schema_version") != 1 or inventory.get("phase") != phase:
            errors.append(f"{path} has an invalid schema_version or phase binding.")
        enumerated = _module_list(
            errors, inventory.get("enumerated_modules"), f"{path}.enumerated_modules"
        )
        eligible = _module_list(
            errors, inventory.get("eligible_modules"), f"{path}.eligible_modules"
        )
        measured = _module_list(
            errors, inventory.get("measured_modules"), f"{path}.measured_modules"
        )
        identity_changed = _module_list(
            errors,
            inventory.get("identity_changed_modules"),
            f"{path}.identity_changed_modules",
        )
        if identity_changed:
            errors.append(f"{path} records live module or weight identity changes.")
        discovery_errors = _module_list(
            errors,
            inventory.get("discovery_errors"),
            f"{path}.discovery_errors",
        )
        if discovery_errors:
            errors.append(f"{path} records incomplete adapter module discovery.")
        excluded_raw = inventory.get("excluded_modules")
        if not isinstance(excluded_raw, list):
            errors.append(f"{path}.excluded_modules must be an array.")
            continue
        excluded_names: list[str] = []
        for index, item_raw in enumerate(excluded_raw):
            item_path = f"{path}.excluded_modules[{index}]"
            module = _excluded_module_name(errors, item_raw, item_path)
            if module is not None:
                excluded_names.append(module)
        if len(excluded_names) != len(set(excluded_names)) or excluded_names != sorted(
            excluded_names
        ):
            errors.append(
                f"{path}.excluded_modules must be sorted and unique by module."
            )
        if enumerated is None or eligible is None or measured is None:
            continue
        enumerated_set = set(enumerated)
        eligible_set = set(eligible)
        measured_set = set(measured)
        excluded_set = set(excluded_names)
        if measured_set & excluded_set or measured_set | excluded_set != enumerated_set:
            errors.append(
                f"{path} measured/excluded modules must exactly partition enumeration."
            )
        if not measured_set <= eligible_set or not eligible_set <= enumerated_set:
            errors.append(f"{path} eligibility and measurement sets are inconsistent.")
        for item_raw in excluded_raw:
            item = _mapping(item_raw)
            if item is None or not isinstance(item.get("module"), str):
                continue
            module = str(item["module"])
            if item.get("stage") == "selection" and module in eligible_set:
                errors.append(
                    f"{path} selection exclusion {module!r} is marked eligible."
                )
            if item.get("stage") == "measurement" and module not in eligible_set:
                errors.append(
                    f"{path} measurement exclusion {module!r} is not marked eligible."
                )
            if (
                item.get("reason") == "parameter_alias"
                and item.get("alias_of") not in eligible_set
            ):
                errors.append(
                    f"{path} parameter alias {module!r} lacks an eligible primary."
                )
        count_values = {
            "enumerated_count": len(enumerated_set),
            "eligible_count": len(eligible_set),
            "measured_count": len(measured_set),
            "excluded_count": len(excluded_set),
            "identity_changed_count": len(identity_changed or []),
            "discovery_error_count": len(discovery_errors or []),
        }
        for field, expected in count_values.items():
            if _nonnegative_int(inventory.get(field)) != expected:
                errors.append(f"{path}.{field} disagrees with the retained inventory.")
        if reference_enumerated is None:
            reference_enumerated = enumerated_set
            reference_eligible = eligible_set
        else:
            if enumerated_set != reference_enumerated:
                errors.append(
                    f"{path}.enumerated_modules disagrees across measurement phases."
                )
            if eligible_set != reference_eligible:
                errors.append(
                    f"{path}.eligible_modules disagrees across measurement phases."
                )
        inventories[phase] = {
            "enumerated": enumerated_set,
            "eligible": eligible_set,
            "measured": measured_set,
        }
    prepare = inventories.get("prepare")
    if prepare is None:
        errors.append(f"{source}.measurement_inventory.prepare is required.")
    elif prepare["measured"] != baseline_modules:
        errors.append(
            f"{source}.measurement_inventory.prepare.measured_modules disagrees "
            "with baseline module sigmas."
        )
    ledger = _mapping(entry.get("correction_ledger"))
    ledger_phase = str(ledger.get("phase") or "") if ledger is not None else ""
    final_phase = (
        f"{ledger_phase}_post_correction"
        if f"{ledger_phase}_post_correction" in inventories
        else ledger_phase
    )
    if not final_phase or final_phase not in inventories:
        errors.append(
            f"{source}.measurement_inventory lacks the correction-ledger final phase."
        )
    elif inventories[final_phase]["measured"] != final_modules:
        errors.append(
            f"{source}.measurement_inventory.{final_phase}.measured_modules "
            "disagrees with final_metrics."
        )
    return inventories


__all__ = ["replay_measurement_inventory"]
