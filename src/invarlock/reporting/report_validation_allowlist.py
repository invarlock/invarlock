from __future__ import annotations

from typing import Any

from invarlock.public_contracts import load_json_contract


class ValidationAllowlistContractError(RuntimeError):
    """Raised when the public validation allow-list contract cannot be trusted."""


DEFAULT_VALIDATION_ALLOWLIST = {
    "primary_metric_acceptable",
    "primary_metric_tail_acceptable",
    "preview_final_drift_acceptable",
    "guard_overhead_acceptable",
    "invariants_pass",
    "spectral_stable",
    "rmt_stable",
    "hysteresis_applied",
    "moe_observed",
    "moe_identity_ok",
}


def _load_validation_allowlist_default() -> set[str]:
    return set(DEFAULT_VALIDATION_ALLOWLIST)


def _normalize_validation_allowlist_payload(data: object) -> set[str]:
    if not isinstance(data, list):
        raise ValidationAllowlistContractError(
            "Validation key contract must be a non-empty JSON array of strings."
        )
    keys = {str(key).strip() for key in data if isinstance(key, str) and key.strip()}
    if not keys:
        raise ValidationAllowlistContractError(
            "Validation key contract must declare at least one concrete key."
        )
    return keys


def load_validation_allowlist_strict() -> set[str]:
    try:
        data = load_json_contract("validation_keys.json")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationAllowlistContractError(
            "Failed to load validation key contract from contracts/validation_keys.json"
        ) from exc
    return _normalize_validation_allowlist_payload(data)


def load_validation_allowlist_with_source() -> tuple[set[str], str]:
    try:
        return load_validation_allowlist_strict(), "contracts"
    except ValidationAllowlistContractError as exc:
        if exc.__cause__ is not None:
            return _load_validation_allowlist_default(), "fallback:load-error"
        return (
            _load_validation_allowlist_default(),
            "fallback:invalid-contract-validation-keys",
        )


def load_validation_allowlist() -> set[str]:
    keys, _ = load_validation_allowlist_with_source()
    return keys


def apply_validation_allowlist_schema(
    report_json_schema: dict[str, Any], validation_keys: set[str]
) -> None:
    schema_properties = report_json_schema.get("properties")
    if not isinstance(schema_properties, dict):
        raise RuntimeError(
            "REPORT_JSON_SCHEMA.properties must be a mapping to enforce validation "
            "allow-list constraints."
        )
    validation_spec = schema_properties.get("validation")
    if not isinstance(validation_spec, dict):
        raise RuntimeError(
            "REPORT_JSON_SCHEMA.properties.validation must be a mapping to enforce "
            "validation allow-list constraints."
        )
    validation_spec["properties"] = {
        key: {"type": "boolean"} for key in validation_keys
    }
    validation_spec["additionalProperties"] = False
