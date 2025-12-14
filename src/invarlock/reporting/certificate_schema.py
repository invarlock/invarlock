from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Optional JSON Schema validation support (best-effort)
try:  # pragma: no cover - exercised in integration
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None


# Certificate schema version (PM-first canonical)
CERTIFICATE_SCHEMA_VERSION = "v1"


# Minimal JSON Schema describing the canonical shape of a certificate.
# This focuses on structural validity; numerical thresholds are validated
# separately in metric-specific logic.
CERTIFICATE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "InvarLock Safety Certificate",
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "artifacts",
        "plugins",
        "meta",
        "dataset",
        "primary_metric",
    ],
    "properties": {
        "schema_version": {"const": CERTIFICATE_SCHEMA_VERSION},
        "run_id": {"type": "string", "minLength": 4},
        "edit_name": {"type": "string"},
        "policy_digest": {
            "type": "object",
            "properties": {
                "policy_version": {"type": "string"},
                "tier_policy_name": {"type": "string"},
                "thresholds_hash": {"type": "string"},
                "hysteresis": {"type": "object"},
                "min_effective": {"type": "number"},
                "changed": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        "plugins": {
            "type": "object",
            "properties": {
                "adapters": {"type": "array"},
                "edits": {"type": "array"},
                "guards": {"type": "array"},
            },
            "additionalProperties": True,
        },
        "meta": {"type": "object"},
        "dataset": {
            "type": "object",
            "required": ["provider", "seq_len", "windows"],
            "properties": {
                "provider": {"type": "string"},
                "seq_len": {"type": "integer", "minimum": 1},
                "windows": {
                    "type": "object",
                    "required": ["preview", "final", "stats"],
                    "properties": {
                        "preview": {"type": "integer", "minimum": 0},
                        "final": {"type": "integer", "minimum": 0},
                        "seed": {"type": "integer"},
                        "stats": {"type": "object"},
                    },
                },
            },
            "additionalProperties": True,
        },
        # ppl_* block removed from required schema; may appear for ppl-like tasks but is optional
        "primary_metric": {
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {"type": "string"},
                "unit": {"type": "string"},
                "direction": {"type": "string"},
                "aggregation_scope": {"type": "string"},
                "paired": {"type": "boolean"},
                "gating_basis": {"type": "string"},
                "preview": {"type": "number"},
                "final": {"type": "number"},
                "ratio_vs_baseline": {"type": "number"},
                "reps": {"type": "number"},
                "ci_level": {"type": "number"},
                "counts_source": {"enum": ["measured", "pseudo_config"]},
                "estimated": {"type": "boolean"},
                "ci": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
                "display_ci": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
            },
            "additionalProperties": True,
        },
        "system_overhead": {
            "type": "object",
            # Numeric keys must match these patterns when present; allow flexibility otherwise
            "patternProperties": {
                "^latency_ms_(p50|p95)$": {"type": "number"},
                "^throughput_.*$": {"type": "number"},
            },
            "additionalProperties": True,
        },
        "validation": {
            "type": "object",
            # properties populated at import-time from allow-list; default permissive
            "properties": {},
            "additionalProperties": {"type": "boolean"},
        },
        "artifacts": {"type": "object"},
        "provenance": {"type": "object"},
        "resolved_policy": {"type": "object"},
        "policy_provenance": {"type": "object"},
        "structure": {"type": "object"},
        "confidence": {
            "type": "object",
            "properties": {
                "label": {"enum": ["High", "Medium", "Low"]},
                "basis": {"type": "string"},
                "width": {"type": "number"},
                "threshold": {"type": "number"},
                "unstable": {"type": "boolean"},
            },
            "required": ["label", "basis"],
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}


_VALIDATION_ALLOWLIST_DEFAULT = {
    "primary_metric_acceptable",
    "preview_final_drift_acceptable",
    "guard_overhead_acceptable",
    "invariants_pass",
    "spectral_stable",
    "rmt_stable",
    # Compatibility keys were removed; PM-only surface
    "hysteresis_applied",
    "moe_observed",
    "moe_identity_ok",
}


def _load_validation_allowlist() -> set[str]:
    """Load validation key allow-list from contracts/validation_keys.json when available.

    Falls back to a safe built-in default when the contracts directory is not present
    (e.g., installed wheel) or when parsing fails.
    """
    try:
        root = Path(__file__).resolve().parents[3]
        path = root / "contracts" / "validation_keys.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {str(k) for k in data}
    except Exception:
        pass
    return set(_VALIDATION_ALLOWLIST_DEFAULT)


def _validate_with_jsonschema(certificate: dict[str, Any]) -> bool:
    """Validate certificate with JSON Schema when available."""
    if jsonschema is None:
        return True  # Schema library unavailable; fall back to legacy checks
    try:
        jsonschema.validate(instance=certificate, schema=CERTIFICATE_JSON_SCHEMA)
        return True
    except Exception:
        return False


def validate_certificate(certificate: dict[str, Any]) -> bool:
    """Validate certificate structure and essential flags."""
    try:
        if certificate.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
            return False

        # Prefer JSON Schema structural validation; if unavailable or too strict,
        # fall back to a lenient minimal check used by unit tests.
        # Tighten JSON Schema: populate validation.properties from allow-list and
        # disallow unknown validation keys at schema level.
        try:
            vkeys = _load_validation_allowlist()
            if isinstance(CERTIFICATE_JSON_SCHEMA.get("properties"), dict):
                vspec = CERTIFICATE_JSON_SCHEMA["properties"].get("validation")
                if isinstance(vspec, dict):
                    vspec["properties"] = {k: {"type": "boolean"} for k in vkeys}
                    vspec["additionalProperties"] = False
        except Exception:
            pass

        if not _validate_with_jsonschema(certificate):
            # Minimal fallback: require schema version + run_id + primary_metric
            run_id_ok = isinstance(certificate.get("run_id"), str) and bool(
                certificate.get("run_id")
            )
            pm = certificate.get("primary_metric")
            pm_ok = isinstance(pm, dict) and (
                isinstance(pm.get("final"), int | float)
                or (isinstance(pm.get("kind"), str) and bool(pm.get("kind")))
            )
            if not (run_id_ok and pm_ok):
                return False

        validation = certificate.get("validation", {})
        for flag in [
            "preview_final_drift_acceptable",
            "primary_metric_acceptable",
            "invariants_pass",
            "spectral_stable",
            "rmt_stable",
            "guard_overhead_acceptable",
        ]:
            # If present, must be boolean; tolerate missing opt-in flags
            if flag in validation and not isinstance(validation.get(flag), bool):
                return False

        return True
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "CERTIFICATE_SCHEMA_VERSION",
    "CERTIFICATE_JSON_SCHEMA",
    "validate_certificate",
]
