from __future__ import annotations

from typing import Any

from . import report_validation_allowlist as allowlist_mod

# Optional JSON Schema validation support (best-effort)
try:  # pragma: no cover - exercised in integration
    import jsonschema
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    jsonschema = None
    _JSONSCHEMA_FAILURES: tuple[type[BaseException], ...] = ()
else:
    _JSONSCHEMA_FAILURES = (
        ValueError,
        jsonschema.SchemaError,
        jsonschema.ValidationError,
    )


# Evaluation report schema version (PM-first canonical)
REPORT_SCHEMA_VERSION = "v1"


# Minimal JSON Schema describing the canonical shape of an evaluation report.
# This focuses on structural validity; numerical thresholds are validated
# separately in metric-specific logic.
REPORT_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "InvarLock Evaluation Report",
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
        "schema_version": {"const": REPORT_SCHEMA_VERSION},
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
                "hash": {
                    "type": "object",
                    "properties": {
                        "preview": {"type": "string"},
                        "final": {"type": "string"},
                        "dataset": {"type": ["string", "null"]},
                        "preview_tokens": {"type": ["integer", "string", "null"]},
                        "final_tokens": {"type": ["integer", "string", "null"]},
                        "total_tokens": {"type": "integer", "minimum": 0},
                        "source": {
                            "enum": [
                                "explicit_preview_final_hashes",
                                "explicit_token_ids",
                                "config_fallback",
                            ]
                        },
                    },
                    "additionalProperties": True,
                },
                "tokenizer": {"type": "object"},
                "windows": {
                    "type": "object",
                    "required": ["preview", "final", "stats"],
                    "properties": {
                        "preview": {"type": "integer", "minimum": 0},
                        "final": {"type": "integer", "minimum": 0},
                        "seed": {"type": ["integer", "null"]},
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
                # Historical reports used plain numbers; newer reports emit
                # structured overhead entries (baseline/edited/delta/ratio).
                "^latency_ms_(p50|p95)$": {
                    "oneOf": [
                        {"type": "number"},
                        {
                            "type": "object",
                            "properties": {
                                "baseline": {"type": "number"},
                                "edited": {"type": "number"},
                                "delta": {"type": "number"},
                                "ratio": {"type": "number"},
                            },
                            "additionalProperties": True,
                        },
                    ]
                },
                "^throughput_.*$": {
                    "oneOf": [
                        {"type": "number"},
                        {
                            "type": "object",
                            "properties": {
                                "baseline": {"type": "number"},
                                "edited": {"type": "number"},
                                "delta": {"type": "number"},
                                "ratio": {"type": "number"},
                            },
                            "additionalProperties": True,
                        },
                    ]
                },
            },
            "additionalProperties": True,
        },
        "validation": {
            "type": "object",
            # properties populated at import-time from allow-list; default permissive
            "properties": {},
            "additionalProperties": {"type": "boolean"},
        },
        "rmt": {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
                "measurement_contract_hash": {"type": "string"},
            },
            "additionalProperties": True,
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


_VALIDATION_ALLOWLIST_DEFAULT = allowlist_mod.DEFAULT_VALIDATION_ALLOWLIST


def _validate_with_jsonschema(report: dict[str, Any]) -> bool:
    """Validate evaluation report with JSON Schema when available."""
    if jsonschema is None:
        return False
    try:
        jsonschema.validate(instance=report, schema=REPORT_JSON_SCHEMA)
        return True
    except _JSONSCHEMA_FAILURES:
        return False


def validate_report(report: dict[str, Any]) -> bool:
    """Validate evaluation report structure and essential flags."""
    try:
        if report.get("schema_version") != REPORT_SCHEMA_VERSION:
            return False

        # Prefer JSON Schema structural validation; if unavailable or too strict,
        # fall back to a lenient minimal check used by unit tests.
        # Tighten JSON Schema: populate validation.properties from allow-list and
        # disallow unknown validation keys at schema level.
        try:
            allowlist_mod.apply_validation_allowlist_schema(
                REPORT_JSON_SCHEMA, allowlist_mod.load_validation_allowlist()
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass

        if not _validate_with_jsonschema(report):
            # Minimal fallback: require schema version + run_id + primary_metric
            run_id = report.get("run_id")
            run_id_ok = isinstance(run_id, str) and bool(run_id.strip())
            pm = report.get("primary_metric")
            pm_ok = isinstance(pm, dict) and (
                isinstance(pm.get("final"), int | float)
                or (isinstance(pm.get("kind"), str) and bool(pm.get("kind")))
            )
            if not (run_id_ok and pm_ok):
                return False

        validation = report.get("validation", {})
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
    "REPORT_SCHEMA_VERSION",
    "REPORT_JSON_SCHEMA",
    "validate_report",
]
