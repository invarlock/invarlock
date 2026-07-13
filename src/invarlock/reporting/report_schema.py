from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any

from invarlock.core.metric_kind_contract import (
    MetricKindContractError,
    load_metric_kind_catalog,
    normalize_metric_kind,
)
from invarlock.public_contracts import REPORT_SCHEMA_VERSION, load_json_contract


class ValidationAllowlistContractError(RuntimeError):
    """Raised when the public validation allow-list contract cannot be trusted."""


DEFAULT_VALIDATION_ALLOWLIST = {
    "guard_warning_policy_acceptable",
    "guard_warnings_present",
    "primary_metric_acceptable",
    "primary_metric_tail_acceptable",
    "preview_final_drift_acceptable",
    "guard_metric_impact_acceptable",
    "invariants_pass",
    "spectral_stable",
    "rmt_stable",
    "hysteresis_applied",
    "moe_observed",
    "moe_identity_ok",
}


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
    if load_json_contract is _DEFAULT_JSON_CONTRACT_LOADER:
        return set(_load_default_validation_allowlist())
    try:
        data = load_json_contract("validation_keys.json")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationAllowlistContractError(
            "Failed to load validation key contract from contracts/validation_keys.json"
        ) from exc
    return _normalize_validation_allowlist_payload(data)


_DEFAULT_JSON_CONTRACT_LOADER = load_json_contract


@lru_cache(maxsize=1)
def _load_default_validation_allowlist() -> tuple[str, ...]:
    try:
        data = _DEFAULT_JSON_CONTRACT_LOADER("validation_keys.json")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationAllowlistContractError(
            "Failed to load validation key contract from contracts/validation_keys.json"
        ) from exc
    return tuple(sorted(_normalize_validation_allowlist_payload(data)))


def load_validation_allowlist() -> set[str]:
    return load_validation_allowlist_strict()


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


# JSON Schema validation is required for canonical report acceptance.
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


_PREVIEW_FINAL_SLICE_DELTA_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "mean",
        "ci",
        "basis",
        "paired",
        "ci_method",
        "preview_windows",
        "final_windows",
        "degenerate",
    ],
    "properties": {
        "mean": {"type": "number"},
        "ci": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number"},
        },
        "basis": {"const": "independent_disjoint_slices"},
        "paired": {"const": False},
        "ci_method": {"enum": ["independent_percentile_delta_log", "none"]},
        "ci_reason": {"type": ["string", "null"]},
        "preview_windows": {"type": "integer", "minimum": 0},
        "final_windows": {"type": "integer", "minimum": 0},
        "degenerate": {"type": "boolean"},
        "degenerate_reason": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

_SYSTEM_OVERHEAD_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "baseline": {"type": "number"},
        "edited": {"type": "number"},
        "delta": {"type": "number"},
        "ratio": {"type": "number"},
    },
    "required": ["edited"],
    "allOf": [
        {
            "if": {"required": ["baseline"]},
            "then": {"required": ["delta"]},
        }
    ],
    "additionalProperties": False,
}

_GUARD_METRIC_IMPACT_DIAGNOSTIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["kind", "severity", "message", "details"],
    "properties": {
        "kind": {"type": "string", "minLength": 1},
        "severity": {"type": "string", "minLength": 1},
        "message": {"type": "string", "minLength": 1},
        "details": {"type": "object"},
    },
    "additionalProperties": False,
}

_GUARD_METRIC_IMPACT_CHECKS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "boolean"},
}

_GUARD_METRIC_IMPACT_EVALUATED_CHECKS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "metric_kind_matches",
        "measurements_valid",
        "guard_metric_impact",
        "arm_facts_replay",
    ],
    "additionalProperties": {"type": "boolean"},
}

_GUARD_METRIC_IMPACT_ACCURACY_FACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["correct", "total", "example_ids_digest"],
    "properties": {
        "correct": {"type": "integer", "minimum": 0},
        "total": {"type": "integer", "minimum": 1},
        "example_ids_digest": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
    },
    "additionalProperties": False,
}

_GUARD_METRIC_IMPACT_PPL_FACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["weighted_logloss_sum", "token_count", "example_ids_digest"],
    "properties": {
        "weighted_logloss_sum": {"type": "number"},
        "token_count": {"type": "integer", "minimum": 1},
        "example_ids_digest": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
    },
    "additionalProperties": False,
}

_GUARD_METRIC_IMPACT_BARE_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["primary_metric", "final", "status"],
    "properties": {
        "primary_metric": {
            "type": "object",
            "required": ["kind", "final"],
            "properties": {
                "kind": {"enum": ["accuracy", "ppl_causal", "ppl_mlm", "ppl_seq2seq"]},
                "final": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "final": {"type": "object"},
        "status": {"enum": ["success", "completed", "ok"]},
    },
    "allOf": [
        {
            "if": {
                "properties": {
                    "primary_metric": {"properties": {"kind": {"const": "accuracy"}}}
                }
            },
            "then": {
                "properties": {
                    "final": {
                        "type": "object",
                        "required": ["correct_total", "total"],
                        "properties": {
                            "correct_total": {"type": "integer", "minimum": 0},
                            "total": {"type": "integer", "minimum": 1},
                            "example_ids": {"type": "array", "minItems": 1},
                        },
                        "additionalProperties": False,
                    }
                }
            },
            "else": {
                "properties": {
                    "final": {
                        "type": "object",
                        "required": ["logloss"],
                        "properties": {
                            "logloss": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "number", "minimum": 0},
                            },
                            "token_counts": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "integer", "minimum": 1},
                            },
                            "masked_token_counts": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "integer", "minimum": 1},
                            },
                            "window_ids": {"type": "array", "minItems": 1},
                        },
                        "anyOf": [
                            {"required": ["token_counts"]},
                            {"required": ["masked_token_counts"]},
                        ],
                        "additionalProperties": False,
                    }
                }
            },
        }
    ],
    "additionalProperties": False,
}

_GUARD_METRIC_IMPACT_EVALUATED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "metric_kind",
        "direction",
        "degradation_basis",
        "bare_value",
        "guarded_value",
        "bare_facts",
        "guarded_facts",
        "bare_report",
        "degradation",
        "degradation_limit",
        "display_value",
        "display_unit",
        "evaluated",
        "passed",
        "checks",
        "diagnostics",
        "source",
        "schedule_digest",
    ],
    "properties": {
        "metric_kind": {"enum": ["accuracy", "ppl_causal", "ppl_mlm", "ppl_seq2seq"]},
        "direction": {"enum": ["lower", "higher"]},
        "degradation_basis": {"enum": ["relative_increase", "absolute_drop"]},
        "bare_value": {"type": "number"},
        "guarded_value": {"type": "number"},
        "bare_facts": {"type": "object"},
        "guarded_facts": {"type": "object"},
        "bare_report": copy.deepcopy(_GUARD_METRIC_IMPACT_BARE_REPORT_SCHEMA),
        "degradation": {"type": "number"},
        "degradation_limit": {"type": "number", "minimum": 0},
        "display_value": {"type": "number"},
        "display_unit": {"enum": ["percent", "percentage_points"]},
        "evaluated": {"const": True},
        "passed": {"type": "boolean"},
        "checks": copy.deepcopy(_GUARD_METRIC_IMPACT_EVALUATED_CHECKS_SCHEMA),
        "diagnostics": {
            "type": "array",
            "items": copy.deepcopy(_GUARD_METRIC_IMPACT_DIAGNOSTIC_SCHEMA),
        },
        "mode": {"const": "bare"},
        "source": {"type": "string", "minLength": 1},
        "schedule_digest": {"type": "string", "pattern": "^[a-f0-9]{32}$"},
    },
    "allOf": [
        {
            "if": {"properties": {"metric_kind": {"const": "accuracy"}}},
            "then": {
                "properties": {
                    "direction": {"const": "higher"},
                    "degradation_basis": {"const": "absolute_drop"},
                    "display_unit": {"const": "percentage_points"},
                    "bare_facts": copy.deepcopy(
                        _GUARD_METRIC_IMPACT_ACCURACY_FACTS_SCHEMA
                    ),
                    "guarded_facts": copy.deepcopy(
                        _GUARD_METRIC_IMPACT_ACCURACY_FACTS_SCHEMA
                    ),
                }
            },
            "else": {
                "properties": {
                    "direction": {"const": "lower"},
                    "degradation_basis": {"const": "relative_increase"},
                    "display_unit": {"const": "percent"},
                    "bare_facts": copy.deepcopy(_GUARD_METRIC_IMPACT_PPL_FACTS_SCHEMA),
                    "guarded_facts": copy.deepcopy(
                        _GUARD_METRIC_IMPACT_PPL_FACTS_SCHEMA
                    ),
                }
            },
        },
        {
            "if": {"properties": {"bare_facts": {"required": ["example_ids_digest"]}}},
            "then": {
                "properties": {"guarded_facts": {"required": ["example_ids_digest"]}}
            },
        },
        {
            "if": {
                "properties": {"guarded_facts": {"required": ["example_ids_digest"]}}
            },
            "then": {
                "properties": {"bare_facts": {"required": ["example_ids_digest"]}}
            },
        },
    ],
    "additionalProperties": False,
}

_GUARD_METRIC_IMPACT_UNEVALUATED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "degradation_limit",
        "evaluated",
        "passed",
        "checks",
        "diagnostics",
        "source",
    ],
    "properties": {
        "degradation_limit": {"type": "number", "minimum": 0},
        "evaluated": {"const": False},
        "passed": {"const": False},
        "checks": copy.deepcopy(_GUARD_METRIC_IMPACT_CHECKS_SCHEMA),
        "diagnostics": {
            "type": "array",
            "items": copy.deepcopy(_GUARD_METRIC_IMPACT_DIAGNOSTIC_SCHEMA),
        },
        "source": {"type": "string", "minLength": 1},
        "schedule_digest": {"type": "string", "pattern": "^[a-f0-9]{32}$"},
        "skipped": {"const": True},
        "skip_reason": {"type": "string", "minLength": 1},
        "mode": {"enum": ["skipped", "unevaluated"]},
    },
    "allOf": [
        {
            "if": {"required": ["skipped"]},
            "then": {
                "required": ["mode", "skip_reason"],
                "properties": {"mode": {"const": "skipped"}},
            },
        }
    ],
    "additionalProperties": False,
}


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
        "run_id": {"type": "string", "minLength": 1},
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
                "dataset_name": {"type": "string", "minLength": 1},
                "config_name": {"type": "string", "minLength": 1},
                "revision": {"type": "string", "minLength": 1},
                "seq_len": {"type": "integer", "minimum": 0},
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
                        "stats": {
                            "type": "object",
                            "properties": {
                                "preview_final_slice_delta_summary": copy.deepcopy(
                                    _PREVIEW_FINAL_SLICE_DELTA_SUMMARY_SCHEMA
                                ),
                            },
                            "additionalProperties": True,
                            "not": {"required": ["paired_delta_summary"]},
                        },
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
                "delta_vs_baseline_pp": {"type": "number"},
                "reps": {"type": "number"},
                "ci_level": {"type": "number"},
                "counts_source": {"enum": ["measured", "pseudo_config"]},
                "estimated": {"type": "boolean"},
                "n_preview": {"type": "integer", "minimum": 0},
                "n_final": {"type": "integer", "minimum": 0},
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
            "allOf": [
                {
                    "if": {"properties": {"kind": {"const": "accuracy"}}},
                    "then": {
                        "required": ["preview", "final", "delta_vs_baseline_pp"],
                        "properties": {
                            "preview": {"type": "number", "minimum": 0, "maximum": 1},
                            "final": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "not": {"required": ["ratio_vs_baseline"]},
                    },
                },
                {
                    "if": {
                        "properties": {
                            "kind": {"enum": ["ppl_causal", "ppl_mlm", "ppl_seq2seq"]}
                        }
                    },
                    "then": {
                        "required": ["preview", "final", "ratio_vs_baseline"],
                        "properties": {
                            "preview": {"type": "number", "minimum": 1},
                            "final": {"type": "number", "minimum": 1},
                            "ratio_vs_baseline": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                            },
                        },
                        "not": {"required": ["delta_vs_baseline_pp"]},
                    },
                },
            ],
        },
        "metrics": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "object",
                    "properties": {
                        "n_correct": {"type": "integer", "minimum": 0},
                        "n_total": {"type": "integer", "minimum": 0},
                        "counts_source": {"enum": ["measured", "pseudo_config"]},
                        "estimated": {"type": "boolean"},
                        "preview": {"type": "object"},
                        "final": {"type": "object"},
                    },
                    "additionalProperties": True,
                },
                "preview_final_slice_delta_summary": copy.deepcopy(
                    _PREVIEW_FINAL_SLICE_DELTA_SUMMARY_SCHEMA
                ),
            },
            "additionalProperties": True,
            "not": {"required": ["paired_delta_summary"]},
        },
        "evaluation_realism": {
            "type": "object",
            "properties": {
                "mode": {
                    "enum": [
                        "generation",
                        "logprob",
                        "teacher_forced",
                        "classification",
                        "benchmark_harness",
                    ]
                },
                "prompt_template_hash": {
                    "type": "string",
                    "pattern": "^sha256:[a-f0-9]{64}$",
                },
                "decoding_config": {"type": "object"},
                "max_tokens": {"type": "integer", "minimum": 0},
                "truncation_policy": {"type": "string", "minLength": 1},
                "dataset_or_task_id": {"type": "string", "minLength": 1},
                "metric_is_generation_realistic": {"type": "boolean"},
                "proxy_metric_warning": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
        "system_overhead": {
            "type": "object",
            "patternProperties": {
                "^latency_ms_(p50|p95)$": copy.deepcopy(_SYSTEM_OVERHEAD_ENTRY_SCHEMA),
                "^throughput_.*$": copy.deepcopy(_SYSTEM_OVERHEAD_ENTRY_SCHEMA),
            },
            "additionalProperties": False,
        },
        "guard_metric_impact": {
            "oneOf": [
                copy.deepcopy(_GUARD_METRIC_IMPACT_EVALUATED_SCHEMA),
                copy.deepcopy(_GUARD_METRIC_IMPACT_UNEVALUATED_SCHEMA),
            ]
        },
        "validation": {
            "type": "object",
            # Properties are populated from the validation-key contract when
            # available. The empty baseline remains fail-closed.
            "properties": {},
            "additionalProperties": False,
        },
        "rmt": {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
                "measurement_contract_hash": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "guard_warnings": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "warning_count": {"type": "integer", "minimum": 0},
                "warnings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "guard": {"type": "string"},
                            "kind": {"type": "string"},
                            "severity": {"type": "string"},
                            "family": {"type": "string"},
                            "module": {"type": "string"},
                            "policy_gate": {"type": "string"},
                            "baseline": {"type": "object"},
                            "subject": {"type": "object"},
                            "message": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                },
            },
            "additionalProperties": True,
        },
        "primary_metric_tail": {
            "type": "object",
            "required": ["mode", "evaluated", "passed"],
            "properties": {
                "evaluated": {"type": "boolean"},
                "passed": {"type": "boolean"},
                "warned": {"type": "boolean"},
                "mode": {"enum": ["off", "warn", "fail"]},
                "policy": {"type": "object"},
                "stats": {"type": "object"},
                "reason": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "artifacts": {"type": "object"},
        "provenance": {"type": "object"},
        "resolved_policy": {"type": "object"},
        "policy_provenance": {"type": "object"},
        "policy_resolution": {"type": "object"},
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


_VALIDATION_ALLOWLIST_DEFAULT: set[str] = set()

try:
    apply_validation_allowlist_schema(
        REPORT_JSON_SCHEMA,
        load_validation_allowlist_strict(),
    )
except (
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
    ValidationAllowlistContractError,
):
    pass

try:
    REPORT_JSON_SCHEMA["properties"]["primary_metric"]["properties"]["kind"]["enum"] = (
        sorted(load_metric_kind_catalog())
    )
except (KeyError, MetricKindContractError, TypeError):
    pass


def _validate_with_jsonschema(report: dict[str, Any], schema: Any = None) -> bool:
    """Validate evaluation report with JSON Schema when available."""
    if jsonschema is None:
        return False
    active_schema = REPORT_JSON_SCHEMA if schema is None else schema
    try:
        if not isinstance(active_schema, dict) and callable(
            getattr(active_schema, "validate", None)
        ):
            active_schema.validate(report)
        else:
            jsonschema.validate(instance=report, schema=active_schema)
        return True
    except _JSONSCHEMA_FAILURES:
        return False


def _compile_jsonschema_validator(schema: dict[str, Any]) -> Any | None:
    """Compile a schema once when the installed jsonschema API supports it."""
    if jsonschema is None:
        return None
    validators = getattr(jsonschema, "validators", None)
    validator_for = getattr(validators, "validator_for", None)
    if not callable(validator_for):
        return None
    validator_type = validator_for(schema)
    validator_type.check_schema(schema)
    return validator_type(schema)


_compiled_validator_runtime: Any = object()
_compiled_validator_runtime_generation = 0


def _compiled_validator_runtime_generation_key() -> int:
    """Advance the cache key whenever the JSON Schema runtime object changes."""
    global _compiled_validator_runtime
    global _compiled_validator_runtime_generation
    if jsonschema is not _compiled_validator_runtime:
        _compiled_validator_runtime = jsonschema
        _compiled_validator_runtime_generation += 1
    return _compiled_validator_runtime_generation


@lru_cache(maxsize=16)
def _compiled_report_validator(
    validation_keys: tuple[str, ...], schema_runtime_generation: int
) -> Any | None:
    """Return a compiled validator for one immutable validation-key contract."""
    del schema_runtime_generation
    schema = copy.deepcopy(REPORT_JSON_SCHEMA)
    apply_validation_allowlist_schema(schema, set(validation_keys))
    return _compile_jsonschema_validator(schema)


def validate_report(report: object) -> bool:
    """Validate evaluation report structure and essential flags."""
    if not isinstance(report, dict):
        return False
    validation_keys = _VALIDATION_ALLOWLIST_DEFAULT
    schema_properties = REPORT_JSON_SCHEMA.get("properties")
    if not isinstance(schema_properties, dict):
        return False
    validation_spec = schema_properties.get("validation")
    if not isinstance(validation_spec, dict):
        return False
    try:
        if report.get("schema_version") != REPORT_SCHEMA_VERSION:
            return False

        try:
            validation_keys = load_validation_allowlist_strict()
        except (
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            ValidationAllowlistContractError,
        ):
            return False

        try:
            immutable_validation_keys = tuple(sorted(validation_keys))
            compiled_validator = _compiled_report_validator(
                immutable_validation_keys, _compiled_validator_runtime_generation_key()
            )
            schema_for_validation: Any = compiled_validator
            if schema_for_validation is None:
                schema_for_validation = copy.deepcopy(REPORT_JSON_SCHEMA)
                apply_validation_allowlist_schema(
                    schema_for_validation, set(immutable_validation_keys)
                )
            jsonschema_ok = _validate_with_jsonschema(
                report,
                schema_for_validation,
            )
        except TypeError:
            jsonschema_ok = _validate_with_jsonschema(report)
        except _JSONSCHEMA_FAILURES:
            return False

        if not jsonschema_ok:
            return False

        primary_metric = report.get("primary_metric")
        if not isinstance(primary_metric, dict):
            return False
        try:
            normalized_kind = normalize_metric_kind(primary_metric.get("kind"))
        except (MetricKindContractError, ValueError):
            return False
        if normalized_kind is None:
            return False

        validation = report.get("validation", {})
        if "validation" in report and not isinstance(validation, dict):
            return False
        if not isinstance(validation, dict):
            validation = {}
        if any(key not in validation_keys for key in validation):
            return False
        for flag in [
            "preview_final_drift_acceptable",
            "primary_metric_acceptable",
            "invariants_pass",
            "spectral_stable",
            "rmt_stable",
            "guard_metric_impact_acceptable",
            "guard_warnings_present",
            "guard_warning_policy_acceptable",
        ]:
            # If present, must be boolean; tolerate missing opt-in flags
            if flag in validation and not isinstance(validation.get(flag), bool):
                return False

        return True
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "DEFAULT_VALIDATION_ALLOWLIST",
    "REPORT_SCHEMA_VERSION",
    "REPORT_JSON_SCHEMA",
    "ValidationAllowlistContractError",
    "apply_validation_allowlist_schema",
    "load_validation_allowlist",
    "load_validation_allowlist_strict",
    "validate_report",
]
