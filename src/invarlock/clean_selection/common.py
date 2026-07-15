# ruff: noqa: UP045  # This shell-facing contract must parse on Python 3.9.
"""Shared schemas and strict parsing for clean-transformation selection evidence.

The evidence-pack verifier ships from :mod:`src`, whereas generation helpers
live under ``scripts/`` and are deliberately excluded from built artifacts.
This module owns the dependency-light constants, data structures, canonical
JSON helpers, and strict parsers shared by the package's artifact, bundle,
candidate, and snapshot validators. Those owners perform sidecar validation,
winner replay, and immutable snapshot checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from invarlock.evidence_pack_json import (
    StrictJsonError,
    load_json_object,
    read_json_object_snapshot,
    read_regular_file_bytes,
    sha256_prefixed,
)

CLEAN_SELECTION_CONTRACT_VERSION = "clean-transformation-selection-v1"
CANDIDATE_RECORD_SCHEMA = "invarlock/clean-transformation-candidate-record-v1"
SELECTION_RECEIPT_SCHEMA = "invarlock/clean-transformation-selection-receipt-v1"
SELECTED_ENTRY_SCHEMA = "invarlock/clean-transformation-selected-entry-v1"
CLEAN_SELECTION_BUNDLE_SCHEMA = "invarlock/clean-transformation-selection-bundle-v1"
SELECTION_CONFIG_SCHEMA = "invarlock/clean-transformation-selection-config-v1"
EVALUATION_SCHEDULE_SCHEMA = "invarlock/clean-transformation-schedule-v1"
DECISION_RULE_SCHEMA = "invarlock/clean-transformation-decision-rule-v1"
CANDIDATE_EVALUATION_SCHEMA = "invarlock/clean-transformation-candidate-evaluation-v1"
SELECTION_EXECUTION_RECEIPT_SCHEMA = (
    "invarlock/clean-transformation-selection-execution-receipt-v1"
)
EVALUATOR_PROVENANCE_SCHEMA = "invarlock/clean-transformation-evaluator-provenance-v1"
REPORT_SELECTION_BINDING_SCHEMA = (
    "invarlock/clean-transformation-report-selection-binding-v1"
)
TRANSFORMATION_REPLAY_SCHEMA = "invarlock/generated-transformation-replay-v1"
RUNTIME_RELOAD_PROOF_SCHEMA = "invarlock/transformation-runtime-reload-proof-v1"
RUNTIME_LOAD_DIAGNOSTICS_SCHEMA = "invarlock/pretrained-load-diagnostics-v1"
RUNTIME_STORAGE_KEY_AUDIT_SCHEMA = "invarlock/safetensors-storage-key-audit-v1"
TRANSFORMATION_SCOPE_POLICY = "architecture-aware-transformation-v1"
TRANSFORMATION_PARAMETERS_SCHEMA = "invarlock/transformation-parameters-v1"
TRANSFORMATION_CONTRACT_VERSION = "verifier-grade-transformation-v1"
MINIMUM_CLEAN_SELECTION_CANDIDATES = 2

_SHA256_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
_REVISION_RE = re.compile(r"[a-f0-9]{40}(?:[a-f0-9]{24})?\Z")
_CANDIDATE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_MODEL_KEY_RE = re.compile(r"[^\x00\r\n]+\Z")
_NONNEGATIVE_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_RUNTIME_DEVICE_RE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)\Z")
_RUNTIME_RELOAD_PROMPT = "InvarLock verifier-grade transformation runtime proof."
_RUNTIME_RELOAD_PROMPT_SHA256 = (
    "sha256:" + hashlib.sha256(_RUNTIME_RELOAD_PROMPT.encode("utf-8")).hexdigest()
)
_RUNTIME_RELOAD_PROOF_FIELDS = frozenset(
    {
        "schema",
        "ok",
        "replay_schema",
        "edit_type",
        "artifact_identity",
        "replay_artifact_identity",
        "prompt_sha256",
        "device",
        "input_device",
        "reload_runs",
        "token_ids_sha256",
        "token_ids_shape",
        "logits_sha256",
        "logits_shape",
        "all_logits_finite",
        "repeat_deterministic",
        "load_diagnostics",
        "storage_key_audit",
    }
)
_RUNTIME_LOAD_DIAGNOSTIC_FIELDS = frozenset(
    {"unexpected_keys", "missing_keys", "mismatched_keys", "error_msgs"}
)
_RUNTIME_LOAD_DIAGNOSTICS_FIELDS = frozenset({"schema", "reloads"})
_RUNTIME_STORAGE_KEY_AUDIT_FIELDS = frozenset(
    {
        "artifact_storage_key_count",
        "artifact_storage_keys_sha256",
        "model_state_key_count",
        "model_state_keys_sha256",
        "unexpected_storage_keys",
    }
)
_RUNTIME_STORAGE_KEY_AUDIT_ENVELOPE_FIELDS = frozenset({"schema", "reloads"})
_SUPPORTED_PARAMETERS: dict[str, frozenset[str]] = {
    "quant_rtn": frozenset({"bits", "group_size"}),
    "synthetic_lowrank_delta": frozenset({"rank", "scale"}),
    "synthetic_dense_update": frozenset({"step_size", "iterations"}),
}
_MAX_SYNTHETIC_LOWRANK_RANK = 32
_MAX_SYNTHETIC_DENSE_ITERATIONS = 16
_ALGORITHMS = {
    "quant_rtn": "groupwise_rtn_dequantized_per_row_v1",
    "synthetic_lowrank_delta": "deterministic_synthetic_lowrank_delta_v1",
    "synthetic_dense_update": "deterministic_synthetic_dense_update_v1",
}
_ELIGIBLE_VALIDATION_FIELDS = (
    "invariants_pass",
    "spectral_stable",
    "rmt_stable",
    "preview_final_drift_acceptable",
    "primary_metric_acceptable",
    "primary_metric_tail_acceptable",
    "guard_metric_impact_acceptable",
    "guard_warning_policy_acceptable",
)


class CleanSelectionEvidenceError(ValueError):
    """Raised when a staged clean-selection bundle is not verifier-grade."""


@dataclass(frozen=True)
class SelectionBundleSnapshot:
    """One verified clean-selection bundle and every byte it authenticated.

    Direct staging and final-replay attachment must not hash a pathname and
    later reopen it.  This snapshot keeps the exact regular-file bytes that
    were parsed and verified, so callers can atomically publish those same
    bytes rather than a later substitution at the source pathname.
    """

    bundle: dict[str, object]
    bundle_bytes: bytes
    sidecar_bytes: Mapping[str, bytes]


def _no_bare_selected_by(value: object, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower().startswith("selected_by_"):
                raise CleanSelectionEvidenceError(
                    f"bare selected_by claim is not evidence at {location}.{key}"
                )
            _no_bare_selected_by(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _no_bare_selected_by(child, location=f"{location}[{index}]")
    elif isinstance(value, str) and value.lower().startswith("selected_by_"):
        raise CleanSelectionEvidenceError(
            f"bare selected_by claim is not evidence at {location}"
        )


def canonical_json_sha256(value: object) -> str:
    """Return the canonical JSON digest used by the v1 selection contract."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CleanSelectionEvidenceError(
            "selection evidence cannot be canonicalized as JSON"
        ) from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def raw_file_sha256(path: Path) -> str:
    """Return a raw-byte digest for one regular staged sidecar."""

    try:
        data = read_regular_file_bytes(path, label="selection evidence")
    except StrictJsonError as exc:
        raise CleanSelectionEvidenceError(str(exc)) from exc
    return sha256_prefixed(data)


def strict_json_object(path: Path, *, label: str) -> dict[str, object]:
    """Load one regular duplicate-free UTF-8 JSON object."""

    try:
        return cast(dict[str, object], load_json_object(path, label=label))
    except StrictJsonError as exc:
        raise CleanSelectionEvidenceError(str(exc)) from exc


def strict_json_object_snapshot(
    path: Path, *, label: str
) -> tuple[bytes, dict[str, object]]:
    """Read, hash, and parse one JSON sidecar from exactly one byte snapshot."""

    try:
        raw, payload = read_json_object_snapshot(path, label=label)
    except StrictJsonError as exc:
        raise CleanSelectionEvidenceError(str(exc)) from exc
    return raw, cast(dict[str, object], payload)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CleanSelectionEvidenceError(f"{label} must be an object")
    return value


def _exact_mapping(
    value: object, *, label: str, fields: frozenset[str]
) -> Mapping[str, object]:
    mapping = _mapping(value, label=label)
    if set(mapping) != fields:
        missing = sorted(fields - set(mapping))
        extra = sorted(set(mapping) - fields)
        raise CleanSelectionEvidenceError(
            f"{label} has unbound, missing, or arbitrary fields "
            f"(missing={missing}, extra={extra})"
        )
    return mapping


def _text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _MODEL_KEY_RE.fullmatch(value) is None
    ):
        raise CleanSelectionEvidenceError(f"{label} must be a non-empty trimmed string")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CleanSelectionEvidenceError(
            f"{label} must be a sha256:<64 lowercase hex> digest"
        )
    return value


def _identity(value: object, *, label: str) -> dict[str, str]:
    payload = _exact_mapping(value, label=label, fields=frozenset({"kind", "sha256"}))
    if payload["kind"] != "local_checkpoint_tree":
        raise CleanSelectionEvidenceError(f"{label}.kind must be local_checkpoint_tree")
    return {
        "kind": "local_checkpoint_tree",
        "sha256": _sha256(payload["sha256"], label=f"{label}.sha256"),
    }


def _scope(value: object, *, label: str) -> str:
    """Mirror the generator scope grammar and reject noncanonical spellings."""

    if not isinstance(value, str):
        raise CleanSelectionEvidenceError(
            f"{label} must be a canonical transformation scope"
        )
    text = value.strip()
    if not text or text.count("@") > 1:
        raise CleanSelectionEvidenceError(
            f"{label} must be a canonical transformation scope"
        )
    if "@" in text:
        raw_base, raw_qualifiers = text.split("@", 1)
    else:
        raw_base, raw_qualifiers = text, ""
    base = raw_base.strip().lower()
    if base not in {"ffn", "attn", "all"}:
        raise CleanSelectionEvidenceError(
            f"{label} must be a canonical transformation scope"
        )
    values: dict[str, int] = {}
    if "@" in text:
        if not raw_qualifiers.strip():
            raise CleanSelectionEvidenceError(
                f"{label} must be a canonical transformation scope"
            )
        for raw_item in raw_qualifiers.split(","):
            item = raw_item.strip()
            if not item or item.count("=") != 1:
                raise CleanSelectionEvidenceError(
                    f"{label} must be a canonical transformation scope"
                )
            raw_name, raw_value = (part.strip() for part in item.split("=", 1))
            name = raw_name.lower()
            if name not in {"layers", "layer"} or name in values:
                raise CleanSelectionEvidenceError(
                    f"{label} must be a canonical transformation scope"
                )
            if _NONNEGATIVE_DECIMAL_RE.fullmatch(raw_value) is None:
                raise CleanSelectionEvidenceError(
                    f"{label} must be a canonical transformation scope"
                )
            parsed = int(raw_value)
            if name == "layers" and parsed == 0:
                raise CleanSelectionEvidenceError(
                    f"{label} must be a canonical transformation scope"
                )
            values[name] = parsed
    layer_limit = values.get("layers")
    layer = values.get("layer")
    if layer_limit is not None and layer is not None and layer >= layer_limit:
        raise CleanSelectionEvidenceError(
            f"{label} must be a canonical transformation scope"
        )
    qualifiers: list[str] = []
    if layer_limit is not None:
        qualifiers.append(f"layers={layer_limit}")
    if layer is not None:
        qualifiers.append(f"layer={layer}")
    canonical = base if not qualifiers else f"{base}@{','.join(qualifiers)}"
    if value != canonical:
        raise CleanSelectionEvidenceError(f"{label} must use canonical scope syntax")
    return canonical


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CleanSelectionEvidenceError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CleanSelectionEvidenceError(f"{label} must be a finite number")
    return normalized


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CleanSelectionEvidenceError(f"{label} must be a positive integer")
    return value


def _transform(
    value: object, *, label: str, expected_edit_type: Optional[str] = None
) -> dict[str, object]:
    payload = _exact_mapping(
        value, label=label, fields=frozenset({"edit_type", "parameters", "scope"})
    )
    edit_type = _text(payload["edit_type"], label=f"{label}.edit_type")
    if edit_type not in _SUPPORTED_PARAMETERS:
        raise CleanSelectionEvidenceError(f"{label}.edit_type is unsupported")
    if expected_edit_type is not None and edit_type != expected_edit_type:
        raise CleanSelectionEvidenceError(f"{label}.edit_type mismatch")
    raw_parameters = _exact_mapping(
        payload["parameters"],
        label=f"{label}.parameters",
        fields=_SUPPORTED_PARAMETERS[edit_type],
    )
    parameters: dict[str, object]
    if edit_type == "quant_rtn":
        bits = _positive_int(raw_parameters["bits"], label=f"{label}.parameters.bits")
        if not 2 <= bits <= 8:
            raise CleanSelectionEvidenceError(
                f"{label}.parameters.bits must be in [2, 8]"
            )
        parameters = {
            "bits": bits,
            "group_size": _positive_int(
                raw_parameters["group_size"], label=f"{label}.parameters.group_size"
            ),
        }
    elif edit_type == "synthetic_lowrank_delta":
        scale = _finite(raw_parameters["scale"], label=f"{label}.parameters.scale")
        if scale <= 0:
            raise CleanSelectionEvidenceError(
                f"{label}.parameters.scale must be positive"
            )
        rank = _positive_int(raw_parameters["rank"], label=f"{label}.parameters.rank")
        if rank > _MAX_SYNTHETIC_LOWRANK_RANK:
            raise CleanSelectionEvidenceError(
                f"{label}.parameters.rank must be at most {_MAX_SYNTHETIC_LOWRANK_RANK}"
            )
        parameters = {
            "rank": rank,
            "scale": scale,
        }
    else:
        step_size = _finite(
            raw_parameters["step_size"], label=f"{label}.parameters.step_size"
        )
        if step_size <= 0:
            raise CleanSelectionEvidenceError(
                f"{label}.parameters.step_size must be positive"
            )
        iterations = _positive_int(
            raw_parameters["iterations"], label=f"{label}.parameters.iterations"
        )
        if iterations > _MAX_SYNTHETIC_DENSE_ITERATIONS:
            raise CleanSelectionEvidenceError(
                f"{label}.parameters.iterations must be at most "
                f"{_MAX_SYNTHETIC_DENSE_ITERATIONS}"
            )
        parameters = {"step_size": step_size, "iterations": iterations}
    if canonical_json_sha256(raw_parameters) != canonical_json_sha256(parameters):
        raise CleanSelectionEvidenceError(
            f"{label}.parameters must use canonical numeric forms"
        )
    return {
        "edit_type": edit_type,
        "parameters": parameters,
        "scope": _scope(payload["scope"], label=f"{label}.scope"),
    }


def _selection_config(value: object) -> dict[str, object]:
    payload = _exact_mapping(
        value,
        label="selection_config",
        fields=frozenset({"schema", "dataset", "seed", "schedule"}),
    )
    if payload["schema"] != SELECTION_CONFIG_SCHEMA:
        raise CleanSelectionEvidenceError("selection_config has an unrecognized schema")
    dataset = _exact_mapping(
        payload["dataset"],
        label="selection_config.dataset",
        fields=frozenset({"name", "revision", "split", "content_sha256"}),
    )
    revision = _text(dataset["revision"], label="selection_config.dataset.revision")
    if (
        _REVISION_RE.fullmatch(revision) is None
        and _SHA256_RE.fullmatch(revision) is None
    ):
        raise CleanSelectionEvidenceError(
            "selection_config.dataset.revision must be immutable"
        )
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CleanSelectionEvidenceError("selection_config.seed must be non-negative")
    schedule = _exact_mapping(
        payload["schedule"],
        label="selection_config.schedule",
        fields=frozenset(
            {
                "schema",
                "candidate_order",
                "evaluation_repeats",
                "max_examples",
                "batch_size",
                "shuffle",
            }
        ),
    )
    if schedule["schema"] != EVALUATION_SCHEDULE_SCHEMA:
        raise CleanSelectionEvidenceError(
            "selection_config.schedule has an unrecognized schema"
        )
    if schedule["candidate_order"] != "candidate_id_ascending":
        raise CleanSelectionEvidenceError(
            "selection_config.schedule.candidate_order must be candidate_id_ascending"
        )
    for key in ("evaluation_repeats", "max_examples", "batch_size"):
        _positive_int(schedule[key], label=f"selection_config.schedule.{key}")
    if not isinstance(schedule["shuffle"], bool):
        raise CleanSelectionEvidenceError(
            "selection_config.schedule.shuffle must be boolean"
        )
    return cast(dict[str, object], dict(payload))


def _decision_rule(value: object) -> dict[str, object]:
    payload = _exact_mapping(
        value,
        label="decision_rule",
        fields=frozenset({"schema", "kind", "metric_order", "tie_breaker"}),
    )
    if payload["schema"] != DECISION_RULE_SCHEMA:
        raise CleanSelectionEvidenceError("decision_rule has an unrecognized schema")
    if payload["kind"] != "lexicographic_metrics_v1":
        raise CleanSelectionEvidenceError("decision_rule.kind is unsupported")
    if payload["tie_breaker"] != "candidate_id_ascending":
        raise CleanSelectionEvidenceError("decision_rule.tie_breaker is unsupported")
    if payload["metric_order"] != ["quality_loss"]:
        raise CleanSelectionEvidenceError(
            "decision_rule.metric_order must be exactly ['quality_loss'] until runtime metrics have a report contract"
        )
    return cast(dict[str, object], dict(payload))


def _safe_relative_json_path(value: object, *, label: str) -> str:
    path = _text(value, label=label)
    if "\\" in path or path.startswith("/") or not path.endswith(".json"):
        raise CleanSelectionEvidenceError(f"{label} must be a safe relative JSON path")
    parts = path.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise CleanSelectionEvidenceError(
            f"{label} must not traverse outside evidence root"
        )
    return path


def _reference(
    value: object,
    *,
    label: str,
    baseline_identity: Mapping[str, str],
    artifact_identity: Optional[Mapping[str, str]] = None,
    replay_identity: Optional[Mapping[str, str]] = None,
) -> dict[str, object]:
    fields = {"path", "sha256", "artifact_identity", "baseline_identity"}
    if replay_identity is not None:
        fields.add("replay_artifact_identity")
    payload = _exact_mapping(value, label=label, fields=frozenset(fields))
    normalized_artifact = _identity(
        payload["artifact_identity"], label=f"{label}.artifact_identity"
    )
    normalized_baseline = _identity(
        payload["baseline_identity"], label=f"{label}.baseline_identity"
    )
    if normalized_baseline != dict(baseline_identity):
        raise CleanSelectionEvidenceError(f"{label}.baseline_identity mismatch")
    if artifact_identity is not None and normalized_artifact != dict(artifact_identity):
        raise CleanSelectionEvidenceError(f"{label}.artifact_identity mismatch")
    result: dict[str, object] = {
        "path": _safe_relative_json_path(payload["path"], label=f"{label}.path"),
        "sha256": _sha256(payload["sha256"], label=f"{label}.sha256"),
        "artifact_identity": normalized_artifact,
        "baseline_identity": normalized_baseline,
    }
    if replay_identity is not None:
        observed_replay = _identity(
            payload["replay_artifact_identity"],
            label=f"{label}.replay_artifact_identity",
        )
        if observed_replay != dict(replay_identity):
            raise CleanSelectionEvidenceError(
                f"{label}.replay_artifact_identity mismatch"
            )
        result["replay_artifact_identity"] = observed_replay
    return result


def _sidecar_reference(value: object, *, label: str) -> dict[str, str]:
    """Canonicalize a non-model JSON reference retained with a candidate."""

    payload = _exact_mapping(value, label=label, fields=frozenset({"path", "sha256"}))
    return {
        "path": _safe_relative_json_path(payload["path"], label=f"{label}.path"),
        "sha256": _sha256(payload["sha256"], label=f"{label}.sha256"),
    }


def _candidate_report_reference(
    value: object,
    *,
    label: str,
    baseline_identity: Mapping[str, str],
    artifact_identity: Optional[Mapping[str, str]] = None,
) -> dict[str, object]:
    """Canonicalize one evaluator report plus its immutable runtime manifest."""

    payload = _exact_mapping(
        value,
        label=label,
        fields=frozenset({"report", "runtime_manifest"}),
    )
    report = _reference(
        payload["report"],
        label=f"{label}.report",
        baseline_identity=baseline_identity,
        artifact_identity=artifact_identity,
    )
    return {
        "report": report,
        "runtime_manifest": _sidecar_reference(
            payload["runtime_manifest"], label=f"{label}.runtime_manifest"
        ),
    }
