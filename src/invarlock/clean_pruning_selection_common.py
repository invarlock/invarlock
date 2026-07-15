"""Verifier-owned evidence contract for clean magnitude-pruning selection.

This is deliberately separate from the generated-transformation selector.
Magnitude pruning is not a generic parameter edit: an acceptable candidate must
carry a replay that binds its architecture-specific target topology, exact
per-tensor flattened-tie rule, baseline tree, and materialized artifact.  The
contract retains every evaluated candidate, authenticates every retained JSON
sidecar from a single-read snapshot, and recomputes the mean-quality-loss
winner.  It does not provide a synthetic producer or make an unexecuted
candidate eligible.

The module is package-owned so a final evidence-pack verifier can enforce the
same rules without importing the script-only pruning materializer.  A caller
must still supply real candidate artifacts and evaluator output; absence of a
real producer is intentionally a fail-closed integration state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from . import pruning_contract as _pruning_contract
from .evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    read_regular_file_bytes,
    sha256_prefixed,
)

CLEAN_PRUNING_SELECTION_CONTRACT_VERSION = "clean-pruning-selection-v1"
CLEAN_PRUNING_CANDIDATE_RECORD_SCHEMA = "invarlock/clean-pruning-candidate-record-v1"
CLEAN_PRUNING_SELECTION_RECEIPT_SCHEMA = "invarlock/clean-pruning-selection-receipt-v1"
CLEAN_PRUNING_SELECTED_ENTRY_SCHEMA = "invarlock/clean-pruning-selected-entry-v1"
CLEAN_PRUNING_SELECTION_BUNDLE_SCHEMA = "invarlock/clean-pruning-selection-bundle-v1"
CLEAN_PRUNING_SELECTION_CONFIG_SCHEMA = "invarlock/clean-pruning-selection-config-v1"
CLEAN_PRUNING_EVALUATION_SCHEDULE_SCHEMA = "invarlock/clean-pruning-schedule-v1"
CLEAN_PRUNING_DECISION_RULE_SCHEMA = "invarlock/clean-pruning-decision-rule-v1"
CLEAN_PRUNING_CANDIDATE_EVALUATION_SCHEMA = (
    "invarlock/clean-pruning-candidate-evaluation-v1"
)
CLEAN_PRUNING_EXECUTION_RECEIPT_SCHEMA = (
    "invarlock/clean-pruning-selection-execution-receipt-v1"
)
CLEAN_PRUNING_EVALUATOR_PROVENANCE_SCHEMA = (
    "invarlock/clean-pruning-evaluator-provenance-v1"
)
CLEAN_PRUNING_REPORT_BINDING_SCHEMA = "invarlock/clean-pruning-report-binding-v1"
CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME = "bundle.json"
MINIMUM_CLEAN_PRUNING_SELECTION_CANDIDATES = 2

PRUNING_REPLAY_SCHEMA = _pruning_contract.PRUNING_REPLAY_SCHEMA
PRUNING_TARGET_MANIFEST_SCHEMA = _pruning_contract.PRUNING_TARGET_MANIFEST_SCHEMA
PRUNING_SCOPE_POLICY = _pruning_contract.PRUNING_SCOPE_POLICY_VERSION
PRUNING_ALGORITHM = _pruning_contract.PRUNING_ALGORITHM
PRUNING_STORAGE_POLICY = _pruning_contract.PRUNING_STORAGE_POLICY
RUNTIME_RELOAD_PROOF_SCHEMA = "invarlock/transformation-runtime-reload-proof-v1"
RUNTIME_LOAD_DIAGNOSTICS_SCHEMA = "invarlock/pretrained-load-diagnostics-v1"
RUNTIME_STORAGE_KEY_AUDIT_SCHEMA = "invarlock/safetensors-storage-key-audit-v1"

_SHA256_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
_REVISION_RE = re.compile(r"[a-f0-9]{40}(?:[a-f0-9]{24})?\Z")
_CANDIDATE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_MODEL_KEY_RE = re.compile(r"[^\x00\r\n]+\Z")
_MODEL_TYPE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_TENSOR_NAME_RE = re.compile(r"[^\x00\r\n]+\Z")
_TORCH_DTYPE_RE = re.compile(r"torch\.[A-Za-z0-9_]+\Z")
_RUNTIME_DEVICE_RE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)\Z")
_RUNTIME_RELOAD_PROMPT = "InvarLock verifier-grade transformation runtime proof."
_RUNTIME_RELOAD_PROMPT_SHA256 = (
    "sha256:" + hashlib.sha256(_RUNTIME_RELOAD_PROMPT.encode("utf-8")).hexdigest()
)

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
_PRUNING_REPLAY_FIELDS = frozenset(
    {
        "schema",
        "ok",
        "edit_type",
        "scope",
        "target_sparsity",
        "scope_policy",
        "pruning_algorithm",
        "storage_policy",
        "model_type",
        "architecture",
        "config_sha256",
        "target_manifest",
        "target_manifest_sha256",
        "baseline_identity",
        "artifact_identity",
        "checked_tensors",
        "selected_tensors",
        "selected_params",
        "total_params",
        "expected_pruned_params",
        "expected_changed_params",
        "observed_changed_params",
        "original_zero_params",
        "observed_zero_params",
        "out_of_scope_tensors_checked",
        "out_of_scope_bytes_checked",
        "support_files_checked",
        "issues",
    }
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


class CleanPruningSelectionEvidenceError(ValueError):
    """Raised when clean-pruning selection evidence is incomplete or forged."""


@dataclass(frozen=True)
class CleanPruningSelectionBundleSnapshot:
    """Verified bundle plus the exact sidecar bytes it authenticated.

    Staging must publish these bytes rather than reopen mutable source paths.
    """

    bundle: dict[str, object]
    bundle_bytes: bytes
    sidecar_bytes: Mapping[str, bytes]


def canonical_json_sha256(value: object) -> str:
    """Return the canonical semantic digest used by this strict contract."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CleanPruningSelectionEvidenceError(
            "pruning selection evidence cannot be canonicalized as JSON"
        ) from exc
    return sha256_prefixed(encoded)


def raw_file_sha256(path: Path) -> str:
    """Return a raw-byte digest for one regular retained sidecar."""

    try:
        return sha256_prefixed(
            read_regular_file_bytes(path, label="selection evidence")
        )
    except StrictJsonError as exc:
        raise CleanPruningSelectionEvidenceError(str(exc)) from exc


def strict_json_object_snapshot(
    path: Path, *, label: str
) -> tuple[bytes, dict[str, object]]:
    """Read, parse, and retain one duplicate-free JSON object exactly once."""

    try:
        raw, payload = read_json_object_snapshot(path, label=label)
    except StrictJsonError as exc:
        raise CleanPruningSelectionEvidenceError(str(exc)) from exc
    return raw, cast(dict[str, object], payload)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CleanPruningSelectionEvidenceError(f"{label} must be an object")
    return value


def _exact_mapping(
    value: object, *, label: str, fields: frozenset[str]
) -> Mapping[str, object]:
    payload = _mapping(value, label=label)
    payload_fields = frozenset(payload)
    if payload_fields != fields:
        missing = sorted(fields - payload_fields)
        extra = sorted(payload_fields - fields)
        raise CleanPruningSelectionEvidenceError(
            f"{label} has unbound, missing, or arbitrary fields "
            f"(missing={missing}, extra={extra})"
        )
    return payload


def _text(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _MODEL_KEY_RE.fullmatch(value) is None
    ):
        raise CleanPruningSelectionEvidenceError(
            f"{label} must be a non-empty trimmed string"
        )
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CleanPruningSelectionEvidenceError(
            f"{label} must be a sha256:<64 lowercase hex> digest"
        )
    return value


def _identity(value: object, *, label: str) -> dict[str, str]:
    payload = _exact_mapping(value, label=label, fields=frozenset({"kind", "sha256"}))
    if payload["kind"] != "local_checkpoint_tree":
        raise CleanPruningSelectionEvidenceError(
            f"{label}.kind must be local_checkpoint_tree"
        )
    return {
        "kind": "local_checkpoint_tree",
        "sha256": _sha256(payload["sha256"], label=f"{label}.sha256"),
    }


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CleanPruningSelectionEvidenceError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CleanPruningSelectionEvidenceError(
            f"{label} must be a non-negative integer"
        )
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CleanPruningSelectionEvidenceError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CleanPruningSelectionEvidenceError(f"{label} must be a finite number")
    return normalized


def _scope(value: object, *, label: str) -> str:
    if value not in {"ffn", "attn", "all"}:
        raise CleanPruningSelectionEvidenceError(
            f"{label} must be one of ffn, attn, or all"
        )
    return value


def _pruning_spec(value: object, *, label: str) -> dict[str, object]:
    payload = _exact_mapping(
        value,
        label=label,
        fields=frozenset({"edit_type", "scope", "target_sparsity"}),
    )
    if payload["edit_type"] != "magnitude_prune":
        raise CleanPruningSelectionEvidenceError(
            f"{label}.edit_type must be magnitude_prune"
        )
    sparsity = _finite(payload["target_sparsity"], label=f"{label}.target_sparsity")
    if not 0.0 < sparsity < 1.0:
        raise CleanPruningSelectionEvidenceError(
            f"{label}.target_sparsity must be in (0, 1)"
        )
    result = {
        "edit_type": "magnitude_prune",
        "scope": _scope(payload["scope"], label=f"{label}.scope"),
        "target_sparsity": sparsity,
    }
    if canonical_json_sha256(payload) != canonical_json_sha256(result):
        raise CleanPruningSelectionEvidenceError(
            f"{label} must use canonical numeric forms"
        )
    return result


def _selection_config(value: object) -> dict[str, object]:
    payload = _exact_mapping(
        value,
        label="selection_config",
        fields=frozenset({"schema", "dataset", "seed", "schedule"}),
    )
    if payload["schema"] != CLEAN_PRUNING_SELECTION_CONFIG_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "selection_config has an unrecognized schema"
        )
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
        raise CleanPruningSelectionEvidenceError(
            "selection_config.dataset.revision must be immutable"
        )
    normalized_dataset = {
        "name": _text(dataset["name"], label="selection_config.dataset.name"),
        "revision": revision,
        "split": _text(dataset["split"], label="selection_config.dataset.split"),
        "content_sha256": _sha256(
            dataset["content_sha256"], label="selection_config.dataset.content_sha256"
        ),
    }
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CleanPruningSelectionEvidenceError(
            "selection_config.seed must be non-negative"
        )
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
    if schedule["schema"] != CLEAN_PRUNING_EVALUATION_SCHEDULE_SCHEMA:
        raise CleanPruningSelectionEvidenceError(
            "selection_config.schedule has an unrecognized schema"
        )
    if schedule["candidate_order"] != "candidate_id_ascending":
        raise CleanPruningSelectionEvidenceError(
            "selection_config.schedule.candidate_order must be candidate_id_ascending"
        )
    if schedule["shuffle"] is not False:
        raise CleanPruningSelectionEvidenceError(
            "selection_config.schedule.shuffle must be false"
        )
    normalized_schedule = {
        "schema": CLEAN_PRUNING_EVALUATION_SCHEDULE_SCHEMA,
        "candidate_order": "candidate_id_ascending",
        "evaluation_repeats": _positive_int(
            schedule["evaluation_repeats"],
            label="selection_config.schedule.evaluation_repeats",
        ),
        "max_examples": _positive_int(
            schedule["max_examples"], label="selection_config.schedule.max_examples"
        ),
        "batch_size": _positive_int(
            schedule["batch_size"], label="selection_config.schedule.batch_size"
        ),
        "shuffle": False,
    }
    return {
        "schema": CLEAN_PRUNING_SELECTION_CONFIG_SCHEMA,
        "dataset": normalized_dataset,
        "seed": seed,
        "schedule": normalized_schedule,
    }


def _decision_rule(value: object) -> dict[str, object]:
    payload = _exact_mapping(
        value,
        label="decision_rule",
        fields=frozenset({"schema", "metric", "direction", "tie_breaker"}),
    )
    expected: dict[str, object] = {
        "schema": CLEAN_PRUNING_DECISION_RULE_SCHEMA,
        "metric": "mean_quality_loss_from_strict_reports_v1",
        "direction": "minimize",
        "tie_breaker": "candidate_id_ascending",
    }
    if dict(payload) != expected:
        raise CleanPruningSelectionEvidenceError(
            "decision_rule must be the fixed deterministic mean-quality-loss rule"
        )
    return expected


def _selection_domain(value: object) -> dict[str, str]:
    payload = _exact_mapping(
        value,
        label="selection_domain",
        fields=frozenset(
            {
                "edit_type",
                "scope_policy",
                "pruning_algorithm",
                "storage_policy",
                "target_manifest_schema",
            }
        ),
    )
    expected = {
        "edit_type": "magnitude_prune",
        "scope_policy": PRUNING_SCOPE_POLICY,
        "pruning_algorithm": PRUNING_ALGORITHM,
        "storage_policy": PRUNING_STORAGE_POLICY,
        "target_manifest_schema": PRUNING_TARGET_MANIFEST_SCHEMA,
    }
    if dict(payload) != expected:
        raise CleanPruningSelectionEvidenceError(
            "selection_domain must bind the supported magnitude-pruning contract"
        )
    return expected


def _safe_relative_json_path(value: object, *, label: str) -> str:
    path = _text(value, label=label)
    if "\\" in path or path.startswith("/") or not path.endswith(".json"):
        raise CleanPruningSelectionEvidenceError(
            f"{label} must be a safe relative JSON path"
        )
    parts = path.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise CleanPruningSelectionEvidenceError(
            f"{label} must not traverse outside evidence root"
        )
    return path


def _sidecar_reference(value: object, *, label: str) -> dict[str, str]:
    payload = _exact_mapping(value, label=label, fields=frozenset({"path", "sha256"}))
    return {
        "path": _safe_relative_json_path(payload["path"], label=f"{label}.path"),
        "sha256": _sha256(payload["sha256"], label=f"{label}.sha256"),
    }


def _bound_reference(
    value: object,
    *,
    label: str,
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str] | None = None,
    replay_identity: Mapping[str, str] | None = None,
) -> dict[str, object]:
    fields = {"path", "sha256", "artifact_identity", "baseline_identity"}
    if replay_identity is not None:
        fields.add("replay_artifact_identity")
    payload = _exact_mapping(value, label=label, fields=frozenset(fields))
    baseline = _identity(
        payload["baseline_identity"], label=f"{label}.baseline_identity"
    )
    artifact = _identity(
        payload["artifact_identity"], label=f"{label}.artifact_identity"
    )
    if baseline != dict(baseline_identity):
        raise CleanPruningSelectionEvidenceError(f"{label}.baseline_identity mismatch")
    if artifact_identity is not None and artifact != dict(artifact_identity):
        raise CleanPruningSelectionEvidenceError(f"{label}.artifact_identity mismatch")
    result: dict[str, object] = {
        "path": _safe_relative_json_path(payload["path"], label=f"{label}.path"),
        "sha256": _sha256(payload["sha256"], label=f"{label}.sha256"),
        "artifact_identity": artifact,
        "baseline_identity": baseline,
    }
    if replay_identity is not None:
        replay = _identity(
            payload["replay_artifact_identity"],
            label=f"{label}.replay_artifact_identity",
        )
        if replay != dict(replay_identity):
            raise CleanPruningSelectionEvidenceError(
                f"{label}.replay_artifact_identity mismatch"
            )
        result["replay_artifact_identity"] = replay
    return result


def _report_reference(
    value: object,
    *,
    label: str,
    baseline_identity: Mapping[str, str],
    artifact_identity: Mapping[str, str] | None = None,
) -> dict[str, object]:
    payload = _exact_mapping(
        value, label=label, fields=frozenset({"report", "runtime_manifest"})
    )
    return {
        "report": _bound_reference(
            payload["report"],
            label=f"{label}.report",
            baseline_identity=baseline_identity,
            artifact_identity=artifact_identity,
        ),
        "runtime_manifest": _sidecar_reference(
            payload["runtime_manifest"], label=f"{label}.runtime_manifest"
        ),
    }
