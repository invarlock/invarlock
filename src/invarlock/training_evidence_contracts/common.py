"""Shared closed-contract primitives for training artifact evidence.

This module is intentionally independent of the optional training runtime.  A
producer may use ``torch``, ``transformers`` and ``peft`` to create an edit,
but an evidence-pack verifier must be able to validate the resulting
provenance without importing any of those dependencies.

The contract is a binding sidecar for artifact replay, not an attestation of
historical optimizer execution. Callers supply the copied
``training_receipt.json`` together with a proof sidecar. The sidecar binds that
receipt to the evaluated baseline and subject identities, records independent
save/reload facts, and, for a LoRA merge, retains receipt-bound adapter state
and merge metadata. The profile-specific backend label is a constrained
producer declaration, not an attestation of historical execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping

from invarlock.core.checkpoint_identity import validated_model_identity
from invarlock.training_model_load import (
    TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA as TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA,
)
from invarlock.training_model_load import (
    load_diagnostics_sha256 as load_diagnostics_sha256,
)
from invarlock.training_protocol import (
    LORA_MERGE_PROOF_SCHEMA as LORA_MERGE_PROOF_SCHEMA,
)
from invarlock.training_protocol import (
    TRAINING_ARTIFACT_REPLAY_SCHEMA as TRAINING_ARTIFACT_REPLAY_SCHEMA,
)
from invarlock.training_protocol import (
    TRAINING_EVIDENCE_PROOF_SCHEMA as TRAINING_EVIDENCE_PROOF_SCHEMA,
)
from invarlock.training_protocol import (
    TRAINING_RECEIPT_SCHEMA as TRAINING_RECEIPT_SCHEMA,
)
from invarlock.training_protocol import (
    TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA as TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA,
)

TRAINING_ARTIFACT_REPLAY_PROVENANCE_KIND = "artifact_replay_verification"

_CANONICAL_PRODUCER_DECLARATIONS = {
    "fine_tune": "full_parameter_optimizer_training",
    "lora_merge": "peft_lora_train_and_merge",
}
TRAINING_EDIT_TYPES = frozenset(_CANONICAL_PRODUCER_DECLARATIONS)

_SHA256_RE = re.compile(r"sha256:[a-f0-9]{64}\Z")
_REVISION_RE = re.compile(r"[a-f0-9]{40,64}\Z")
_PROFILE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]{0,127}\Z")
_PATH_RE = re.compile(r"[^\x00\r\n]+\Z")

_PROOF_COMMON_FIELDS = frozenset(
    {
        "schema",
        "proof_sha256",
        "edit_type",
        "provenance",
        "training_receipt",
        "baseline_identity",
        "artifact_identity",
        "artifact_replay",
        "runtime_reload",
    }
)
_PROOF_LORA_FIELDS = _PROOF_COMMON_FIELDS | frozenset({"lora_merge"})

_PROVENANCE_FIELDS = frozenset({"kind", "producer_declared_training_backend"})
_RECEIPT_BINDING_FIELDS = frozenset(
    {
        "schema",
        "receipt_sha256",
        "profile_id",
        "profile_sha256",
        "edit_type",
        "dataset_provider",
    }
)
_ARTIFACT_REPLAY_FIELDS = frozenset(
    {
        "schema",
        "passed",
        "receipt_sha256",
        "baseline_identity",
        "artifact_identity",
        "baseline_tree_sha256",
        "subject_tree_sha256",
        "baseline_state_sha256",
        "post_training_state_sha256",
        "reloaded_subject_state_sha256",
        "delta_sha256",
        "changed_tensors",
        "changed_params",
        "total_params",
        "max_abs_delta",
        "baseline_load_diagnostics_sha256",
        "loss_function",
        "saved_artifact_verified",
        "reloaded_artifact_verified",
    }
)
_RUNTIME_RELOAD_FIELDS = frozenset(
    {
        "schema",
        "passed",
        "receipt_sha256",
        "artifact_identity",
        "subject_state_sha256",
        "reload_runs",
        "input_sha256",
        "logits_sha256",
        "logits_shape",
        "all_logits_finite",
        "repeat_deterministic",
        "device",
    }
)
_LORA_MERGE_FIELDS = frozenset(
    {
        "schema",
        "adapter_identity",
        "adapter_tree_sha256",
        "profile_lora_config_sha256",
        "serialized_adapter_config_sha256",
        "initial_adapter_state_sha256",
        "trained_adapter_state_sha256",
        "serialized_adapter_state_sha256",
        "base_state_before_adapter_sha256",
        "base_state_after_training_sha256",
        "base_state_manifest_sha256",
        "base_state_manifest_before_adapter_sha256",
        "base_state_manifest_after_training_sha256",
        "state_evidence_policy",
        "expected_merge_target_names_sha256",
        "merge_target_names",
        "observed_merged_changed_names_sha256",
        "merged_changed_tensor_count",
        "merge_scope_exact",
        "merged_state_sha256",
        "adapter_optimizer_steps",
        "trainable_parameter_count",
        "adapter_modules_before_merge",
        "adapter_modules_after_merge",
        "merge_method",
        "adapter_training_performed",
        "adapter_merge_performed",
    }
)

_RECEIPT_COMMON_FIELDS = frozenset(
    {
        "schema",
        "receipt_sha256",
        "profile_id",
        "profile_sha256",
        "edit_type",
        "dataset_provider",
        "model",
        "training_data",
        "optimizer",
        "training",
        "seed",
        "runtime",
        "hashes",
        "changes",
        "reload_smoke",
    }
)
_RECEIPT_LORA_FIELDS = _RECEIPT_COMMON_FIELDS | frozenset({"lora"})
_MODEL_FIELDS = frozenset(
    {"model_id", "model_revision", "tokenizer_sha256", "baseline_load"}
)
_BASELINE_LOAD_FIELDS = frozenset(
    {"loss_function", "diagnostics", "diagnostics_sha256"}
)
_LOAD_DIAGNOSTIC_FIELDS = frozenset(
    {
        "schema",
        "policy",
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    }
)
_DATA_FIELDS = frozenset(
    {"path", "sha256", "rows", "text_field", "token_count", "preprocessing_sha256"}
)
_OPTIMIZER_FIELDS = frozenset({"name", "learning_rate", "betas", "eps", "weight_decay"})
_TRAINING_FIELDS = frozenset(
    {
        "requested_steps",
        "completed_steps",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_sequence_length",
        "losses",
        "initial_loss",
        "final_loss",
        "optimization_performed",
        "training_data_used",
    }
)
_SEED_FIELDS = frozenset(
    {"python", "torch_cpu", "torch_cuda", "deterministic_algorithms"}
)
_RUNTIME_BASE_FIELDS = frozenset({"device", "dtype", "toolchain"})
_RUNTIME_IMAGE_FIELDS = _RUNTIME_BASE_FIELDS | frozenset({"container_image_digest"})
_HASH_FIELDS = frozenset(
    {
        "baseline_state_sha256",
        "baseline_tree_sha256",
        "pre_training_state_sha256",
        "post_training_state_sha256",
        "delta_sha256",
        "subject_tree_sha256",
        "reloaded_subject_state_sha256",
    }
)
_CHANGE_FIELDS = frozenset(
    {"changed_tensors", "changed_params", "total_params", "max_abs_delta"}
)
_RELOAD_SMOKE_FIELDS = frozenset(
    {
        "passed",
        "state_hash_matches",
        "inference_performed",
        "all_logits_finite",
        "repeat_runs",
        "input_sha256",
        "logits_sha256",
        "logits_shape",
        "device",
    }
)
_LORA_RECEIPT_FIELDS = frozenset(
    {
        "profile_lora_config_sha256",
        "serialized_adapter_config_sha256",
        "initial_adapter_state_sha256",
        "trained_adapter_state_sha256",
        "serialized_adapter_state_sha256",
        "adapter_tree_sha256",
        "base_state_before_adapter_sha256",
        "base_state_after_training_sha256",
        "base_state_manifest_sha256",
        "base_state_manifest_before_adapter_sha256",
        "base_state_manifest_after_training_sha256",
        "state_evidence_policy",
        "expected_merge_target_names_sha256",
        "merge_target_names",
        "observed_merged_changed_names_sha256",
        "merged_changed_tensor_count",
        "merge_scope_exact",
        "merged_state_sha256",
        "adapter_training_performed",
        "adapter_optimizer_steps",
        "trainable_parameter_count",
        "adapter_merge_performed",
        "adapter_modules_before_merge",
        "adapter_modules_after_merge",
        "merge_method",
    }
)


class TrainingEvidenceProofError(ValueError):
    """Raised when training-subject evidence does not meet the closed contract."""


def canonical_json_sha256(value: object) -> str:
    """Return the canonical JSON digest used for receipt and proof bindings."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainingEvidenceProofError(
            "training evidence cannot be canonicalized as JSON"
        ) from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical_training_receipt_sha256(receipt: Mapping[str, object]) -> str:
    """Return a receipt digest without its self-referential digest field."""

    payload = copy.deepcopy(dict(receipt))
    payload.pop("receipt_sha256", None)
    return canonical_json_sha256(payload)


def canonical_training_evidence_proof_sha256(proof: Mapping[str, object]) -> str:
    """Return a proof digest without its self-referential digest field."""

    payload = copy.deepcopy(dict(proof))
    payload.pop("proof_sha256", None)
    return canonical_json_sha256(payload)


def with_training_evidence_proof_digest(
    proof: Mapping[str, object],
) -> dict[str, object]:
    """Return a detached proof with its canonical self-digest populated."""

    payload = copy.deepcopy(dict(proof))
    payload["proof_sha256"] = canonical_training_evidence_proof_sha256(payload)
    return payload


def is_training_edit_type(value: object) -> bool:
    """Whether an edit type has a training-profile evidence contract."""

    return isinstance(value, str) and value in TRAINING_EDIT_TYPES


def canonical_producer_declared_training_backend(edit_type: str) -> str:
    """Return the fixed producer declaration for a training-profile edit type.

    This constrains the producer metadata to the selected profile family. It
    does not independently attest the producer's historical optimizer
    execution.
    """

    try:
        return _CANONICAL_PRODUCER_DECLARATIONS[edit_type]
    except KeyError as exc:
        raise TrainingEvidenceProofError(
            f"{edit_type!r} has no canonical training producer declaration"
        ) from exc


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _exact_mapping(
    value: object,
    *,
    label: str,
    fields: frozenset[str],
    errors: list[str],
) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        errors.append(f"{label} must be an object")
        return None
    observed = set(value)
    if observed != fields:
        missing = sorted(fields - observed)
        extra = sorted(observed - fields)
        errors.append(
            f"{label} has unbound, missing, or arbitrary fields "
            f"(missing={missing}, extra={extra})"
        )
    return value


def _is_text(value: object, *, pattern: re.Pattern[str] = _NAME_RE) -> bool:
    return (
        isinstance(value, str)
        and value == value.strip()
        and pattern.fullmatch(value) is not None
    )


def _identity(
    value: object,
    *,
    label: str,
    errors: list[str],
    allow_remote: bool,
) -> dict[str, str] | None:
    identity = validated_model_identity(value)
    if identity is None:
        errors.append(f"{label} must be a canonical model identity")
        return None
    if not allow_remote and identity["kind"] != "local_checkpoint_tree":
        errors.append(f"{label} must be a local_checkpoint_tree identity")
        return None
    return identity


def _adapter_identity(
    value: object, *, label: str, errors: list[str]
) -> dict[str, str] | None:
    mapping = _exact_mapping(
        value,
        label=label,
        fields=frozenset({"kind", "sha256"}),
        errors=errors,
    )
    if mapping is None:
        return None
    if mapping.get("kind") != "local_checkpoint_tree":
        errors.append(f"{label}.kind must be local_checkpoint_tree")
    digest = mapping.get("sha256")
    if not _is_sha256(digest):
        errors.append(f"{label}.sha256 must be a sha256 digest")
    if mapping.get("kind") != "local_checkpoint_tree" or not _is_sha256(digest):
        return None
    return {"kind": "local_checkpoint_tree", "sha256": str(digest)}


def _require_sha(
    mapping: Mapping[str, object], field: str, *, label: str, errors: list[str]
) -> str | None:
    value = mapping.get(field)
    if not _is_sha256(value):
        errors.append(f"{label}.{field} must be a sha256 digest")
        return None
    return str(value)
