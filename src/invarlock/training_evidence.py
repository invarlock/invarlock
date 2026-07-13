"""Fail-closed package validation for training-subject artifact evidence."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from invarlock.training_evidence_contracts.common import (
    _ARTIFACT_REPLAY_FIELDS,
    _LORA_MERGE_FIELDS,
    _PROOF_COMMON_FIELDS,
    _PROOF_LORA_FIELDS,
    _PROVENANCE_FIELDS,
    _RECEIPT_BINDING_FIELDS,
    _RUNTIME_RELOAD_FIELDS,
    LORA_MERGE_PROOF_SCHEMA,
    TRAINING_ARTIFACT_REPLAY_PROVENANCE_KIND,
    TRAINING_ARTIFACT_REPLAY_SCHEMA,
    TRAINING_EDIT_TYPES,
    TRAINING_EVIDENCE_PROOF_SCHEMA,
    TRAINING_RECEIPT_SCHEMA,
    TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA,
    TrainingEvidenceProofError,
    _adapter_identity,
    _exact_mapping,
    _identity,
    _is_sha256,
    _is_text,
    canonical_json_sha256,
    canonical_producer_declared_training_backend,
    canonical_training_evidence_proof_sha256,
    canonical_training_receipt_sha256,
    is_training_edit_type,
    with_training_evidence_proof_digest,
)
from invarlock.training_evidence_contracts.receipt import _receipt_errors


def _receipt_binding_errors(
    binding: object, *, receipt: Mapping[str, object], errors: list[str]
) -> None:
    payload = _exact_mapping(
        binding,
        label="training proof.training_receipt",
        fields=_RECEIPT_BINDING_FIELDS,
        errors=errors,
    )
    if payload is None:
        return
    for field in _RECEIPT_BINDING_FIELDS:
        if payload.get(field) != receipt.get(field):
            errors.append(
                f"training proof.training_receipt.{field} does not bind receipt"
            )


def _provenance_errors(
    provenance: object, *, edit_type: object, errors: list[str]
) -> None:
    payload = _exact_mapping(
        provenance,
        label="training proof.provenance",
        fields=_PROVENANCE_FIELDS,
        errors=errors,
    )
    if payload is None:
        return
    if payload.get("kind") != TRAINING_ARTIFACT_REPLAY_PROVENANCE_KIND:
        errors.append(
            "training proof.provenance.kind must be artifact_replay_verification"
        )
    declared_backend = payload.get("producer_declared_training_backend")
    if not _is_text(declared_backend):
        errors.append(
            "training proof.provenance.producer_declared_training_backend must be "
            "a non-empty canonical producer declaration"
        )
    elif isinstance(edit_type, str) and is_training_edit_type(edit_type):
        expected_backend = canonical_producer_declared_training_backend(edit_type)
        if declared_backend != expected_backend:
            errors.append(
                "training proof.provenance.producer_declared_training_backend must "
                "be the canonical producer declaration "
                f"{expected_backend!r} for {edit_type!r}"
            )


def _artifact_replay_errors(
    replay: object,
    *,
    receipt: Mapping[str, object],
    baseline_identity: dict[str, str] | None,
    artifact_identity: dict[str, str] | None,
    errors: list[str],
) -> None:
    payload = _exact_mapping(
        replay,
        label="training proof.artifact_replay",
        fields=_ARTIFACT_REPLAY_FIELDS,
        errors=errors,
    )
    if payload is None:
        return
    if payload.get("schema") != TRAINING_ARTIFACT_REPLAY_SCHEMA:
        errors.append("training proof.artifact_replay has an unrecognized schema")
    for field in ("passed", "saved_artifact_verified", "reloaded_artifact_verified"):
        if payload.get(field) is not True:
            errors.append(f"training proof.artifact_replay.{field} must be true")
    if payload.get("receipt_sha256") != receipt.get("receipt_sha256"):
        errors.append(
            "training proof.artifact_replay.receipt_sha256 does not bind receipt"
        )
    replay_baseline = _identity(
        payload.get("baseline_identity"),
        label="training proof.artifact_replay.baseline_identity",
        errors=errors,
        allow_remote=True,
    )
    replay_artifact = _identity(
        payload.get("artifact_identity"),
        label="training proof.artifact_replay.artifact_identity",
        errors=errors,
        allow_remote=False,
    )
    if baseline_identity is not None and replay_baseline != baseline_identity:
        errors.append("training proof.artifact_replay.baseline_identity mismatch")
    if artifact_identity is not None and replay_artifact != artifact_identity:
        errors.append("training proof.artifact_replay.artifact_identity mismatch")

    hashes = receipt.get("hashes")
    changes = receipt.get("changes")
    expected: dict[str, object] = {}
    if isinstance(hashes, Mapping):
        expected.update(
            {
                "baseline_tree_sha256": hashes.get("baseline_tree_sha256"),
                "subject_tree_sha256": hashes.get("subject_tree_sha256"),
                "baseline_state_sha256": hashes.get("baseline_state_sha256"),
                "post_training_state_sha256": hashes.get("post_training_state_sha256"),
                "reloaded_subject_state_sha256": hashes.get(
                    "reloaded_subject_state_sha256"
                ),
                "delta_sha256": hashes.get("delta_sha256"),
            }
        )
    if isinstance(changes, Mapping):
        expected.update(
            {
                "changed_tensors": changes.get("changed_tensors"),
                "changed_params": changes.get("changed_params"),
                "total_params": changes.get("total_params"),
                "max_abs_delta": changes.get("max_abs_delta"),
            }
        )
    model = receipt.get("model")
    if isinstance(model, Mapping):
        baseline_load = model.get("baseline_load")
        if isinstance(baseline_load, Mapping):
            expected.update(
                {
                    "baseline_load_diagnostics_sha256": baseline_load.get(
                        "diagnostics_sha256"
                    ),
                    "loss_function": baseline_load.get("loss_function"),
                }
            )
    for field, value in expected.items():
        if payload.get(field) != value:
            errors.append(
                f"training proof.artifact_replay.{field} does not bind receipt"
            )


def _runtime_reload_errors(
    runtime_reload: object,
    *,
    receipt: Mapping[str, object],
    artifact_identity: dict[str, str] | None,
    errors: list[str],
) -> None:
    payload = _exact_mapping(
        runtime_reload,
        label="training proof.runtime_reload",
        fields=_RUNTIME_RELOAD_FIELDS,
        errors=errors,
    )
    if payload is None:
        return
    if payload.get("schema") != TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA:
        errors.append("training proof.runtime_reload has an unrecognized schema")
    for field in ("passed", "all_logits_finite", "repeat_deterministic"):
        if payload.get(field) is not True:
            errors.append(f"training proof.runtime_reload.{field} must be true")
    if payload.get("receipt_sha256") != receipt.get("receipt_sha256"):
        errors.append(
            "training proof.runtime_reload.receipt_sha256 does not bind receipt"
        )
    runtime_artifact = _identity(
        payload.get("artifact_identity"),
        label="training proof.runtime_reload.artifact_identity",
        errors=errors,
        allow_remote=False,
    )
    if artifact_identity is not None and runtime_artifact != artifact_identity:
        errors.append("training proof.runtime_reload.artifact_identity mismatch")

    hashes = receipt.get("hashes")
    reload_smoke = receipt.get("reload_smoke")
    runtime = receipt.get("runtime")
    if isinstance(hashes, Mapping):
        for field in ("subject_state_sha256",):
            expected = hashes.get("post_training_state_sha256")
            if payload.get(field) != expected:
                errors.append(
                    f"training proof.runtime_reload.{field} does not bind receipt"
                )
    if isinstance(reload_smoke, Mapping):
        for field, receipt_field in (
            ("input_sha256", "input_sha256"),
            ("logits_sha256", "logits_sha256"),
            ("logits_shape", "logits_shape"),
        ):
            if payload.get(field) != reload_smoke.get(receipt_field):
                errors.append(
                    f"training proof.runtime_reload.{field} does not bind receipt"
                )
        if payload.get("reload_runs") != reload_smoke.get("repeat_runs"):
            errors.append(
                "training proof.runtime_reload.reload_runs does not bind receipt"
            )
    if isinstance(runtime, Mapping) and payload.get("device") != runtime.get("device"):
        errors.append("training proof.runtime_reload.device does not bind receipt")


def _lora_proof_errors(
    lora_proof: object,
    *,
    receipt: Mapping[str, object],
    errors: list[str],
) -> None:
    payload = _exact_mapping(
        lora_proof,
        label="training proof.lora_merge",
        fields=_LORA_MERGE_FIELDS,
        errors=errors,
    )
    if payload is None:
        return
    if payload.get("schema") != LORA_MERGE_PROOF_SCHEMA:
        errors.append("training proof.lora_merge has an unrecognized schema")
    _adapter_identity(
        payload.get("adapter_identity"),
        label="training proof.lora_merge.adapter_identity",
        errors=errors,
    )
    lora = receipt.get("lora")
    if not isinstance(lora, Mapping):
        errors.append("training proof.lora_merge requires a LoRA training receipt")
        return
    for field in (
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
    ):
        if payload.get(field) != lora.get(field):
            errors.append(f"training proof.lora_merge.{field} does not bind receipt")
    if payload.get("adapter_training_performed") is not True:
        errors.append(
            "training proof.lora_merge.adapter_training_performed must be true"
        )
    if payload.get("adapter_merge_performed") is not True:
        errors.append("training proof.lora_merge.adapter_merge_performed must be true")
    if payload.get("adapter_modules_after_merge") != 0:
        errors.append(
            "training proof.lora_merge.adapter_modules_after_merge must be zero"
        )
    if payload.get("merge_method") != "PeftModel.merge_and_unload":
        errors.append(
            "training proof.lora_merge.merge_method must be PeftModel.merge_and_unload"
        )


def training_evidence_proof_errors(
    proof: object,
    receipt: object,
    *,
    expected_edit_type: str | None = None,
    expected_baseline_identity: Mapping[str, object] | None = None,
    expected_artifact_identity: Mapping[str, object] | None = None,
) -> list[str]:
    """Return all closed-contract errors for a training artifact-replay proof.

    ``receipt`` is the staged, parsed ``training_receipt.json`` payload.  The
    verifier deliberately accepts it as an argument rather than a host path so
    pack integration can authenticate and parse the exact same file snapshot
    before dispatching this check.
    """

    receipt_errors, receipt_mapping = _receipt_errors(receipt)
    errors = list(receipt_errors)
    if not isinstance(proof, Mapping) or not all(isinstance(key, str) for key in proof):
        return [*errors, "training evidence proof must be an object"]

    edit_type = proof.get("edit_type")
    expected_fields = (
        _PROOF_LORA_FIELDS if edit_type == "lora_merge" else _PROOF_COMMON_FIELDS
    )
    _exact_mapping(
        proof,
        label="training evidence proof",
        fields=expected_fields,
        errors=errors,
    )
    if proof.get("schema") != TRAINING_EVIDENCE_PROOF_SCHEMA:
        errors.append("training evidence proof has an unrecognized schema")
    if not is_training_edit_type(edit_type):
        errors.append(
            "training evidence proof edit_type must be fine_tune or lora_merge"
        )
    if expected_edit_type is not None:
        if not is_training_edit_type(expected_edit_type):
            errors.append("expected edit type is not a training-profile edit type")
        elif edit_type != expected_edit_type:
            errors.append(
                "training evidence proof edit_type does not match expected edit type"
            )

    proof_digest = proof.get("proof_sha256")
    if not _is_sha256(proof_digest):
        errors.append("training evidence proof.proof_sha256 must be a sha256 digest")
    else:
        try:
            expected_digest = canonical_training_evidence_proof_sha256(proof)
        except TrainingEvidenceProofError as exc:
            errors.append(str(exc))
        else:
            if proof_digest != expected_digest:
                errors.append(
                    "training evidence proof.proof_sha256 does not bind content"
                )

    _provenance_errors(proof.get("provenance"), edit_type=edit_type, errors=errors)
    baseline_identity = _identity(
        proof.get("baseline_identity"),
        label="training evidence proof.baseline_identity",
        errors=errors,
        allow_remote=True,
    )
    artifact_identity = _identity(
        proof.get("artifact_identity"),
        label="training evidence proof.artifact_identity",
        errors=errors,
        allow_remote=False,
    )
    if receipt_mapping is not None:
        _receipt_binding_errors(
            proof.get("training_receipt"), receipt=receipt_mapping, errors=errors
        )
        model = receipt_mapping.get("model")
        if (
            baseline_identity is not None
            and baseline_identity["kind"] == "remote_revision"
            and isinstance(model, Mapping)
            and baseline_identity.get("revision") != model.get("model_revision")
        ):
            errors.append(
                "training evidence proof.baseline_identity does not bind receipt revision"
            )
        _artifact_replay_errors(
            proof.get("artifact_replay"),
            receipt=receipt_mapping,
            baseline_identity=baseline_identity,
            artifact_identity=artifact_identity,
            errors=errors,
        )
        _runtime_reload_errors(
            proof.get("runtime_reload"),
            receipt=receipt_mapping,
            artifact_identity=artifact_identity,
            errors=errors,
        )
        if edit_type == "lora_merge":
            _lora_proof_errors(
                proof.get("lora_merge"), receipt=receipt_mapping, errors=errors
            )
        elif "lora_merge" in proof:
            errors.append("fine_tune proof must not carry a LoRA merge proof")

    if expected_baseline_identity is not None:
        expected_baseline_errors: list[str] = []
        expected_baseline = _identity(
            expected_baseline_identity,
            label="expected baseline identity",
            errors=expected_baseline_errors,
            allow_remote=True,
        )
        errors.extend(expected_baseline_errors)
        if expected_baseline is not None and baseline_identity != expected_baseline:
            errors.append(
                "training evidence proof baseline_identity does not match expected baseline"
            )
    if expected_artifact_identity is not None:
        expected_artifact_errors: list[str] = []
        expected_artifact = _identity(
            expected_artifact_identity,
            label="expected artifact identity",
            errors=expected_artifact_errors,
            allow_remote=False,
        )
        errors.extend(expected_artifact_errors)
        if expected_artifact is not None and artifact_identity != expected_artifact:
            errors.append(
                "training evidence proof artifact_identity does not match expected artifact"
            )
    return errors


def require_valid_training_evidence_proof(
    proof: object,
    receipt: object,
    *,
    expected_edit_type: str | None = None,
    expected_baseline_identity: Mapping[str, object] | None = None,
    expected_artifact_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a detached valid proof or raise a fail-closed validation error."""

    errors = training_evidence_proof_errors(
        proof,
        receipt,
        expected_edit_type=expected_edit_type,
        expected_baseline_identity=expected_baseline_identity,
        expected_artifact_identity=expected_artifact_identity,
    )
    if errors:
        raise TrainingEvidenceProofError("; ".join(errors))
    assert isinstance(proof, Mapping)  # established above when no errors exist
    return copy.deepcopy(dict(proof))


__all__ = [
    "LORA_MERGE_PROOF_SCHEMA",
    "TRAINING_ARTIFACT_REPLAY_SCHEMA",
    "TRAINING_ARTIFACT_REPLAY_PROVENANCE_KIND",
    "TRAINING_EDIT_TYPES",
    "TRAINING_EVIDENCE_PROOF_SCHEMA",
    "TRAINING_RECEIPT_SCHEMA",
    "TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA",
    "TrainingEvidenceProofError",
    "canonical_producer_declared_training_backend",
    "canonical_json_sha256",
    "canonical_training_evidence_proof_sha256",
    "canonical_training_receipt_sha256",
    "is_training_edit_type",
    "require_valid_training_evidence_proof",
    "training_evidence_proof_errors",
    "with_training_evidence_proof_digest",
]
