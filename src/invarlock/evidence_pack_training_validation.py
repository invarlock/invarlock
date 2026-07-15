"""LoRA and full-fine-tuning receipt and proof validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from invarlock.evidence_pack_edit_common import (
    TRAINING_EVIDENCE_PROOF_SIDECAR,
    TRAINING_PROFILE_SNAPSHOT_SCHEMA,
    TRAINING_RECEIPT_SIDECAR,
    _load_json_sidecar,
    _same_finite_number,
)
from invarlock.evidence_pack_json import (
    StrictJsonError,
    read_json_object_snapshot,
    sha256_prefixed,
)
from invarlock.evidence_pack_scenario_contract import (
    ProofHandler,
    ScenarioContract,
)
from invarlock.training_evidence import (
    TrainingEvidenceProofError,
    require_valid_training_evidence_proof,
)
from invarlock.training_evidence_contracts.common import canonical_json_sha256
from invarlock.training_model_load import (
    TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA,
    load_diagnostics_sha256,
)

_DATASET_PROVIDER_SNAPSHOT_SCHEMA = "invarlock.dataset-provider-input.v1"


def _training_canonical_digest(payload: object) -> str | None:
    """Return the canonical digest shared by immutable training contracts."""

    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _training_profile_digest(profile: object) -> str | None:
    """Return the immutable profile digest used by the training producer."""

    if not isinstance(profile, dict):
        return None
    payload = dict(profile)
    payload.pop("profile_sha256", None)
    return _training_canonical_digest(payload)


def _dataset_provider_policy_errors(
    *, pack_dir: Path, prefix: str, receipt: dict[str, Any]
) -> list[str]:
    """Bind a training receipt to the pack's sealed dataset-provider policy.

    The receipt's provider hash proves only that its own provider coordinates
    were serialized consistently.  The pack snapshot is the independently
    sealed policy identity, so a receipt and proof rehashed around another
    provider must not become acceptable.
    """

    snapshot_path = pack_dir / "metadata" / "dataset_provider.json"
    try:
        _, snapshot = read_json_object_snapshot(
            snapshot_path, label="dataset provider policy snapshot"
        )
    except (OSError, StrictJsonError) as exc:
        return [prefix + "dataset provider policy snapshot is unavailable: " + str(exc)]

    if set(snapshot) != {"schema", "provider", "provider_sha256"}:
        return [prefix + "dataset provider policy snapshot shape is invalid"]
    if snapshot.get("schema") != _DATASET_PROVIDER_SNAPSHOT_SCHEMA:
        return [prefix + "dataset provider policy snapshot schema is invalid"]
    provider = snapshot.get("provider")
    if not isinstance(provider, dict) or not provider:
        return [prefix + "dataset provider policy snapshot provider is invalid"]
    provider_sha256 = snapshot.get("provider_sha256")
    if provider_sha256 != canonical_json_sha256(provider):
        return [
            prefix
            + "dataset provider policy snapshot provider digest does not bind content"
        ]
    expected_binding = {
        "provider": provider,
        "provider_sha256": provider_sha256,
    }
    if receipt.get("dataset_provider") != expected_binding:
        return [
            prefix
            + "training receipt dataset provider does not bind sealed provider policy"
        ]
    return []


def _training_model_load_binding_errors(
    *, prefix: str, profile: dict[str, Any], receipt_model: object
) -> list[str]:
    errors: list[str] = []
    model_load = profile.get("model_load")
    expected_unexpected_keys: list[str] | None = None
    if not isinstance(model_load, dict):
        errors.append(prefix + "training profile model_load must be an object")
    elif set(model_load) != {"loss_function", "expected_unexpected_keys"}:
        errors.append(prefix + "training profile model_load shape is invalid")
    else:
        if model_load.get("loss_function") != "ForCausalLM":
            errors.append(
                prefix + "training profile model_load loss_function is invalid"
            )
        unexpected = model_load.get("expected_unexpected_keys")
        if (
            not isinstance(unexpected, list)
            or unexpected != sorted(set(unexpected))
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in unexpected
            )
        ):
            errors.append(
                prefix
                + "training profile model_load expected_unexpected_keys is invalid"
            )
        else:
            expected_unexpected_keys = unexpected

    if not isinstance(receipt_model, dict):
        return errors
    baseline_load = receipt_model.get("baseline_load")
    if not isinstance(baseline_load, dict):
        return errors + [prefix + "training receipt baseline_load must be an object"]
    if set(baseline_load) != {"loss_function", "diagnostics", "diagnostics_sha256"}:
        return errors + [prefix + "training receipt baseline_load shape is invalid"]
    if isinstance(model_load, dict) and baseline_load.get(
        "loss_function"
    ) != model_load.get("loss_function"):
        errors.append(
            prefix + "training profile model_load loss_function does not bind receipt"
        )
    diagnostics = baseline_load.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return errors + [
            prefix + "training receipt baseline_load diagnostics must be an object"
        ]
    if set(diagnostics) != {
        "schema",
        "policy",
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    }:
        return errors + [
            prefix + "training receipt baseline_load diagnostics shape is invalid"
        ]
    if diagnostics.get("schema") != TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA:
        errors.append(
            prefix + "training receipt baseline_load diagnostics schema is invalid"
        )
    if diagnostics.get("policy") != "exact_source_key_migration":
        errors.append(
            prefix + "training receipt baseline_load diagnostics policy is invalid"
        )
    for field in ("missing_keys", "mismatched_keys", "error_msgs"):
        if diagnostics.get(field) != []:
            errors.append(
                prefix
                + f"training receipt baseline_load diagnostics {field} must be empty"
            )
    if (
        expected_unexpected_keys is not None
        and diagnostics.get("unexpected_keys") != expected_unexpected_keys
    ):
        errors.append(
            prefix
            + "training profile model_load expected_unexpected_keys does not bind receipt"
        )
    if baseline_load.get("diagnostics_sha256") != load_diagnostics_sha256(diagnostics):
        errors.append(
            prefix + "training receipt baseline_load diagnostics do not bind digest"
        )
    return errors


def _training_edit_parameter_errors(
    *,
    prefix: str,
    contract: ScenarioContract,
    profile: dict[str, Any],
    receipt: dict[str, Any],
    optimizer: object,
) -> list[str]:
    assert contract.edit is not None
    params = contract.edit.parameter_dict
    if contract.edit.edit_type.value == "lora_merge":
        errors: list[str] = []
        lora = profile.get("lora")
        receipt_lora = receipt.get("lora")
        if not isinstance(lora, dict):
            return [prefix + "training profile LoRA block must be an object"]
        if set(lora) != {
            "rank",
            "alpha",
            "dropout",
            "target_modules",
            "bias",
            "task_type",
            "fan_in_fan_out",
        }:
            errors.append(prefix + "training profile LoRA block shape is invalid")
        if lora.get("rank") != params.get("rank"):
            errors.append(prefix + "training profile LoRA rank mismatch")
        if not _same_finite_number(lora.get("alpha"), params.get("alpha")):
            errors.append(prefix + "training profile LoRA alpha mismatch")
        expected_lora_digest = _training_canonical_digest(lora)
        if (
            not isinstance(receipt_lora, dict)
            or receipt_lora.get("profile_lora_config_sha256") != expected_lora_digest
        ):
            errors.append(
                prefix + "training profile LoRA configuration does not bind receipt"
            )
        return errors
    if not isinstance(optimizer, dict):
        return []
    errors = []
    if not _same_finite_number(
        optimizer.get("learning_rate"), params.get("learning_rate")
    ):
        errors.append(prefix + "training profile learning_rate mismatch")
    if profile.get("steps") != params.get("steps"):
        errors.append(prefix + "training profile steps mismatch")
    return errors


def _training_profile_snapshot_errors(
    *,
    pack_dir: Path,
    scenario_id: str,
    contract: ScenarioContract,
    receipt: dict[str, Any],
) -> list[str]:
    """Bind typed training parameters to a sealed profile snapshot and receipt.

    A receipt proves that a particular immutable profile was used, while the
    scenario says which rank/alpha or learning-rate/step contract the report
    represents.  Neither may be interpreted in isolation: the package carries
    a byte-digested profile snapshot and this verifier checks all three views.
    The explicit snapshot ``scope`` is a reviewed scenario policy, never an
    inferred interpretation of adapter module names.
    """

    prefix = f"{scenario_id}: "
    if contract.training_profile is None:
        return [prefix + "training scenario has no profile snapshot binding"]
    if contract.edit is None or contract.edit.scope is None:
        return [prefix + "training scenario has no typed parameter scope"]

    binding = contract.training_profile
    snapshot_path = pack_dir / binding.snapshot_path
    try:
        snapshot_bytes, snapshot = read_json_object_snapshot(
            snapshot_path,
            label="training profile snapshot",
        )
    except (OSError, StrictJsonError) as exc:
        return [prefix + "training profile snapshot is unavailable: " + str(exc)]

    errors: list[str] = []
    if sha256_prefixed(snapshot_bytes) != binding.snapshot_sha256:
        errors.append(prefix + "training profile snapshot digest mismatch")
    expected_snapshot_fields = {
        "schema",
        "profile_id",
        "profile_sha256",
        "scope",
        "profile",
    }
    if set(snapshot) != expected_snapshot_fields:
        errors.append(
            prefix + "training profile snapshot has missing or unsupported fields"
        )
        return errors
    if snapshot.get("schema") != TRAINING_PROFILE_SNAPSHOT_SCHEMA:
        errors.append(prefix + "training profile snapshot schema mismatch")
    if snapshot.get("profile_id") != binding.profile_id:
        errors.append(prefix + "training profile snapshot profile_id mismatch")
    if snapshot.get("profile_sha256") != binding.profile_sha256:
        errors.append(prefix + "training profile snapshot profile_sha256 mismatch")
    if snapshot.get("scope") != contract.edit.scope:
        errors.append(prefix + "training profile snapshot scope mismatch")

    profile = snapshot.get("profile")
    if not isinstance(profile, dict):
        errors.append(prefix + "training profile snapshot profile must be an object")
        return errors
    expected_profile_fields = {
        "profile_sha256",
        "edit_type",
        "model_id",
        "model_revision",
        "training_data",
        "optimizer",
        "steps",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_sequence_length",
        "seed",
        "deterministic_algorithms",
        "device",
        "dtype",
        "toolchain",
        "model_load",
    }
    if contract.edit.edit_type.value == "lora_merge":
        expected_profile_fields.add("lora")
    if set(profile) != expected_profile_fields:
        errors.append(prefix + "training profile snapshot profile shape is invalid")
        return errors
    if profile.get("profile_sha256") != binding.profile_sha256:
        errors.append(
            prefix + "training profile snapshot profile digest field mismatch"
        )
    if _training_profile_digest(profile) != binding.profile_sha256:
        errors.append(
            prefix + "training profile snapshot profile digest does not bind content"
        )
    if profile.get("edit_type") != contract.edit.edit_type.value:
        errors.append(prefix + "training profile edit_type mismatch")

    optimizer = profile.get("optimizer")
    training = receipt.get("training")
    receipt_optimizer = receipt.get("optimizer")
    receipt_model = receipt.get("model")
    receipt_data = receipt.get("training_data")
    if receipt.get("profile_id") != binding.profile_id:
        errors.append(prefix + "training receipt profile_id mismatch")
    if receipt.get("profile_sha256") != binding.profile_sha256:
        errors.append(prefix + "training receipt profile_sha256 mismatch")
    if receipt.get("edit_type") != contract.edit.edit_type.value:
        errors.append(prefix + "training receipt edit_type mismatch")

    if not isinstance(optimizer, dict):
        errors.append(prefix + "training profile optimizer must be an object")
    if not isinstance(training, dict):
        errors.append(prefix + "training receipt training block must be an object")
    if not isinstance(receipt_optimizer, dict):
        errors.append(prefix + "training receipt optimizer block must be an object")
    if not isinstance(receipt_model, dict):
        errors.append(prefix + "training receipt model block must be an object")
    if not isinstance(receipt_data, dict):
        errors.append(prefix + "training receipt training_data block must be an object")

    errors.extend(
        _training_model_load_binding_errors(
            prefix=prefix, profile=profile, receipt_model=receipt_model
        )
    )

    if isinstance(receipt_model, dict):
        for field in ("model_id", "model_revision"):
            if profile.get(field) != receipt_model.get(field):
                errors.append(
                    prefix + f"training profile {field} does not bind receipt"
                )
    if isinstance(receipt_data, dict) and isinstance(
        profile.get("training_data"), dict
    ):
        profile_data = cast(dict[str, Any], profile["training_data"])
        for field in ("path", "sha256", "rows", "text_field"):
            if profile_data.get(field) != receipt_data.get(field):
                errors.append(
                    prefix
                    + f"training profile training_data.{field} does not bind receipt"
                )
    if isinstance(training, dict):
        for field in (
            "requested_steps",
            "completed_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_sequence_length",
        ):
            profile_field = (
                "steps" if field in {"requested_steps", "completed_steps"} else field
            )
            if profile.get(profile_field) != training.get(field):
                errors.append(
                    prefix + f"training profile {profile_field} does not bind receipt"
                )
    if isinstance(optimizer, dict) and isinstance(receipt_optimizer, dict):
        for field in ("name", "betas", "eps", "weight_decay"):
            if optimizer.get(field) != receipt_optimizer.get(field):
                errors.append(
                    prefix + f"training profile optimizer.{field} does not bind receipt"
                )
        if not _same_finite_number(
            optimizer.get("learning_rate"), receipt_optimizer.get("learning_rate")
        ):
            errors.append(
                prefix
                + "training profile optimizer.learning_rate does not bind receipt"
            )

    errors.extend(
        _training_edit_parameter_errors(
            prefix=prefix,
            contract=contract,
            profile=profile,
            receipt=receipt,
            optimizer=optimizer,
        )
    )
    return errors


def _require_training_evidence_proof(
    *,
    pack_dir: Path,
    scenario_id: str,
    contract: ScenarioContract,
    report_dir: Path,
    report: dict[str, Any],
) -> list[str]:
    """Require a receipt-bound artifact-replay proof for a training subject."""

    prefix = f"{scenario_id}: "
    if contract.proof_handler is not ProofHandler.EXTERNAL_TRAINING:
        return [prefix + "internal training proof dispatch mismatch"]
    if contract.edit is None:
        return [prefix + "training scenario has no typed edit contract"]

    receipt_path = report_dir / TRAINING_RECEIPT_SIDECAR
    proof_path = report_dir / TRAINING_EVIDENCE_PROOF_SIDECAR
    errors: list[str] = []
    if not receipt_path.is_file() or receipt_path.is_symlink():
        errors.append(prefix + "training receipt sidecar missing")
    if not proof_path.is_file() or proof_path.is_symlink():
        errors.append(prefix + "training evidence proof sidecar missing")
    if errors:
        return errors

    receipt, receipt_error = _load_json_sidecar(receipt_path)
    proof, proof_error = _load_json_sidecar(proof_path)
    if receipt_error is not None or receipt is None:
        errors.append(prefix + "training receipt sidecar is invalid")
    if proof_error is not None or proof is None:
        errors.append(prefix + "training evidence proof sidecar is invalid")
    if errors:
        return errors
    assert receipt is not None
    assert proof is not None

    errors.extend(
        _dataset_provider_policy_errors(
            pack_dir=pack_dir, prefix=prefix, receipt=receipt
        )
    )

    metadata, metadata_error = _load_json_sidecar(report_dir / "edit_metadata.json")
    if metadata_error is not None or metadata is None:
        return [prefix + "training edit metadata sidecar is invalid"]
    coverage = metadata.get("coverage")
    changes = receipt.get("changes")
    replay = proof.get("artifact_replay")
    if not isinstance(coverage, dict) or not isinstance(changes, dict):
        errors.append(prefix + "training edit coverage binding is unavailable")
    else:
        for metadata_field, receipt_field in (
            ("edited_tensors", "changed_tensors"),
            ("edited_params", "changed_params"),
            ("total_params", "total_params"),
        ):
            if coverage.get(metadata_field) != changes.get(receipt_field):
                errors.append(
                    prefix
                    + f"training edit coverage.{metadata_field} does not bind receipt"
                )
        changed_params = changes.get("changed_params")
        total_params = changes.get("total_params")
        expected_ratio = (
            changed_params / total_params
            if isinstance(changed_params, int)
            and not isinstance(changed_params, bool)
            and isinstance(total_params, int)
            and not isinstance(total_params, bool)
            and total_params > 0
            else None
        )
        if not _same_finite_number(coverage.get("coverage_ratio"), expected_ratio):
            errors.append(
                prefix + "training edit coverage.coverage_ratio does not bind receipt"
            )
    if isinstance(replay, dict) and isinstance(changes, dict):
        for field in ("changed_tensors", "changed_params", "total_params"):
            if replay.get(field) != changes.get(field):
                errors.append(
                    prefix + f"training artifact replay {field} does not bind receipt"
                )

    dataset = report.get("dataset")
    provider = {
        output_key: dataset.get(report_key)
        for output_key, report_key in (
            ("kind", "provider"),
            ("dataset_name", "dataset_name"),
            ("config_name", "config_name"),
            ("revision", "revision"),
        )
        if isinstance(dataset, dict) and dataset.get(report_key) is not None
    }
    expected_provider_binding = {
        "provider": provider,
        "provider_sha256": canonical_json_sha256(provider),
    }
    if report and (
        not provider or receipt.get("dataset_provider") != expected_provider_binding
    ):
        errors.append(prefix + "training receipt dataset provider does not bind report")

    report_meta = report.get("meta")
    report_identity = (
        report_meta.get("model_identity") if isinstance(report_meta, dict) else None
    )
    baseline_ref = report.get("baseline_ref")
    baseline_identity = (
        baseline_ref.get("model_identity") if isinstance(baseline_ref, dict) else None
    )
    if not isinstance(report_identity, dict):
        errors.append(prefix + "evaluation subject identity missing")
    if not isinstance(baseline_identity, dict):
        errors.append(prefix + "evaluation baseline identity missing")
    if errors:
        return errors

    try:
        require_valid_training_evidence_proof(
            proof,
            receipt,
            expected_edit_type=contract.edit.edit_type.value,
            expected_baseline_identity=baseline_identity,
            expected_artifact_identity=report_identity,
        )
    except TrainingEvidenceProofError as exc:
        errors.append(prefix + "training evidence proof is invalid: " + str(exc))
    errors.extend(
        _training_profile_snapshot_errors(
            pack_dir=pack_dir,
            scenario_id=scenario_id,
            contract=contract,
            receipt=receipt,
        )
    )
    return errors
