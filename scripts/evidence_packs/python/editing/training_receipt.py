"""Fail-closed receipts for real fine-tune and trained LoRA-merge edits."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any

from invarlock.training_model_load import (
    TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA,
    load_diagnostics_sha256,
)
from invarlock.training_protocol import TRAINING_RECEIPT_SCHEMA

from .training_contract import (
    FineTuneTrainingProfile,
    LoraTrainingProfile,
    TrainingProfile,
    canonical_sha256,
    lora_config_digest,
)

_SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_TOP_LEVEL_KEYS = set(
    "schema receipt_sha256 profile_id profile_sha256 edit_type model "
    "dataset_provider training_data optimizer training seed runtime hashes changes "
    "reload_smoke lora".split()
)
_MODEL_KEYS = {"model_id", "model_revision", "tokenizer_sha256", "baseline_load"}
_BASELINE_LOAD_KEYS = {"loss_function", "diagnostics", "diagnostics_sha256"}
_LOAD_DIAGNOSTIC_KEYS = {
    "schema",
    "policy",
    "missing_keys",
    "unexpected_keys",
    "mismatched_keys",
    "error_msgs",
}
_DATA_KEYS = set("path sha256 rows text_field token_count preprocessing_sha256".split())
_OPTIMIZER_KEYS = {"name", "learning_rate", "betas", "eps", "weight_decay"}
_TRAINING_KEYS = set(
    "requested_steps completed_steps micro_batch_size gradient_accumulation_steps "
    "max_sequence_length losses initial_loss final_loss optimization_performed "
    "training_data_used".split()
)
_SEED_KEYS = {"python", "torch_cpu", "torch_cuda", "deterministic_algorithms"}
_RUNTIME_KEYS = {"device", "dtype", "toolchain", "container_image_digest"}
_HASH_KEYS = set(
    "baseline_state_sha256 baseline_tree_sha256 pre_training_state_sha256 "
    "post_training_state_sha256 delta_sha256 subject_tree_sha256 "
    "reloaded_subject_state_sha256".split()
)
_CHANGE_KEYS = {"changed_tensors", "changed_params", "total_params", "max_abs_delta"}
_RELOAD_KEYS = {
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
_LORA_RECEIPT_KEYS = set(
    "profile_lora_config_sha256 serialized_adapter_config_sha256 "
    "initial_adapter_state_sha256 trained_adapter_state_sha256 "
    "serialized_adapter_state_sha256 adapter_tree_sha256 base_state_before_adapter_sha256 "
    "base_state_after_training_sha256 merged_state_sha256 "
    "adapter_training_performed adapter_optimizer_steps trainable_parameter_count "
    "adapter_merge_performed adapter_modules_before_merge "
    "adapter_modules_after_merge merge_method base_state_manifest_sha256 "
    "base_state_manifest_before_adapter_sha256 "
    "base_state_manifest_after_training_sha256 state_evidence_policy "
    "expected_merge_target_names_sha256 observed_merged_changed_names_sha256 "
    "merge_target_names merged_changed_tensor_count merge_scope_exact".split()
)


class TrainingReceiptError(ValueError):
    """Raised when a training-profile receipt cannot support its claimed edit."""


def canonical_receipt_digest(receipt: Mapping[str, Any]) -> str:
    """Hash a receipt without its self-referential ``receipt_sha256`` field."""

    payload = copy.deepcopy(dict(receipt))
    payload.pop("receipt_sha256", None)
    return canonical_sha256(payload)


def with_receipt_digest(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached receipt with its canonical digest populated."""

    payload = copy.deepcopy(dict(receipt))
    payload["receipt_sha256"] = canonical_receipt_digest(payload)
    return payload


def _is_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _mapping(
    errors: list[str], value: Any, *, path: str, allowed: set[str]
) -> Mapping[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return None
    unknown = sorted(set(value) - allowed)
    if unknown:
        errors.append(f"{path} contains unsupported field(s): {', '.join(unknown)}")
    return value


def _require_sha_fields(
    errors: list[str], value: Mapping[str, Any], *, path: str, fields: set[str]
) -> None:
    for field in sorted(fields):
        if not _is_sha256(value.get(field)):
            errors.append(f"{path}.{field} must be a canonical sha256 digest")


def _same_float(left: Any, right: float) -> bool:
    parsed = _finite_float(left)
    return parsed is not None and math.isclose(
        parsed, right, rel_tol=1e-12, abs_tol=1e-15
    )


def _model_errors(receipt: Mapping[str, Any], profile: TrainingProfile) -> list[str]:
    errors: list[str] = []
    model = _mapping(errors, receipt.get("model"), path="model", allowed=_MODEL_KEYS)
    if model is None:
        return errors
    if model.get("model_id") != profile.model_id:
        errors.append("model.model_id does not match the immutable profile")
    if model.get("model_revision") != profile.model_revision:
        errors.append("model.model_revision does not match the immutable profile")
    if not _is_sha256(model.get("tokenizer_sha256")):
        errors.append("model.tokenizer_sha256 must be a canonical sha256 digest")
    baseline_load = _mapping(
        errors,
        model.get("baseline_load"),
        path="model.baseline_load",
        allowed=_BASELINE_LOAD_KEYS,
    )
    if baseline_load is None:
        return errors
    if baseline_load.get("loss_function") != profile.model_load.loss_function:
        errors.append("model.baseline_load.loss_function does not match the profile")
    diagnostics = _mapping(
        errors,
        baseline_load.get("diagnostics"),
        path="model.baseline_load.diagnostics",
        allowed=_LOAD_DIAGNOSTIC_KEYS,
    )
    if diagnostics is None:
        return errors
    if diagnostics.get("schema") != TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA:
        errors.append("model.baseline_load.diagnostics schema is invalid")
    if diagnostics.get("policy") != "exact_source_key_migration":
        errors.append("model.baseline_load.diagnostics policy is invalid")
    for field in ("missing_keys", "mismatched_keys", "error_msgs"):
        if diagnostics.get(field) != []:
            errors.append(f"model.baseline_load.diagnostics.{field} must be empty")
    if diagnostics.get("unexpected_keys") != list(
        profile.model_load.expected_unexpected_keys
    ):
        errors.append(
            "model.baseline_load.diagnostics.unexpected_keys does not match the profile"
        )
    digest = baseline_load.get("diagnostics_sha256")
    if not _is_sha256(digest):
        errors.append(
            "model.baseline_load.diagnostics_sha256 must be a canonical sha256 digest"
        )
    elif digest != load_diagnostics_sha256(diagnostics):
        errors.append("model.baseline_load diagnostics do not match their digest")
    return errors


def _training_data_errors(
    receipt: Mapping[str, Any], profile: TrainingProfile
) -> list[str]:
    errors: list[str] = []
    data = _mapping(
        errors,
        receipt.get("training_data"),
        path="training_data",
        allowed=_DATA_KEYS,
    )
    if data is None:
        return errors
    expected = profile.training_data
    for field, value in {
        "path": expected.path,
        "sha256": expected.sha256,
        "rows": expected.rows,
        "text_field": expected.text_field,
    }.items():
        if data.get(field) != value:
            errors.append(f"training_data.{field} does not match the profile")
    if not _is_int(data.get("token_count"), minimum=1):
        errors.append("training_data.token_count must be a positive integer")
    if not _is_sha256(data.get("preprocessing_sha256")):
        errors.append(
            "training_data.preprocessing_sha256 must be a canonical sha256 digest"
        )
    return errors


def _optimizer_errors(
    receipt: Mapping[str, Any], profile: TrainingProfile
) -> list[str]:
    errors: list[str] = []
    optimizer = _mapping(
        errors,
        receipt.get("optimizer"),
        path="optimizer",
        allowed=_OPTIMIZER_KEYS,
    )
    if optimizer is None:
        return errors
    expected = profile.optimizer
    if optimizer.get("name") != expected.name:
        errors.append("optimizer.name does not match the profile")
    for field, expected_value in {
        "learning_rate": expected.learning_rate,
        "eps": expected.eps,
        "weight_decay": expected.weight_decay,
    }.items():
        if not _same_float(optimizer.get(field), expected_value):
            errors.append(f"optimizer.{field} does not match the profile")
    betas = optimizer.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        errors.append("optimizer.betas must contain exactly two values")
    elif not all(
        _same_float(observed, expected_value)
        for observed, expected_value in zip(betas, expected.betas, strict=True)
    ):
        errors.append("optimizer.betas do not match the profile")
    return errors


def _training_errors(receipt: Mapping[str, Any], profile: TrainingProfile) -> list[str]:
    errors: list[str] = []
    training = _mapping(
        errors,
        receipt.get("training"),
        path="training",
        allowed=_TRAINING_KEYS,
    )
    if training is None:
        return errors
    expected_fields = {
        "requested_steps": profile.steps,
        "completed_steps": profile.steps,
        "micro_batch_size": profile.micro_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "max_sequence_length": profile.max_sequence_length,
    }
    for field, expected in expected_fields.items():
        if training.get(field) != expected:
            errors.append(f"training.{field} does not match the profile")
    completed_steps = training.get("completed_steps")
    if not _is_int(completed_steps, minimum=1):
        errors.append("training.completed_steps must be a positive integer")
    if training.get("optimization_performed") is not True:
        errors.append("training.optimization_performed must be true")
    if training.get("training_data_used") is not True:
        errors.append("training.training_data_used must be true")

    losses = training.get("losses")
    parsed_losses: list[float] = []
    if not isinstance(losses, list) or not losses:
        errors.append("training.losses must be a non-empty array")
    else:
        for index, loss in enumerate(losses):
            parsed = _finite_float(loss)
            if parsed is None:
                errors.append(f"training.losses[{index}] must be finite")
            else:
                parsed_losses.append(parsed)
        if _is_int(completed_steps, minimum=1) and len(losses) != completed_steps:
            errors.append("training.losses must contain one value per completed step")
    initial = _finite_float(training.get("initial_loss"))
    final = _finite_float(training.get("final_loss"))
    if initial is None:
        errors.append("training.initial_loss must be finite")
    if final is None:
        errors.append("training.final_loss must be finite")
    if (
        parsed_losses
        and initial is not None
        and not math.isclose(parsed_losses[0], initial, rel_tol=1e-12, abs_tol=1e-15)
    ):
        errors.append("training.initial_loss disagrees with the first step loss")
    if (
        parsed_losses
        and final is not None
        and not math.isclose(parsed_losses[-1], final, rel_tol=1e-12, abs_tol=1e-15)
    ):
        errors.append("training.final_loss disagrees with the final step loss")
    return errors


def _seed_errors(receipt: Mapping[str, Any], profile: TrainingProfile) -> list[str]:
    errors: list[str] = []
    seed = _mapping(errors, receipt.get("seed"), path="seed", allowed=_SEED_KEYS)
    if seed is None:
        return errors
    for field in ("python", "torch_cpu", "torch_cuda"):
        if seed.get(field) != profile.seed:
            errors.append(f"seed.{field} does not match the profile")
    if seed.get("deterministic_algorithms") is not profile.deterministic_algorithms:
        errors.append("seed.deterministic_algorithms does not match the profile")
    return errors


def _runtime_errors(receipt: Mapping[str, Any], profile: TrainingProfile) -> list[str]:
    errors: list[str] = []
    runtime = _mapping(
        errors, receipt.get("runtime"), path="runtime", allowed=_RUNTIME_KEYS
    )
    if runtime is None:
        return errors
    if runtime.get("device") != profile.device:
        errors.append("runtime.device does not match the profile")
    if runtime.get("dtype") != profile.dtype:
        errors.append("runtime.dtype does not match the profile")
    toolchain = runtime.get("toolchain")
    if not isinstance(toolchain, dict):
        errors.append("runtime.toolchain must be an object")
    else:
        required = {"python", "torch", "transformers"}
        if isinstance(profile, LoraTrainingProfile):
            required.add("peft")
        for package in sorted(required):
            version = toolchain.get(package)
            if not isinstance(version, str) or not version.strip():
                errors.append(f"runtime.toolchain.{package} must be a version string")
        expected_versions = {
            "python": profile.toolchain.python,
            "torch": profile.toolchain.torch,
            "transformers": profile.toolchain.transformers,
        }
        if profile.toolchain.peft is not None:
            expected_versions["peft"] = profile.toolchain.peft
        for package, expected in expected_versions.items():
            observed = toolchain.get(package)
            if isinstance(observed, str) and observed != expected:
                errors.append(f"runtime.toolchain.{package} does not match the profile")
        if isinstance(profile, FineTuneTrainingProfile) and "peft" in toolchain:
            errors.append("runtime.toolchain.peft is only valid for LoRA receipts")
        unknown = sorted(set(toolchain) - {"python", "torch", "transformers", "peft"})
        if unknown:
            errors.append(
                "runtime.toolchain contains unsupported field(s): " + ", ".join(unknown)
            )
    image_digest = runtime.get("container_image_digest")
    if image_digest is not None and not _is_sha256(image_digest):
        errors.append(
            "runtime.container_image_digest must be a canonical sha256 digest"
        )
    return errors


def _hash_and_change_errors(
    receipt: Mapping[str, Any], profile: TrainingProfile
) -> list[str]:
    errors: list[str] = []
    hashes = _mapping(errors, receipt.get("hashes"), path="hashes", allowed=_HASH_KEYS)
    if hashes is not None:
        _require_sha_fields(errors, hashes, path="hashes", fields=_HASH_KEYS)
        baseline_state = hashes.get("baseline_state_sha256")
        baseline_tree = hashes.get("baseline_tree_sha256")
        pre_training = hashes.get("pre_training_state_sha256")
        post_training = hashes.get("post_training_state_sha256")
        subject_tree = hashes.get("subject_tree_sha256")
        reloaded = hashes.get("reloaded_subject_state_sha256")
        if baseline_state != pre_training:
            errors.append("pre-training state must match the baseline state")
        if post_training == pre_training:
            errors.append("post-training state must differ from the baseline state")
        if subject_tree == baseline_tree:
            errors.append("subject checkpoint tree must differ from the baseline tree")
        if reloaded != post_training:
            errors.append("reloaded subject state must match post-training state")

    changes = _mapping(
        errors, receipt.get("changes"), path="changes", allowed=_CHANGE_KEYS
    )
    if changes is not None:
        if not _is_int(changes.get("changed_tensors"), minimum=1):
            errors.append("changes.changed_tensors must be a positive integer")
        if not _is_int(changes.get("changed_params"), minimum=1):
            errors.append("changes.changed_params must be a positive integer")
        if not _is_int(changes.get("total_params"), minimum=1):
            errors.append("changes.total_params must be a positive integer")
        if (
            _is_int(changes.get("changed_params"), minimum=1)
            and _is_int(changes.get("total_params"), minimum=1)
            and int(changes["changed_params"]) > int(changes["total_params"])
        ):
            errors.append("changes.changed_params must not exceed total_params")
        max_delta = _finite_float(changes.get("max_abs_delta"))
        if max_delta is None or max_delta <= 0.0:
            errors.append("changes.max_abs_delta must be finite and positive")

    reload_smoke = _mapping(
        errors,
        receipt.get("reload_smoke"),
        path="reload_smoke",
        allowed=_RELOAD_KEYS,
    )
    if reload_smoke is not None:
        if reload_smoke.get("passed") is not True:
            errors.append("reload_smoke.passed must be true")
        if reload_smoke.get("state_hash_matches") is not True:
            errors.append("reload_smoke.state_hash_matches must be true")
        if reload_smoke.get("inference_performed") is not True:
            errors.append("reload_smoke.inference_performed must be true")
        if reload_smoke.get("all_logits_finite") is not True:
            errors.append("reload_smoke.all_logits_finite must be true")
        if reload_smoke.get("repeat_runs") != 2:
            errors.append("reload_smoke.repeat_runs must equal 2")
        for field in ("input_sha256", "logits_sha256"):
            if not _is_sha256(reload_smoke.get(field)):
                errors.append(f"reload_smoke.{field} must be a canonical sha256 digest")
        logits_shape = reload_smoke.get("logits_shape")
        if (
            not isinstance(logits_shape, list)
            or not logits_shape
            or any(not _is_int(dimension, minimum=1) for dimension in logits_shape)
        ):
            errors.append(
                "reload_smoke.logits_shape must contain positive integer dimensions"
            )
        if reload_smoke.get("device") != profile.device:
            errors.append("reload_smoke.device must match the training profile")
    return errors


def _lora_errors(receipt: Mapping[str, Any], profile: LoraTrainingProfile) -> list[str]:
    errors: list[str] = []
    lora = _mapping(
        errors,
        receipt.get("lora"),
        path="lora",
        allowed=_LORA_RECEIPT_KEYS,
    )
    if lora is None:
        return errors
    hash_fields = {
        "profile_lora_config_sha256",
        "serialized_adapter_config_sha256",
        "initial_adapter_state_sha256",
        "trained_adapter_state_sha256",
        "serialized_adapter_state_sha256",
        "adapter_tree_sha256",
        "base_state_before_adapter_sha256",
        "base_state_after_training_sha256",
        "merged_state_sha256",
        "base_state_manifest_sha256",
        "base_state_manifest_before_adapter_sha256",
        "base_state_manifest_after_training_sha256",
        "expected_merge_target_names_sha256",
        "observed_merged_changed_names_sha256",
    }
    _require_sha_fields(errors, lora, path="lora", fields=hash_fields)
    if lora.get("profile_lora_config_sha256") != lora_config_digest(profile.lora):
        errors.append("lora.profile_lora_config_sha256 does not match the profile")
    if lora.get("initial_adapter_state_sha256") == lora.get(
        "trained_adapter_state_sha256"
    ):
        errors.append("trained adapter state must differ from its initial state")
    if lora.get("serialized_adapter_state_sha256") != lora.get(
        "trained_adapter_state_sha256"
    ):
        errors.append("serialized adapter state must match the trained adapter state")
    if lora.get("base_state_before_adapter_sha256") != lora.get(
        "base_state_after_training_sha256"
    ):
        errors.append("base model state must remain frozen before the merge")
    manifest_hash = lora.get("base_state_manifest_sha256")
    if (
        lora.get("base_state_manifest_before_adapter_sha256") != manifest_hash
        or lora.get("base_state_manifest_after_training_sha256") != manifest_hash
    ):
        errors.append("streaming base-state manifests must remain identical")
    if lora.get("state_evidence_policy") != "streaming-per-tensor-digests-v1":
        errors.append("lora.state_evidence_policy is invalid")
    if lora.get("merge_scope_exact") is not True:
        errors.append("lora.merge_scope_exact must be true")
    merge_targets = lora.get("merge_target_names")
    if (
        not isinstance(merge_targets, list)
        or not merge_targets
        or merge_targets != sorted(set(merge_targets))
        or any(not isinstance(name, str) or not name for name in merge_targets)
    ):
        errors.append("lora.merge_target_names must be a sorted unique string array")
    elif lora.get("expected_merge_target_names_sha256") != canonical_sha256(
        merge_targets
    ):
        errors.append("lora.merge_target_names do not match their digest")
    elif lora.get("observed_merged_changed_names_sha256") != canonical_sha256(
        merge_targets
    ):
        errors.append("lora observed merge targets must exactly match expected targets")
    if not _is_int(lora.get("merged_changed_tensor_count"), minimum=1):
        errors.append("lora.merged_changed_tensor_count must be positive")
    elif isinstance(merge_targets, list) and lora.get(
        "merged_changed_tensor_count"
    ) != len(merge_targets):
        errors.append("lora merged changed tensor count must match merge targets")
    changes = receipt.get("changes")
    if isinstance(changes, dict) and lora.get(
        "merged_changed_tensor_count"
    ) != changes.get("changed_tensors"):
        errors.append("lora merged changed tensor count must match receipt changes")
    hashes = receipt.get("hashes")
    if isinstance(hashes, dict):
        baseline = hashes.get("baseline_state_sha256")
        post_training = hashes.get("post_training_state_sha256")
        if lora.get("base_state_before_adapter_sha256") != baseline:
            errors.append("LoRA base state must match the receipt baseline state")
        if lora.get("merged_state_sha256") != post_training:
            errors.append("LoRA merged state must match post-training subject state")
    if lora.get("adapter_training_performed") is not True:
        errors.append("lora.adapter_training_performed must be true")
    training = receipt.get("training")
    completed_steps = (
        training.get("completed_steps") if isinstance(training, dict) else None
    )
    if lora.get("adapter_optimizer_steps") != completed_steps or not _is_int(
        lora.get("adapter_optimizer_steps"), minimum=1
    ):
        errors.append("lora.adapter_optimizer_steps must equal completed steps")
    if not _is_int(lora.get("trainable_parameter_count"), minimum=1):
        errors.append("lora.trainable_parameter_count must be positive")
    if lora.get("adapter_merge_performed") is not True:
        errors.append("lora.adapter_merge_performed must be true")
    if not _is_int(lora.get("adapter_modules_before_merge"), minimum=1):
        errors.append("lora.adapter_modules_before_merge must be positive")
    if lora.get("adapter_modules_after_merge") != 0:
        errors.append("lora.adapter_modules_after_merge must be zero")
    merge_method = lora.get("merge_method")
    if not isinstance(merge_method, str) or not merge_method.strip():
        errors.append("lora.merge_method must be a non-empty string")
    return errors


def training_receipt_errors(
    receipt: Any,
    *,
    profile: TrainingProfile,
) -> list[str]:
    """Return schema and internal-consistency errors for a training receipt."""

    if not isinstance(receipt, dict):
        return ["training receipt must be an object"]
    errors: list[str] = []
    unknown = sorted(set(receipt) - _TOP_LEVEL_KEYS)
    if unknown:
        errors.append(
            "training receipt contains unsupported field(s): " + ", ".join(unknown)
        )
    if receipt.get("schema") != TRAINING_RECEIPT_SCHEMA:
        errors.append("training receipt has an unknown schema")
    if receipt.get("profile_id") != profile.profile_id:
        errors.append("profile_id does not match the loaded profile")
    if receipt.get("profile_sha256") != profile.profile_sha256:
        errors.append("profile_sha256 does not match the loaded profile")
    if receipt.get("edit_type") != profile.edit_type:
        errors.append("edit_type does not match the loaded profile")
    provider_binding = receipt.get("dataset_provider")
    if not isinstance(provider_binding, dict) or set(provider_binding) != {
        "provider",
        "provider_sha256",
    }:
        errors.append("dataset_provider must contain provider and provider_sha256")
    else:
        provider = provider_binding.get("provider")
        if not isinstance(provider, dict) or not provider:
            errors.append("dataset_provider.provider must be a non-empty object")
        elif provider_binding.get("provider_sha256") != canonical_sha256(provider):
            errors.append("dataset_provider.provider_sha256 does not bind provider")

    errors.extend(_model_errors(receipt, profile))
    errors.extend(_training_data_errors(receipt, profile))
    errors.extend(_optimizer_errors(receipt, profile))
    errors.extend(_training_errors(receipt, profile))
    errors.extend(_seed_errors(receipt, profile))
    errors.extend(_runtime_errors(receipt, profile))
    errors.extend(_hash_and_change_errors(receipt, profile))

    if isinstance(profile, LoraTrainingProfile):
        errors.extend(_lora_errors(receipt, profile))
    elif isinstance(profile, FineTuneTrainingProfile) and "lora" in receipt:
        errors.append("fine_tune receipts must not contain LoRA merge evidence")

    digest = receipt.get("receipt_sha256")
    if not _is_sha256(digest):
        errors.append("receipt_sha256 must be a canonical sha256 digest")
    else:
        try:
            expected_digest = canonical_receipt_digest(receipt)
        except (TypeError, ValueError) as exc:
            errors.append(f"training receipt is not canonical JSON: {exc}")
        else:
            if digest != expected_digest:
                errors.append("receipt_sha256 does not match canonical receipt content")
    return errors


def require_valid_training_receipt(
    receipt: Any,
    *,
    profile: TrainingProfile,
) -> dict[str, Any]:
    """Return a valid receipt or raise with every fail-closed contract error."""

    errors = training_receipt_errors(receipt, profile=profile)
    if errors:
        raise TrainingReceiptError("; ".join(errors))
    return copy.deepcopy(receipt)


__all__ = [
    "TRAINING_RECEIPT_SCHEMA",
    "TrainingReceiptError",
    "canonical_receipt_digest",
    "require_valid_training_receipt",
    "training_receipt_errors",
    "with_receipt_digest",
]
