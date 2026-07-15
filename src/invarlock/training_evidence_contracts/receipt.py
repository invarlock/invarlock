"""Training receipt structural and LoRA validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from invarlock.training_evidence_contracts.common import (
    _BASELINE_LOAD_FIELDS,
    _CHANGE_FIELDS,
    _DATA_FIELDS,
    _HASH_FIELDS,
    _LOAD_DIAGNOSTIC_FIELDS,
    _LORA_RECEIPT_FIELDS,
    _MODEL_FIELDS,
    _NAME_RE,
    _OPTIMIZER_FIELDS,
    _PATH_RE,
    _PROFILE_ID_RE,
    _RECEIPT_COMMON_FIELDS,
    _RECEIPT_LORA_FIELDS,
    _RELOAD_SMOKE_FIELDS,
    _REVISION_RE,
    _RUNTIME_BASE_FIELDS,
    _RUNTIME_IMAGE_FIELDS,
    _SEED_FIELDS,
    _TRAINING_FIELDS,
    TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA,
    TRAINING_RECEIPT_SCHEMA,
    TrainingEvidenceProofError,
    _exact_mapping,
    _finite_float,
    _is_nonnegative_int,
    _is_positive_int,
    _is_sha256,
    _is_text,
    _require_sha,
    canonical_json_sha256,
    canonical_training_receipt_sha256,
    is_training_edit_type,
    load_diagnostics_sha256,
)


def _receipt_header_errors(receipt: Mapping[str, object], errors: list[str]) -> object:
    edit_type = receipt.get("edit_type")
    expected_fields = (
        _RECEIPT_LORA_FIELDS if edit_type == "lora_merge" else _RECEIPT_COMMON_FIELDS
    )
    _exact_mapping(
        receipt,
        label="training receipt",
        fields=expected_fields,
        errors=errors,
    )
    if receipt.get("schema") != TRAINING_RECEIPT_SCHEMA:
        errors.append("training receipt has an unrecognized schema")
    if not is_training_edit_type(edit_type):
        errors.append("training receipt edit_type must be fine_tune or lora_merge")
    if not _is_text(receipt.get("profile_id"), pattern=_PROFILE_ID_RE):
        errors.append("training receipt profile_id is invalid")
    if not _is_sha256(receipt.get("profile_sha256")):
        errors.append("training receipt profile_sha256 must be a sha256 digest")
    return edit_type


def _model_errors(receipt: Mapping[str, object], errors: list[str]) -> None:
    model = _exact_mapping(
        receipt.get("model"),
        label="training receipt.model",
        fields=_MODEL_FIELDS,
        errors=errors,
    )
    if model is not None:
        if not _is_text(model.get("model_id"), pattern=_PATH_RE):
            errors.append("training receipt.model.model_id is invalid")
        revision = model.get("model_revision")
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            errors.append("training receipt.model.model_revision must be pinned")
        _require_sha(
            model, "tokenizer_sha256", label="training receipt.model", errors=errors
        )
        baseline_load = _exact_mapping(
            model.get("baseline_load"),
            label="training receipt.model.baseline_load",
            fields=_BASELINE_LOAD_FIELDS,
            errors=errors,
        )
        if baseline_load is not None:
            if baseline_load.get("loss_function") != "ForCausalLM":
                errors.append(
                    "training receipt.model.baseline_load.loss_function is invalid"
                )
            diagnostics = _exact_mapping(
                baseline_load.get("diagnostics"),
                label="training receipt.model.baseline_load.diagnostics",
                fields=_LOAD_DIAGNOSTIC_FIELDS,
                errors=errors,
            )
            if diagnostics is not None:
                if diagnostics.get("schema") != TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA:
                    errors.append(
                        "training receipt.model.baseline_load.diagnostics schema is invalid"
                    )
                if diagnostics.get("policy") != "exact_source_key_migration":
                    errors.append(
                        "training receipt.model.baseline_load.diagnostics policy is invalid"
                    )
                for field in ("missing_keys", "mismatched_keys", "error_msgs"):
                    if diagnostics.get(field) != []:
                        errors.append(
                            "training receipt.model.baseline_load.diagnostics."
                            f"{field} must be empty"
                        )
                unexpected = diagnostics.get("unexpected_keys")
                if (
                    not isinstance(unexpected, list)
                    or unexpected != sorted(set(unexpected))
                    or any(not isinstance(item, str) or not item for item in unexpected)
                ):
                    errors.append(
                        "training receipt.model.baseline_load.diagnostics."
                        "unexpected_keys must be sorted and unique"
                    )
                diagnostics_digest = baseline_load.get("diagnostics_sha256")
                if not _is_sha256(diagnostics_digest):
                    errors.append(
                        "training receipt.model.baseline_load.diagnostics_sha256 "
                        "must be a sha256 digest"
                    )
                elif diagnostics_digest != load_diagnostics_sha256(diagnostics):
                    errors.append(
                        "training receipt.model.baseline_load diagnostics do not "
                        "match their digest"
                    )


def _training_data_errors(receipt: Mapping[str, object], errors: list[str]) -> None:
    training_data = _exact_mapping(
        receipt.get("training_data"),
        label="training receipt.training_data",
        fields=_DATA_FIELDS,
        errors=errors,
    )
    if training_data is not None:
        path = training_data.get("path")
        if (
            not isinstance(path, str)
            or not path
            or path != path.strip()
            or _PATH_RE.fullmatch(path) is None
            or path.startswith("/")
            or ".." in path.split("/")
        ):
            errors.append("training receipt.training_data.path is invalid")
        _require_sha(
            training_data,
            "sha256",
            label="training receipt.training_data",
            errors=errors,
        )
        if not _is_positive_int(training_data.get("rows")):
            errors.append("training receipt.training_data.rows must be positive")
        if not _is_text(training_data.get("text_field"), pattern=_NAME_RE):
            errors.append("training receipt.training_data.text_field is invalid")
        if not _is_positive_int(training_data.get("token_count")):
            errors.append("training receipt.training_data.token_count must be positive")
        _require_sha(
            training_data,
            "preprocessing_sha256",
            label="training receipt.training_data",
            errors=errors,
        )


def _optimizer_errors(receipt: Mapping[str, object], errors: list[str]) -> None:
    optimizer = _exact_mapping(
        receipt.get("optimizer"),
        label="training receipt.optimizer",
        fields=_OPTIMIZER_FIELDS,
        errors=errors,
    )
    if optimizer is not None:
        if not _is_text(optimizer.get("name"), pattern=_NAME_RE):
            errors.append("training receipt.optimizer.name is invalid")
        learning_rate = _finite_float(optimizer.get("learning_rate"))
        if learning_rate is None or learning_rate <= 0.0:
            errors.append("training receipt.optimizer.learning_rate must be positive")
        betas = optimizer.get("betas")
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or any(
                beta is None or beta < 0.0 or beta >= 1.0
                for beta in (_finite_float(value) for value in betas)
            )
        ):
            errors.append(
                "training receipt.optimizer.betas must contain two values in [0, 1)"
            )
        eps = _finite_float(optimizer.get("eps"))
        if eps is None or eps <= 0.0:
            errors.append("training receipt.optimizer.eps must be positive")
        weight_decay = _finite_float(optimizer.get("weight_decay"))
        if weight_decay is None or weight_decay < 0.0:
            errors.append("training receipt.optimizer.weight_decay must be nonnegative")


def _training_errors(
    receipt: Mapping[str, object], errors: list[str]
) -> Mapping[str, object] | None:
    training = _exact_mapping(
        receipt.get("training"),
        label="training receipt.training",
        fields=_TRAINING_FIELDS,
        errors=errors,
    )
    if training is not None:
        requested = training.get("requested_steps")
        completed = training.get("completed_steps")
        for field in (
            "requested_steps",
            "completed_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_sequence_length",
        ):
            if not _is_positive_int(training.get(field)):
                errors.append(f"training receipt.training.{field} must be positive")
        if (
            _is_positive_int(requested)
            and _is_positive_int(completed)
            and requested != completed
        ):
            errors.append("training receipt.training must record a completed schedule")
        _training_loss_errors(training, completed=completed, errors=errors)
        if training.get("optimization_performed") is not True:
            errors.append(
                "training receipt.training.optimization_performed must be true"
            )
        if training.get("training_data_used") is not True:
            errors.append("training receipt.training.training_data_used must be true")
    return training


def _training_loss_errors(
    training: Mapping[str, object], *, completed: object, errors: list[str]
) -> None:
    losses = training.get("losses")
    parsed_losses: list[float] = []
    if not isinstance(losses, list) or not losses:
        errors.append("training receipt.training.losses must be non-empty")
    else:
        for index, loss in enumerate(losses):
            parsed = _finite_float(loss)
            if parsed is None:
                errors.append(
                    f"training receipt.training.losses[{index}] must be finite"
                )
            else:
                parsed_losses.append(parsed)
        if _is_positive_int(completed) and len(losses) != completed:
            errors.append("training receipt.training.losses must match completed_steps")
    initial = _finite_float(training.get("initial_loss"))
    final = _finite_float(training.get("final_loss"))
    if initial is None:
        errors.append("training receipt.training.initial_loss must be finite")
    if final is None:
        errors.append("training receipt.training.final_loss must be finite")
    if parsed_losses and initial is not None and parsed_losses[0] != initial:
        errors.append("training receipt.training.initial_loss does not bind losses")
    if parsed_losses and final is not None and parsed_losses[-1] != final:
        errors.append("training receipt.training.final_loss does not bind losses")


def _seed_errors(receipt: Mapping[str, object], errors: list[str]) -> None:
    seed = _exact_mapping(
        receipt.get("seed"),
        label="training receipt.seed",
        fields=_SEED_FIELDS,
        errors=errors,
    )
    if seed is not None:
        seed_values = [
            seed.get(field) for field in ("python", "torch_cpu", "torch_cuda")
        ]
        if not all(_is_nonnegative_int(value) for value in seed_values):
            errors.append("training receipt.seed values must be nonnegative integers")
        elif not (seed_values[0] == seed_values[1] == seed_values[2]):
            errors.append("training receipt.seed values must agree")
        if not isinstance(seed.get("deterministic_algorithms"), bool):
            errors.append(
                "training receipt.seed.deterministic_algorithms must be boolean"
            )


def _runtime_errors(
    receipt: Mapping[str, object], *, edit_type: object, errors: list[str]
) -> Mapping[str, object] | None:
    runtime = _exact_mapping(
        receipt.get("runtime"),
        label="training receipt.runtime",
        fields=(
            _RUNTIME_IMAGE_FIELDS
            if isinstance(receipt.get("runtime"), Mapping)
            and "container_image_digest"
            in cast(Mapping[str, object], receipt["runtime"])
            else _RUNTIME_BASE_FIELDS
        ),
        errors=errors,
    )
    if runtime is not None:
        if runtime.get("device") not in {"cpu", "cuda", "mps"}:
            errors.append("training receipt.runtime.device is invalid")
        if runtime.get("dtype") not in {"bfloat16", "float16", "float32"}:
            errors.append("training receipt.runtime.dtype is invalid")
        toolchain = runtime.get("toolchain")
        required_toolchain_fields = (
            frozenset({"python", "torch", "transformers", "peft"})
            if edit_type == "lora_merge"
            else frozenset({"python", "torch", "transformers"})
        )
        toolchain_mapping = _exact_mapping(
            toolchain,
            label="training receipt.runtime.toolchain",
            fields=required_toolchain_fields,
            errors=errors,
        )
        if toolchain_mapping is not None:
            for field in sorted(required_toolchain_fields):
                if not _is_text(toolchain_mapping.get(field), pattern=_NAME_RE):
                    errors.append(
                        f"training receipt.runtime.toolchain.{field} is invalid"
                    )
        if "container_image_digest" in runtime:
            _require_sha(
                runtime,
                "container_image_digest",
                label="training receipt.runtime",
                errors=errors,
            )
    return runtime


def _hash_errors(
    receipt: Mapping[str, object], errors: list[str]
) -> Mapping[str, object] | None:
    hashes = _exact_mapping(
        receipt.get("hashes"),
        label="training receipt.hashes",
        fields=_HASH_FIELDS,
        errors=errors,
    )
    if hashes is not None:
        values = {
            field: _require_sha(
                hashes, field, label="training receipt.hashes", errors=errors
            )
            for field in _HASH_FIELDS
        }
        if (
            values["baseline_state_sha256"] is not None
            and values["pre_training_state_sha256"] is not None
            and values["baseline_state_sha256"] != values["pre_training_state_sha256"]
        ):
            errors.append("training receipt pre-training state must match baseline")
        if (
            values["post_training_state_sha256"] is not None
            and values["pre_training_state_sha256"] is not None
            and values["post_training_state_sha256"]
            == values["pre_training_state_sha256"]
        ):
            errors.append(
                "training receipt post-training state must differ from baseline"
            )
        if (
            values["reloaded_subject_state_sha256"] is not None
            and values["post_training_state_sha256"] is not None
            and values["reloaded_subject_state_sha256"]
            != values["post_training_state_sha256"]
        ):
            errors.append(
                "training receipt reloaded state must match post-training state"
            )
        if (
            values["subject_tree_sha256"] is not None
            and values["baseline_tree_sha256"] is not None
            and values["subject_tree_sha256"] == values["baseline_tree_sha256"]
        ):
            errors.append(
                "training receipt subject tree must differ from baseline tree"
            )
    return hashes


def _change_errors(receipt: Mapping[str, object], errors: list[str]) -> None:
    changes = _exact_mapping(
        receipt.get("changes"),
        label="training receipt.changes",
        fields=_CHANGE_FIELDS,
        errors=errors,
    )
    if changes is not None:
        changed_tensors = changes.get("changed_tensors")
        changed_params = changes.get("changed_params")
        total_params = changes.get("total_params")
        if not _is_positive_int(changed_tensors):
            errors.append("training receipt.changes.changed_tensors must be positive")
        if not _is_positive_int(changed_params):
            errors.append("training receipt.changes.changed_params must be positive")
        if not _is_positive_int(total_params):
            errors.append("training receipt.changes.total_params must be positive")
        if (
            _is_positive_int(changed_params)
            and _is_positive_int(total_params)
            and cast(int, changed_params) > cast(int, total_params)
        ):
            errors.append(
                "training receipt.changes.changed_params must not exceed total_params"
            )
        max_delta = _finite_float(changes.get("max_abs_delta"))
        if max_delta is None or max_delta <= 0.0:
            errors.append("training receipt.changes.max_abs_delta must be positive")


def _reload_smoke_errors(
    receipt: Mapping[str, object],
    *,
    runtime: Mapping[str, object] | None,
    errors: list[str],
) -> None:
    reload_smoke = _exact_mapping(
        receipt.get("reload_smoke"),
        label="training receipt.reload_smoke",
        fields=_RELOAD_SMOKE_FIELDS,
        errors=errors,
    )
    if reload_smoke is not None:
        for field in (
            "passed",
            "state_hash_matches",
            "inference_performed",
            "all_logits_finite",
        ):
            if reload_smoke.get(field) is not True:
                errors.append(f"training receipt.reload_smoke.{field} must be true")
        if reload_smoke.get("repeat_runs") != 2:
            errors.append("training receipt.reload_smoke.repeat_runs must equal two")
        for field in ("input_sha256", "logits_sha256"):
            _require_sha(
                reload_smoke,
                field,
                label="training receipt.reload_smoke",
                errors=errors,
            )
        shape = reload_smoke.get("logits_shape")
        if (
            not isinstance(shape, list)
            or not shape
            or not all(_is_positive_int(dimension) for dimension in shape)
        ):
            errors.append("training receipt.reload_smoke.logits_shape is invalid")
        if runtime is not None and reload_smoke.get("device") != runtime.get("device"):
            errors.append(
                "training receipt.reload_smoke.device must match runtime.device"
            )


def _receipt_digest_errors(receipt: Mapping[str, object], errors: list[str]) -> None:
    receipt_digest = receipt.get("receipt_sha256")
    if not _is_sha256(receipt_digest):
        errors.append("training receipt.receipt_sha256 must be a sha256 digest")
        return
    try:
        expected_digest = canonical_training_receipt_sha256(receipt)
    except TrainingEvidenceProofError as exc:
        errors.append(str(exc))
    else:
        if receipt_digest != expected_digest:
            errors.append("training receipt.receipt_sha256 does not bind content")


def _receipt_errors(receipt: object) -> tuple[list[str], Mapping[str, object] | None]:
    """Validate the generic, profile-independent portion of receipt v1.

    The generation runtime separately validates a receipt against an immutable
    profile.  The package checker cannot import that optional runtime, so this
    routine verifies the closed surface and every cross-binding that can be
    checked from the staged receipt itself.  Pack integration can additionally
    bind a known profile identifier and profile digest through the proof.
    """

    errors: list[str] = []
    if not isinstance(receipt, Mapping) or not all(
        isinstance(key, str) for key in receipt
    ):
        return ["training receipt must be an object"], None

    edit_type = _receipt_header_errors(receipt, errors)
    provider_binding = _exact_mapping(
        receipt.get("dataset_provider"),
        label="training receipt.dataset_provider",
        fields=frozenset({"provider", "provider_sha256"}),
        errors=errors,
    )
    if provider_binding is not None:
        provider = provider_binding.get("provider")
        if not isinstance(provider, Mapping) or not provider:
            errors.append(
                "training receipt.dataset_provider.provider must be a non-empty object"
            )
        elif provider_binding.get("provider_sha256") != canonical_json_sha256(provider):
            errors.append(
                "training receipt.dataset_provider.provider_sha256 must bind provider"
            )
    _model_errors(receipt, errors)
    _training_data_errors(receipt, errors)
    _optimizer_errors(receipt, errors)
    training = _training_errors(receipt, errors)
    _seed_errors(receipt, errors)
    runtime = _runtime_errors(receipt, edit_type=edit_type, errors=errors)
    hashes = _hash_errors(receipt, errors)
    _change_errors(receipt, errors)
    _reload_smoke_errors(receipt, runtime=runtime, errors=errors)
    if edit_type == "lora_merge":
        _lora_receipt_errors(receipt, hashes=hashes, training=training, errors=errors)
    elif "lora" in receipt:
        errors.append("fine_tune training receipt must not carry LoRA evidence")
    _receipt_digest_errors(receipt, errors)
    return errors, receipt


def _lora_receipt_errors(
    receipt: Mapping[str, object],
    *,
    hashes: Mapping[str, object] | None,
    training: Mapping[str, object] | None,
    errors: list[str],
) -> None:
    lora = _exact_mapping(
        receipt.get("lora"),
        label="training receipt.lora",
        fields=_LORA_RECEIPT_FIELDS,
        errors=errors,
    )
    if lora is None:
        return
    values = _lora_digest_errors(lora, errors)
    merge_targets = _lora_merge_target_errors(lora, values=values, errors=errors)
    _lora_binding_errors(
        receipt,
        lora,
        values=values,
        merge_targets=merge_targets,
        hashes=hashes,
        errors=errors,
    )
    _lora_execution_errors(lora, training=training, errors=errors)


def _lora_digest_errors(
    lora: Mapping[str, object], errors: list[str]
) -> dict[str, str | None]:
    values = {
        field: _require_sha(lora, field, label="training receipt.lora", errors=errors)
        for field in (
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
            "expected_merge_target_names_sha256",
            "observed_merged_changed_names_sha256",
            "merged_state_sha256",
        )
    }
    if (
        values["initial_adapter_state_sha256"] is not None
        and values["trained_adapter_state_sha256"] is not None
        and values["initial_adapter_state_sha256"]
        == values["trained_adapter_state_sha256"]
    ):
        errors.append(
            "training receipt.lora trained adapter must differ from initial adapter"
        )
    if (
        values["serialized_adapter_state_sha256"] is not None
        and values["trained_adapter_state_sha256"] is not None
        and values["serialized_adapter_state_sha256"]
        != values["trained_adapter_state_sha256"]
    ):
        errors.append(
            "training receipt.lora serialized adapter must match trained adapter"
        )
    if (
        values["base_state_before_adapter_sha256"] is not None
        and values["base_state_after_training_sha256"] is not None
        and values["base_state_before_adapter_sha256"]
        != values["base_state_after_training_sha256"]
    ):
        errors.append("training receipt.lora base state must remain frozen")
    manifest = values["base_state_manifest_sha256"]
    if manifest is not None and (
        values["base_state_manifest_before_adapter_sha256"] != manifest
        or values["base_state_manifest_after_training_sha256"] != manifest
    ):
        errors.append("training receipt.lora streaming manifests must remain frozen")
    if lora.get("state_evidence_policy") != "streaming-per-tensor-digests-v1":
        errors.append("training receipt.lora.state_evidence_policy is invalid")
    return values


def _lora_merge_target_errors(
    lora: Mapping[str, object],
    *,
    values: Mapping[str, str | None],
    errors: list[str],
) -> object:
    merge_targets = lora.get("merge_target_names")
    if (
        not isinstance(merge_targets, list)
        or not merge_targets
        or merge_targets != sorted(set(merge_targets))
        or any(not isinstance(name, str) or not name for name in merge_targets)
    ):
        errors.append(
            "training receipt.lora.merge_target_names must be sorted and unique"
        )
    elif values["expected_merge_target_names_sha256"] != canonical_json_sha256(
        merge_targets
    ):
        errors.append("training receipt.lora.merge_target_names digest mismatch")
    elif values["observed_merged_changed_names_sha256"] != canonical_json_sha256(
        merge_targets
    ):
        errors.append(
            "training receipt.lora observed merge targets must match expected targets"
        )
    return merge_targets


def _lora_binding_errors(
    receipt: Mapping[str, object],
    lora: Mapping[str, object],
    *,
    values: Mapping[str, str | None],
    merge_targets: object,
    hashes: Mapping[str, object] | None,
    errors: list[str],
) -> None:
    if lora.get("merge_scope_exact") is not True:
        errors.append("training receipt.lora.merge_scope_exact must be true")
    if not _is_positive_int(lora.get("merged_changed_tensor_count")):
        errors.append(
            "training receipt.lora.merged_changed_tensor_count must be positive"
        )
    elif isinstance(merge_targets, list) and lora.get(
        "merged_changed_tensor_count"
    ) != len(merge_targets):
        errors.append(
            "training receipt.lora changed tensor count must match merge targets"
        )
    changes = receipt.get("changes")
    if isinstance(changes, Mapping) and lora.get(
        "merged_changed_tensor_count"
    ) != changes.get("changed_tensors"):
        errors.append(
            "training receipt.lora changed tensor count must match receipt changes"
        )
    if hashes is not None:
        baseline = hashes.get("baseline_state_sha256")
        post_training = hashes.get("post_training_state_sha256")
        if (
            values["base_state_before_adapter_sha256"] is not None
            and values["base_state_before_adapter_sha256"] != baseline
        ):
            errors.append("training receipt.lora base state must bind baseline")
        if (
            values["merged_state_sha256"] is not None
            and values["merged_state_sha256"] != post_training
        ):
            errors.append("training receipt.lora merged state must bind subject")


def _lora_execution_errors(
    lora: Mapping[str, object],
    *,
    training: Mapping[str, object] | None,
    errors: list[str],
) -> None:
    if lora.get("adapter_training_performed") is not True:
        errors.append("training receipt.lora.adapter_training_performed must be true")
    if lora.get("adapter_merge_performed") is not True:
        errors.append("training receipt.lora.adapter_merge_performed must be true")
    if not _is_positive_int(lora.get("adapter_optimizer_steps")):
        errors.append("training receipt.lora.adapter_optimizer_steps must be positive")
    elif training is not None and lora.get("adapter_optimizer_steps") != training.get(
        "completed_steps"
    ):
        errors.append(
            "training receipt.lora.adapter_optimizer_steps must bind completed_steps"
        )
    if not _is_positive_int(lora.get("trainable_parameter_count")):
        errors.append(
            "training receipt.lora.trainable_parameter_count must be positive"
        )
    if not _is_positive_int(lora.get("adapter_modules_before_merge")):
        errors.append(
            "training receipt.lora.adapter_modules_before_merge must be positive"
        )
    if lora.get("adapter_modules_after_merge") != 0:
        errors.append("training receipt.lora.adapter_modules_after_merge must be zero")
    if lora.get("merge_method") != "PeftModel.merge_and_unload":
        errors.append(
            "training receipt.lora.merge_method must be PeftModel.merge_and_unload"
        )
