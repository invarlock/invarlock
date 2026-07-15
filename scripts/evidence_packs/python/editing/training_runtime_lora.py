"""LoRA-specific execution for the evidence-pack training runtime."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from invarlock.peft_runtime import (
    PeftRuntimeError,
    get_dense_peft_model,
    load_dense_peft_model,
)

from .training_contract import file_sha256, lora_config_digest


def execute_lora_training(
    owner: Any,
    *,
    profile: Any,
    model: Any,
    tokenizer: Any,
    deps: Any,
    baseline_hash: str,
    baseline_manifest: dict[str, object],
    baseline_manifest_hash: str,
    baseline_state: dict[str, Any],
    baseline_load_diagnostics: dict[str, object],
    batches: list[dict[str, Any]],
    staging: Path,
    load_options: dict[str, object],
    local_files_only: bool,
    device: Any,
    dtype: Any,
) -> tuple[Any, list[float], Any, dict[str, Any], dict[str, Any]]:
    """Train, serialize, reload, and merge one genuine LoRA adapter."""
    peft_deps = owner._load_peft_dependencies()
    owner._require_expected_toolchain(profile, deps, peft_deps)
    config = peft_deps.lora_config_cls(
        r=profile.lora.rank,
        lora_alpha=profile.lora.alpha,
        lora_dropout=profile.lora.dropout,
        target_modules=list(profile.lora.target_modules),
        bias=profile.lora.bias,
        task_type=profile.lora.task_type,
        fan_in_fan_out=profile.lora.fan_in_fan_out,
    )
    try:
        peft_model = get_dense_peft_model(
            model,
            config,
            get_peft_model=peft_deps.get_peft_model,
        )
    except PeftRuntimeError as exc:
        raise owner.TrainingRuntimeError(str(exc)) from exc
    base_before_state = owner._peft_base_state(peft_model)
    base_before_hash = owner.tensor_state_sha256(base_before_state, torch=deps.torch)
    if base_before_hash != baseline_hash:
        raise owner.TrainingRuntimeError(
            "PEFT adapter construction mutated the pristine base-model tensors"
        )
    base_manifest_before_hash = owner._require_state_manifest(
        owner._peft_base_state(peft_model),
        baseline_manifest,
        torch=deps.torch,
        label="PEFT adapter construction base state",
    )
    target_names = owner._peft_merge_target_names(
        peft_model,
        baseline_state,
    )
    baseline_targets = {
        name: baseline_state[name].detach().cpu().clone() for name in target_names
    }
    del base_before_state
    del baseline_state
    initial_adapter = {
        name: tensor.detach().cpu().clone()
        for name, tensor in peft_deps.get_peft_model_state_dict(
            peft_model, save_embedding_layers=False
        ).items()
    }
    initial_adapter_hash = owner.tensor_state_sha256(initial_adapter, torch=deps.torch)
    trainable = [
        (name, parameter)
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise owner.TrainingRuntimeError(
            "PEFT optimizer must contain only LoRA adapter parameters"
        )
    losses = owner._train(
        peft_model,
        [parameter for _, parameter in trainable],
        batches,
        profile,
        deps=deps,
        device=device,
    )
    base_after_state = owner._peft_base_state(peft_model)
    base_after_hash = owner.tensor_state_sha256(base_after_state, torch=deps.torch)
    if base_after_hash != baseline_hash:
        raise owner.TrainingRuntimeError(
            "LoRA training mutated frozen base-model tensors"
        )
    base_manifest_after_hash = owner._require_state_manifest(
        owner._peft_base_state(peft_model),
        baseline_manifest,
        torch=deps.torch,
        label="LoRA training base state",
    )
    del base_after_state
    trained_adapter = {
        name: tensor.detach().cpu().clone()
        for name, tensor in peft_deps.get_peft_model_state_dict(
            peft_model, save_embedding_layers=False
        ).items()
    }
    trained_adapter_hash = owner.tensor_state_sha256(trained_adapter, torch=deps.torch)
    if trained_adapter_hash == initial_adapter_hash:
        raise owner.TrainingRuntimeError(
            "LoRA optimizer steps did not change adapter tensors"
        )
    adapter_modules_before = owner._adapter_module_count(peft_model.state_dict())
    if adapter_modules_before < 1:
        raise owner.TrainingRuntimeError("PEFT model contains no LoRA adapter modules")
    adapter_dir = staging / "adapter"
    with owner._hf_offline_if(local_files_only):
        peft_model.save_pretrained(
            adapter_dir,
            safe_serialization=True,
            save_embedding_layers=False,
        )
    adapter_tree_hash = owner.directory_sha256(adapter_dir)
    serialized_adapter_config_hash = file_sha256(adapter_dir / "adapter_config.json")
    del peft_model
    del model
    gc.collect()
    if deps.torch.cuda.is_available():
        deps.torch.cuda.empty_cache()
    serialized_base, repeated_baseline_diagnostics = owner._load_profile_baseline(
        deps, profile, load_options=load_options
    )
    if repeated_baseline_diagnostics != baseline_load_diagnostics:
        raise owner.TrainingRuntimeError(
            "upstream baseline loading diagnostics changed during training"
        )
    if hasattr(serialized_base, "config"):
        serialized_base.config.pad_token_id = tokenizer.pad_token_id
    serialized_base.to(device=device, dtype=dtype)
    with owner._hf_offline_if(local_files_only):
        serialized_config = peft_deps.lora_config_cls.from_pretrained(
            adapter_dir,
            local_files_only=local_files_only,
        )
        try:
            serialized_adapter = load_dense_peft_model(
                serialized_base,
                serialized_config,
                adapter_dir,
                from_pretrained=peft_deps.peft_model_cls.from_pretrained,
                is_trainable=False,
                local_files_only=local_files_only,
            )
        except PeftRuntimeError as exc:
            raise owner.TrainingRuntimeError(str(exc)) from exc
    serialized_adapter_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in peft_deps.get_peft_model_state_dict(
            serialized_adapter, save_embedding_layers=False
        ).items()
    }
    serialized_adapter_hash = owner.tensor_state_sha256(
        serialized_adapter_state, torch=deps.torch
    )
    if serialized_adapter_hash != trained_adapter_hash:
        raise owner.TrainingRuntimeError(
            "serialized LoRA adapter state does not match trained adapter state"
        )
    subject_model = serialized_adapter.merge_and_unload()
    adapter_modules_after = owner._adapter_module_count(subject_model.state_dict())
    if adapter_modules_after != 0:
        raise owner.TrainingRuntimeError(
            "merge_and_unload left LoRA adapter modules behind"
        )
    trainable_parameter_count = sum(p.numel() for _, p in trainable)
    del serialized_adapter
    del serialized_base
    del trainable
    del initial_adapter
    del trained_adapter
    del serialized_adapter_state
    lora_receipt = {
        "profile_lora_config_sha256": lora_config_digest(profile.lora),
        "serialized_adapter_config_sha256": serialized_adapter_config_hash,
        "initial_adapter_state_sha256": initial_adapter_hash,
        "trained_adapter_state_sha256": trained_adapter_hash,
        "serialized_adapter_state_sha256": serialized_adapter_hash,
        "adapter_tree_sha256": adapter_tree_hash,
        "base_state_before_adapter_sha256": base_before_hash,
        "base_state_after_training_sha256": base_after_hash,
        "base_state_manifest_sha256": baseline_manifest_hash,
        "base_state_manifest_before_adapter_sha256": base_manifest_before_hash,
        "base_state_manifest_after_training_sha256": base_manifest_after_hash,
        "adapter_training_performed": True,
        "adapter_optimizer_steps": len(losses),
        "trainable_parameter_count": trainable_parameter_count,
        "adapter_merge_performed": True,
        "adapter_modules_before_merge": adapter_modules_before,
        "adapter_modules_after_merge": adapter_modules_after,
        "merge_method": "PeftModel.merge_and_unload",
    }
    return subject_model, losses, peft_deps, lora_receipt, baseline_targets
