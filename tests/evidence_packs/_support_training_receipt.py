from __future__ import annotations

import hashlib
from typing import Any

from invarlock.training_model_load import (
    TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA,
    load_diagnostics_sha256,
)
from scripts.evidence_packs.python.editing.training_contract import (
    LoraTrainingProfile,
    TrainingProfile,
    canonical_sha256,
    lora_config_digest,
)
from scripts.evidence_packs.python.editing.training_receipt import (
    TRAINING_RECEIPT_SCHEMA,
    with_receipt_digest,
)


def receipt_sha(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def valid_training_receipt(profile: TrainingProfile) -> dict[str, Any]:
    baseline_state = receipt_sha("baseline-state")
    post_state = receipt_sha(f"{profile.edit_type}-post-state")
    losses = [2.0 - (index * 0.25) for index in range(profile.steps)]
    load_diagnostics = {
        "schema": TRAINING_MODEL_LOAD_DIAGNOSTICS_SCHEMA,
        "policy": "exact_source_key_migration",
        "missing_keys": [],
        "unexpected_keys": list(profile.model_load.expected_unexpected_keys),
        "mismatched_keys": [],
        "error_msgs": [],
    }
    receipt: dict[str, Any] = {
        "schema": TRAINING_RECEIPT_SCHEMA,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "edit_type": profile.edit_type,
        "dataset_provider": {
            "provider": {"kind": "test-fixture"},
            "provider_sha256": (
                "sha256:c86f4e23865c38f089c00c9f03d79884a489d6770c24ca0cbd02a12f09fa58bd"
            ),
        },
        "model": {
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "tokenizer_sha256": receipt_sha("tokenizer"),
            "baseline_load": {
                "loss_function": profile.model_load.loss_function,
                "diagnostics": load_diagnostics,
                "diagnostics_sha256": load_diagnostics_sha256(load_diagnostics),
            },
        },
        "training_data": {
            "path": profile.training_data.path,
            "sha256": profile.training_data.sha256,
            "rows": profile.training_data.rows,
            "text_field": profile.training_data.text_field,
            "token_count": 128,
            "preprocessing_sha256": receipt_sha("preprocessing"),
        },
        "optimizer": {
            "name": profile.optimizer.name,
            "learning_rate": profile.optimizer.learning_rate,
            "betas": list(profile.optimizer.betas),
            "eps": profile.optimizer.eps,
            "weight_decay": profile.optimizer.weight_decay,
        },
        "training": {
            "requested_steps": profile.steps,
            "completed_steps": profile.steps,
            "micro_batch_size": profile.micro_batch_size,
            "gradient_accumulation_steps": profile.gradient_accumulation_steps,
            "max_sequence_length": profile.max_sequence_length,
            "losses": losses,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "optimization_performed": True,
            "training_data_used": True,
        },
        "seed": {
            "python": profile.seed,
            "torch_cpu": profile.seed,
            "torch_cuda": profile.seed,
            "deterministic_algorithms": profile.deterministic_algorithms,
        },
        "runtime": {
            "device": profile.device,
            "dtype": profile.dtype,
            "toolchain": {
                "python": "3.12.13",
                "torch": "2.11.0",
                "transformers": "5.12.0",
            },
        },
        "hashes": {
            "baseline_state_sha256": baseline_state,
            "baseline_tree_sha256": receipt_sha("baseline-tree"),
            "pre_training_state_sha256": baseline_state,
            "post_training_state_sha256": post_state,
            "delta_sha256": receipt_sha(f"{profile.edit_type}-delta"),
            "subject_tree_sha256": receipt_sha(f"{profile.edit_type}-subject-tree"),
            "reloaded_subject_state_sha256": post_state,
        },
        "changes": {
            "changed_tensors": 4,
            "changed_params": 256,
            "total_params": 1024,
            "max_abs_delta": 0.125,
        },
        "reload_smoke": {
            "passed": True,
            "state_hash_matches": True,
            "inference_performed": True,
            "all_logits_finite": True,
            "repeat_runs": 2,
            "input_sha256": receipt_sha("reload-input"),
            "logits_sha256": receipt_sha("reload-logits"),
            "logits_shape": [2, 32, 32],
            "device": profile.device,
        },
    }
    if isinstance(profile, LoraTrainingProfile):
        merge_target_names = [
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.1.attn.c_attn.weight",
        ]
        receipt["runtime"]["toolchain"]["peft"] = profile.toolchain.peft
        receipt["lora"] = {
            "profile_lora_config_sha256": lora_config_digest(profile.lora),
            "serialized_adapter_config_sha256": receipt_sha(
                "serialized-adapter-config"
            ),
            "initial_adapter_state_sha256": receipt_sha("adapter-initial"),
            "trained_adapter_state_sha256": receipt_sha("adapter-trained"),
            "serialized_adapter_state_sha256": receipt_sha("adapter-trained"),
            "adapter_tree_sha256": receipt_sha("adapter-tree"),
            "base_state_before_adapter_sha256": baseline_state,
            "base_state_after_training_sha256": baseline_state,
            "base_state_manifest_sha256": receipt_sha("base-state-manifest"),
            "base_state_manifest_before_adapter_sha256": receipt_sha(
                "base-state-manifest"
            ),
            "base_state_manifest_after_training_sha256": receipt_sha(
                "base-state-manifest"
            ),
            "state_evidence_policy": "streaming-per-tensor-digests-v1",
            "merge_target_names": merge_target_names,
            "expected_merge_target_names_sha256": canonical_sha256(merge_target_names),
            "observed_merged_changed_names_sha256": canonical_sha256(
                merge_target_names
            ),
            "merged_changed_tensor_count": len(merge_target_names),
            "merge_scope_exact": True,
            "merged_state_sha256": post_state,
            "adapter_training_performed": True,
            "adapter_optimizer_steps": profile.steps,
            "trainable_parameter_count": 64,
            "adapter_merge_performed": True,
            "adapter_modules_before_merge": 2,
            "adapter_modules_after_merge": 0,
            "merge_method": "PeftModel.merge_and_unload",
        }
        receipt["changes"]["changed_tensors"] = len(merge_target_names)
    return with_receipt_digest(receipt)
