from __future__ import annotations

from invarlock.training_evidence import (
    LORA_MERGE_PROOF_SCHEMA,
    TRAINING_ARTIFACT_REPLAY_PROVENANCE_KIND,
    TRAINING_ARTIFACT_REPLAY_SCHEMA,
    TRAINING_EVIDENCE_PROOF_SCHEMA,
    TRAINING_RECEIPT_SCHEMA,
    TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA,
    with_training_evidence_proof_digest,
)
from tests.evidence_packs._support_training_receipt import receipt_sha


def _identity(label: str) -> dict[str, str]:
    return {"kind": "local_checkpoint_tree", "sha256": receipt_sha(label)}


def _proof_for(
    receipt: dict[str, object],
) -> tuple[dict[str, object], dict[str, str], dict[str, str]]:
    hashes = receipt["hashes"]
    changes = receipt["changes"]
    reload_smoke = receipt["reload_smoke"]
    runtime = receipt["runtime"]
    model = receipt["model"]
    training = receipt["training"]
    assert isinstance(hashes, dict)
    assert isinstance(changes, dict)
    assert isinstance(reload_smoke, dict)
    assert isinstance(runtime, dict)
    assert isinstance(model, dict)
    assert isinstance(training, dict)
    baseline_load = model["baseline_load"]
    assert isinstance(baseline_load, dict)

    baseline = {"kind": "remote_revision", "revision": model["model_revision"]}
    artifact = _identity("trained-artifact")
    edit_type = receipt["edit_type"]
    proof: dict[str, object] = {
        "schema": TRAINING_EVIDENCE_PROOF_SCHEMA,
        "edit_type": edit_type,
        "provenance": {
            "kind": TRAINING_ARTIFACT_REPLAY_PROVENANCE_KIND,
            "producer_declared_training_backend": (
                "peft_lora_train_and_merge"
                if edit_type == "lora_merge"
                else "full_parameter_optimizer_training"
            ),
        },
        "training_receipt": {
            "schema": TRAINING_RECEIPT_SCHEMA,
            "receipt_sha256": receipt["receipt_sha256"],
            "profile_id": receipt["profile_id"],
            "profile_sha256": receipt["profile_sha256"],
            "edit_type": edit_type,
            "dataset_provider": receipt["dataset_provider"],
        },
        "baseline_identity": baseline,
        "artifact_identity": artifact,
        "artifact_replay": {
            "schema": TRAINING_ARTIFACT_REPLAY_SCHEMA,
            "passed": True,
            "receipt_sha256": receipt["receipt_sha256"],
            "baseline_identity": baseline,
            "artifact_identity": artifact,
            "baseline_tree_sha256": hashes["baseline_tree_sha256"],
            "subject_tree_sha256": hashes["subject_tree_sha256"],
            "baseline_state_sha256": hashes["baseline_state_sha256"],
            "post_training_state_sha256": hashes["post_training_state_sha256"],
            "reloaded_subject_state_sha256": hashes["reloaded_subject_state_sha256"],
            "delta_sha256": hashes["delta_sha256"],
            "changed_tensors": changes["changed_tensors"],
            "changed_params": changes["changed_params"],
            "total_params": changes["total_params"],
            "max_abs_delta": changes["max_abs_delta"],
            "baseline_load_diagnostics_sha256": baseline_load["diagnostics_sha256"],
            "loss_function": baseline_load["loss_function"],
            "saved_artifact_verified": True,
            "reloaded_artifact_verified": True,
        },
        "runtime_reload": {
            "schema": TRAINING_RUNTIME_RELOAD_PROOF_SCHEMA,
            "passed": True,
            "receipt_sha256": receipt["receipt_sha256"],
            "artifact_identity": artifact,
            "subject_state_sha256": hashes["post_training_state_sha256"],
            "reload_runs": 2,
            "input_sha256": reload_smoke["input_sha256"],
            "logits_sha256": reload_smoke["logits_sha256"],
            "logits_shape": reload_smoke["logits_shape"],
            "all_logits_finite": True,
            "repeat_deterministic": True,
            "device": runtime["device"],
        },
    }
    if edit_type == "lora_merge":
        lora = receipt["lora"]
        assert isinstance(lora, dict)
        proof["lora_merge"] = {
            "schema": LORA_MERGE_PROOF_SCHEMA,
            "adapter_identity": _identity("serialized-lora-adapter"),
            "adapter_tree_sha256": lora["adapter_tree_sha256"],
            "profile_lora_config_sha256": lora["profile_lora_config_sha256"],
            "serialized_adapter_config_sha256": lora[
                "serialized_adapter_config_sha256"
            ],
            "initial_adapter_state_sha256": lora["initial_adapter_state_sha256"],
            "trained_adapter_state_sha256": lora["trained_adapter_state_sha256"],
            "serialized_adapter_state_sha256": lora["serialized_adapter_state_sha256"],
            "base_state_before_adapter_sha256": lora[
                "base_state_before_adapter_sha256"
            ],
            "base_state_after_training_sha256": lora[
                "base_state_after_training_sha256"
            ],
            "base_state_manifest_sha256": lora["base_state_manifest_sha256"],
            "base_state_manifest_before_adapter_sha256": lora[
                "base_state_manifest_before_adapter_sha256"
            ],
            "base_state_manifest_after_training_sha256": lora[
                "base_state_manifest_after_training_sha256"
            ],
            "state_evidence_policy": lora["state_evidence_policy"],
            "merge_target_names": lora["merge_target_names"],
            "expected_merge_target_names_sha256": lora[
                "expected_merge_target_names_sha256"
            ],
            "observed_merged_changed_names_sha256": lora[
                "observed_merged_changed_names_sha256"
            ],
            "merged_changed_tensor_count": lora["merged_changed_tensor_count"],
            "merge_scope_exact": True,
            "merged_state_sha256": lora["merged_state_sha256"],
            "adapter_optimizer_steps": training["completed_steps"],
            "trainable_parameter_count": lora["trainable_parameter_count"],
            "adapter_modules_before_merge": lora["adapter_modules_before_merge"],
            "adapter_modules_after_merge": lora["adapter_modules_after_merge"],
            "merge_method": lora["merge_method"],
            "adapter_training_performed": True,
            "adapter_merge_performed": True,
        }
    return with_training_evidence_proof_digest(proof), baseline, artifact
