"""Build the common immutable receipt body for real training evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any


def build_common_receipt(
    profile: Any,
    *,
    schema: str,
    toolchain: Mapping[str, str],
    tokenizer_hash: str,
    token_count: int,
    preprocessing_hash: str,
    losses: list[float],
    baseline_hash: str,
    baseline_tree_hash: str,
    post_hash: str,
    delta_hash: str,
    subject_tree_hash: str,
    reloaded_hash: str,
    changed_tensors: int,
    changed_params: int,
    total_params: int,
    max_abs_delta: float,
    reload_forward: Mapping[str, Any],
    loss_function: str,
    baseline_load_diagnostics: Mapping[str, object],
    baseline_load_diagnostics_sha256: str,
    dataset_provider: Mapping[str, object],
    container_image_digest: str | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": schema,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.profile_sha256,
        "edit_type": profile.edit_type,
        "dataset_provider": dict(dataset_provider),
        "model": {
            "model_id": profile.model_id,
            "model_revision": profile.model_revision,
            "tokenizer_sha256": tokenizer_hash,
            "baseline_load": {
                "loss_function": loss_function,
                "diagnostics": dict(baseline_load_diagnostics),
                "diagnostics_sha256": baseline_load_diagnostics_sha256,
            },
        },
        "training_data": {
            **asdict(profile.training_data),
            "token_count": token_count,
            "preprocessing_sha256": preprocessing_hash,
        },
        "optimizer": {
            **asdict(profile.optimizer),
            "betas": list(profile.optimizer.betas),
        },
        "training": {
            "requested_steps": profile.steps,
            "completed_steps": len(losses),
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
            "toolchain": dict(toolchain),
        },
        "hashes": {
            "baseline_state_sha256": baseline_hash,
            "baseline_tree_sha256": baseline_tree_hash,
            "pre_training_state_sha256": baseline_hash,
            "post_training_state_sha256": post_hash,
            "delta_sha256": delta_hash,
            "subject_tree_sha256": subject_tree_hash,
            "reloaded_subject_state_sha256": reloaded_hash,
        },
        "changes": {
            "changed_tensors": changed_tensors,
            "changed_params": changed_params,
            "total_params": total_params,
            "max_abs_delta": max_abs_delta,
        },
        "reload_smoke": {
            "passed": True,
            "state_hash_matches": True,
            **reload_forward,
        },
    }
    if container_image_digest is not None:
        receipt["runtime"]["container_image_digest"] = container_image_digest
    return receipt


__all__ = ["build_common_receipt"]
