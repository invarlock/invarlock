from __future__ import annotations

from collections.abc import Callable

import pytest

from invarlock.training_evidence import (
    _lora_proof_errors,
    training_evidence_proof_errors,
    with_training_evidence_proof_digest,
)
from invarlock.training_evidence_contracts import common
from invarlock.training_evidence_contracts.receipt import _receipt_errors
from scripts.evidence_packs.python.editing.training_contract import (
    load_training_profile,
)
from scripts.evidence_packs.python.editing.training_receipt import with_receipt_digest
from tests.evidence_packs._support_training_evidence_proof import _proof_for
from tests.evidence_packs._support_training_receipt import (
    receipt_sha,
    valid_training_receipt,
)

Mutation = Callable[[dict[str, object]], None]


def _set(path: tuple[str, ...], value: object) -> Mutation:
    def mutate(payload: dict[str, object]) -> None:
        target: dict[str, object] = payload
        for part in path[:-1]:
            child = target[part]
            assert isinstance(child, dict)
            target = child
        target[path[-1]] = value

    return mutate


def _delete(path: tuple[str, ...]) -> Mutation:
    def mutate(payload: dict[str, object]) -> None:
        target: dict[str, object] = payload
        for part in path[:-1]:
            child = target[part]
            assert isinstance(child, dict)
            target = child
        del target[path[-1]]

    return mutate


def _copy_value(source: tuple[str, ...], target: tuple[str, ...]) -> Mutation:
    def mutate(payload: dict[str, object]) -> None:
        source_parent: dict[str, object] = payload
        for part in source[:-1]:
            child = source_parent[part]
            assert isinstance(child, dict)
            source_parent = child
        _set(target, source_parent[source[-1]])(payload)

    return mutate


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        (_set(("schema",), "retired"), "unrecognized schema"),
        (_set(("edit_type",), "synthetic"), "edit_type must be"),
        (_set(("profile_id",), " bad "), "profile_id is invalid"),
        (_set(("profile_sha256",), "bad"), "profile_sha256 must be"),
        (_set(("model", "model_id"), ""), "model.model_id is invalid"),
        (_set(("model", "model_revision"), "main"), "model_revision must be pinned"),
        (_set(("model", "tokenizer_sha256"), "bad"), "tokenizer_sha256"),
        (
            _set(("model", "baseline_load", "loss_function"), "fallback"),
            "loss_function is invalid",
        ),
        (
            _set(("model", "baseline_load", "diagnostics", "schema"), "retired"),
            "diagnostics schema is invalid",
        ),
        (
            _set(("model", "baseline_load", "diagnostics", "policy"), "loose"),
            "diagnostics policy is invalid",
        ),
        (
            _set(("model", "baseline_load", "diagnostics", "missing_keys"), ["x"]),
            "missing_keys must be empty",
        ),
        (
            _set(("model", "baseline_load", "diagnostics", "mismatched_keys"), ["x"]),
            "mismatched_keys must be empty",
        ),
        (
            _set(("model", "baseline_load", "diagnostics", "error_msgs"), ["x"]),
            "error_msgs must be empty",
        ),
        (
            _set(
                ("model", "baseline_load", "diagnostics", "unexpected_keys"), ["z", "a"]
            ),
            "unexpected_keys must be sorted",
        ),
        (
            _set(("model", "baseline_load", "diagnostics_sha256"), "bad"),
            "diagnostics_sha256 must be",
        ),
        (
            _set(
                ("model", "baseline_load", "diagnostics_sha256"),
                receipt_sha("wrong diagnostics"),
            ),
            "diagnostics do not match",
        ),
        (_set(("training_data", "path"), "../secret"), "training_data.path is invalid"),
        (_set(("training_data", "sha256"), "bad"), "training_data.sha256"),
        (_set(("training_data", "rows"), 0), "rows must be positive"),
        (_set(("training_data", "text_field"), "bad field"), "text_field is invalid"),
        (_set(("training_data", "token_count"), 0), "token_count must be positive"),
        (
            _set(("training_data", "preprocessing_sha256"), "bad"),
            "preprocessing_sha256",
        ),
        (_set(("optimizer", "name"), "bad name"), "optimizer.name is invalid"),
        (_set(("optimizer", "learning_rate"), 0), "learning_rate must be positive"),
        (_set(("optimizer", "betas"), [0.9]), "betas must contain"),
        (_set(("optimizer", "betas"), [0.9, 1.0]), "betas must contain"),
        (_set(("optimizer", "eps"), float("nan")), "eps must be positive"),
        (_set(("optimizer", "weight_decay"), -1), "weight_decay must be nonnegative"),
        (_set(("training", "requested_steps"), 0), "requested_steps must be positive"),
        (_set(("training", "completed_steps"), 0), "completed_steps must be positive"),
        (
            _set(("training", "micro_batch_size"), 0),
            "micro_batch_size must be positive",
        ),
        (
            _set(("training", "gradient_accumulation_steps"), 0),
            "gradient_accumulation_steps must be positive",
        ),
        (
            _set(("training", "max_sequence_length"), 0),
            "max_sequence_length must be positive",
        ),
        (_set(("training", "completed_steps"), 1), "completed schedule"),
        (_set(("training", "losses"), []), "losses must be non-empty"),
        (_set(("training", "losses"), [float("nan")]), "losses[0] must be finite"),
        (_set(("training", "losses"), [1.0]), "losses must match"),
        (
            _set(("training", "initial_loss"), float("nan")),
            "initial_loss must be finite",
        ),
        (_set(("training", "final_loss"), float("nan")), "final_loss must be finite"),
        (_set(("training", "initial_loss"), 99.0), "initial_loss does not bind"),
        (_set(("training", "final_loss"), 99.0), "final_loss does not bind"),
        (
            _set(("training", "optimization_performed"), False),
            "optimization_performed must be true",
        ),
        (
            _set(("training", "training_data_used"), False),
            "training_data_used must be true",
        ),
        (_set(("seed", "python"), -1), "seed values must be nonnegative"),
        (_set(("seed", "torch_cpu"), 999), "seed values must agree"),
        (
            _set(("seed", "deterministic_algorithms"), "yes"),
            "deterministic_algorithms must be boolean",
        ),
        (_set(("runtime", "device"), "xpu"), "runtime.device is invalid"),
        (_set(("runtime", "dtype"), "float64"), "runtime.dtype is invalid"),
        (
            _set(("runtime", "toolchain", "python"), "bad version"),
            "toolchain.python is invalid",
        ),
        (_set(("hashes", "baseline_state_sha256"), "bad"), "baseline_state_sha256"),
        (
            _set(
                ("hashes", "pre_training_state_sha256"),
                receipt_sha("wrong pretraining"),
            ),
            "pre-training state must match",
        ),
        (
            _copy_value(
                ("hashes", "pre_training_state_sha256"),
                ("hashes", "post_training_state_sha256"),
            ),
            "post-training state must differ",
        ),
        (
            _set(
                ("hashes", "reloaded_subject_state_sha256"), receipt_sha("wrong reload")
            ),
            "reloaded state must match",
        ),
        (
            _copy_value(
                ("hashes", "baseline_tree_sha256"), ("hashes", "subject_tree_sha256")
            ),
            "subject tree must differ",
        ),
        (_set(("changes", "changed_tensors"), 0), "changed_tensors must be positive"),
        (_set(("changes", "max_abs_delta"), 0), "max_abs_delta must be positive"),
        (_set(("reload_smoke", "passed"), False), "reload_smoke.passed must be true"),
        (
            _set(("reload_smoke", "state_hash_matches"), False),
            "state_hash_matches must be true",
        ),
        (
            _set(("reload_smoke", "inference_performed"), False),
            "inference_performed must be true",
        ),
        (
            _set(("reload_smoke", "all_logits_finite"), False),
            "all_logits_finite must be true",
        ),
        (_set(("reload_smoke", "repeat_runs"), 1), "repeat_runs must equal two"),
        (_set(("reload_smoke", "input_sha256"), "bad"), "input_sha256"),
        (_set(("reload_smoke", "logits_shape"), []), "logits_shape is invalid"),
        (_set(("reload_smoke", "device"), "cuda"), "device must match"),
        (_set(("receipt_sha256",), "bad"), "receipt_sha256 must be"),
        (_delete(("model",)), "unbound, missing, or arbitrary fields"),
    ],
)
def test_fine_tune_receipt_rejects_each_corrupt_contract_surface(
    mutation: Mutation,
    error_fragment: str,
) -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    mutation(receipt)

    errors, observed = _receipt_errors(receipt)

    assert observed is receipt
    assert any(error_fragment in error for error in errors), errors


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    [
        (
            _set(("lora", "initial_adapter_state_sha256"), "bad"),
            "initial_adapter_state_sha256",
        ),
        (
            _set(("lora", "trained_adapter_state_sha256"), "bad"),
            "trained_adapter_state_sha256",
        ),
        (
            _set(("lora", "trained_adapter_state_sha256"), None),
            "trained_adapter_state_sha256",
        ),
        (
            _set(("lora", "trained_adapter_state_sha256"), "same"),
            "trained_adapter_state_sha256",
        ),
        (
            _set(
                ("lora", "serialized_adapter_state_sha256"),
                receipt_sha("wrong serialized adapter"),
            ),
            "serialized adapter must match",
        ),
        (
            _set(
                ("lora", "base_state_after_training_sha256"),
                receipt_sha("wrong base state"),
            ),
            "base state must remain frozen",
        ),
        (
            _set(
                ("lora", "base_state_manifest_before_adapter_sha256"),
                receipt_sha("wrong manifest"),
            ),
            "streaming manifests must remain frozen",
        ),
        (
            _set(("lora", "state_evidence_policy"), "full-memory"),
            "state_evidence_policy is invalid",
        ),
        (_set(("lora", "merge_target_names"), []), "merge_target_names must be sorted"),
        (
            _set(
                ("lora", "expected_merge_target_names_sha256"),
                receipt_sha("wrong expected targets"),
            ),
            "merge_target_names digest mismatch",
        ),
        (
            _set(
                ("lora", "observed_merged_changed_names_sha256"),
                receipt_sha("wrong observed targets"),
            ),
            "observed merge targets",
        ),
        (_set(("lora", "merge_scope_exact"), False), "merge_scope_exact must be true"),
        (
            _set(("lora", "merged_changed_tensor_count"), 0),
            "merged_changed_tensor_count must be positive",
        ),
        (
            _set(("lora", "merged_changed_tensor_count"), 99),
            "changed tensor count must match merge targets",
        ),
        (
            _set(("changes", "changed_tensors"), 99),
            "changed tensor count must match receipt changes",
        ),
        (
            _set(
                ("lora", "base_state_before_adapter_sha256"),
                receipt_sha("wrong baseline binding"),
            ),
            "base state must bind baseline",
        ),
        (
            _set(("lora", "merged_state_sha256"), receipt_sha("wrong merged binding")),
            "merged state must bind subject",
        ),
        (
            _set(("lora", "adapter_training_performed"), False),
            "adapter_training_performed must be true",
        ),
        (
            _set(("lora", "adapter_merge_performed"), False),
            "adapter_merge_performed must be true",
        ),
        (
            _set(("lora", "adapter_optimizer_steps"), 0),
            "adapter_optimizer_steps must be positive",
        ),
        (
            _set(("lora", "adapter_optimizer_steps"), 99),
            "adapter_optimizer_steps must bind",
        ),
        (
            _set(("lora", "trainable_parameter_count"), 0),
            "trainable_parameter_count must be positive",
        ),
        (
            _set(("lora", "adapter_modules_before_merge"), 0),
            "adapter_modules_before_merge must be positive",
        ),
        (
            _set(("lora", "adapter_modules_after_merge"), 1),
            "adapter_modules_after_merge must be zero",
        ),
        (_set(("lora", "merge_method"), "manual"), "merge_method must be"),
        (
            _delete(("runtime", "toolchain", "peft")),
            "unbound, missing, or arbitrary fields",
        ),
    ],
)
def test_lora_receipt_rejects_each_corrupt_adapter_or_merge_claim(
    mutation: Mutation,
    error_fragment: str,
) -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_lora_v1"))
    mutation(receipt)

    errors, observed = _receipt_errors(receipt)

    assert observed is receipt
    assert any(error_fragment in error for error in errors), errors


def test_receipt_validator_rejects_non_objects_and_unexpected_lora_evidence() -> None:
    assert _receipt_errors([]) == (["training receipt must be an object"], None)
    assert _receipt_errors({1: "bad"}) == (["training receipt must be an object"], None)

    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    receipt["lora"] = {}
    errors, _ = _receipt_errors(receipt)
    assert "fine_tune training receipt must not carry LoRA evidence" in errors


def test_receipt_digest_failure_from_noncanonical_content_is_reported() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    receipt["receipt_sha256"] = receipt_sha("noncanonical receipt")
    receipt["not_json"] = object()

    errors, _ = _receipt_errors(receipt)

    assert any("canonicalized as JSON" in error for error in errors)


def test_container_runtime_digest_is_checked_when_present() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    runtime = receipt["runtime"]
    assert isinstance(runtime, dict)
    runtime["container_image_digest"] = "bad"
    receipt = with_receipt_digest(receipt)

    errors, _ = _receipt_errors(receipt)

    assert any("container_image_digest" in error for error in errors)


def test_common_contract_helpers_reject_ambiguous_values() -> None:
    assert common._finite_float(True) is None
    assert common._finite_float("nan") is None
    assert common._finite_float(float("inf")) is None
    assert common._finite_float(1) == 1.0
    assert not common._is_sha256(True)
    assert not common._is_text(" bad ")
    assert not common._is_text(1)

    errors: list[str] = []
    assert (
        common._exact_mapping([], label="value", fields=frozenset(), errors=errors)
        is None
    )
    assert errors == ["value must be an object"]

    errors = []
    assert (
        common._identity(
            {"kind": "wrong"},
            label="subject",
            errors=errors,
            allow_remote=False,
        )
        is None
    )
    assert errors


@pytest.mark.parametrize(
    ("profile_id", "mutation", "error_fragment"),
    [
        (
            "tiny_gpt2_full_ft_v1",
            _set(("training_receipt", "profile_id"), "wrong"),
            "training_receipt.profile_id does not bind",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("training_receipt",), []),
            "training proof.training_receipt must be an object",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("provenance",), []),
            "training proof.provenance must be an object",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("provenance", "kind"), "history_claim"),
            "provenance.kind must be",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(
                ("provenance", "producer_declared_training_backend"), "bad declaration"
            ),
            "producer_declared_training_backend must be a non-empty",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("artifact_replay",), []),
            "training proof.artifact_replay must be an object",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("artifact_replay", "schema"), "retired"),
            "artifact_replay has an unrecognized schema",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("artifact_replay", "passed"), False),
            "artifact_replay.passed must be true",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("artifact_replay", "saved_artifact_verified"), False),
            "saved_artifact_verified must be true",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("artifact_replay", "reloaded_artifact_verified"), False),
            "reloaded_artifact_verified must be true",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(
                ("artifact_replay", "receipt_sha256"),
                receipt_sha("wrong replay receipt"),
            ),
            "artifact_replay.receipt_sha256 does not bind",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(
                ("artifact_replay", "baseline_identity"),
                {"kind": "remote_revision", "revision": "a" * 40},
            ),
            "artifact_replay.baseline_identity mismatch",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(
                ("artifact_replay", "artifact_identity"),
                {
                    "kind": "local_checkpoint_tree",
                    "sha256": receipt_sha("wrong artifact"),
                },
            ),
            "artifact_replay.artifact_identity mismatch",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(
                ("artifact_replay", "baseline_tree_sha256"),
                receipt_sha("wrong baseline tree"),
            ),
            "artifact_replay.baseline_tree_sha256 does not bind",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("artifact_replay", "changed_tensors"), 99),
            "artifact_replay.changed_tensors does not bind",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("artifact_replay", "loss_function"), "fallback"),
            "artifact_replay.loss_function does not bind",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("runtime_reload",), []),
            "training proof.runtime_reload must be an object",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("runtime_reload", "schema"), "retired"),
            "runtime_reload has an unrecognized schema",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("runtime_reload", "passed"), False),
            "runtime_reload.passed must be true",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("runtime_reload", "all_logits_finite"), False),
            "all_logits_finite must be true",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("runtime_reload", "repeat_deterministic"), False),
            "repeat_deterministic must be true",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(
                ("runtime_reload", "receipt_sha256"),
                receipt_sha("wrong runtime receipt"),
            ),
            "runtime_reload.receipt_sha256 does not bind",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(
                ("runtime_reload", "artifact_identity"),
                {
                    "kind": "local_checkpoint_tree",
                    "sha256": receipt_sha("wrong runtime artifact"),
                },
            ),
            "runtime_reload.artifact_identity mismatch",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(
                ("runtime_reload", "subject_state_sha256"),
                receipt_sha("wrong runtime state"),
            ),
            "runtime_reload.subject_state_sha256 does not bind",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("runtime_reload", "input_sha256"), receipt_sha("wrong input")),
            "runtime_reload.input_sha256 does not bind",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("runtime_reload", "reload_runs"), 1),
            "runtime_reload.reload_runs does not bind",
        ),
        (
            "tiny_gpt2_full_ft_v1",
            _set(("runtime_reload", "device"), "cuda"),
            "runtime_reload.device does not bind",
        ),
        (
            "tiny_gpt2_lora_v1",
            _set(("lora_merge",), []),
            "training proof.lora_merge must be an object",
        ),
        (
            "tiny_gpt2_lora_v1",
            _set(("lora_merge", "schema"), "retired"),
            "lora_merge has an unrecognized schema",
        ),
        (
            "tiny_gpt2_lora_v1",
            _set(
                ("lora_merge", "adapter_identity"),
                {"kind": "remote_revision", "revision": "a" * 40},
            ),
            "adapter_identity",
        ),
        (
            "tiny_gpt2_lora_v1",
            _set(
                ("lora_merge", "adapter_tree_sha256"), receipt_sha("wrong adapter tree")
            ),
            "lora_merge.adapter_tree_sha256 does not bind",
        ),
        (
            "tiny_gpt2_lora_v1",
            _set(("lora_merge", "adapter_training_performed"), False),
            "adapter_training_performed must be true",
        ),
        (
            "tiny_gpt2_lora_v1",
            _set(("lora_merge", "adapter_merge_performed"), False),
            "adapter_merge_performed must be true",
        ),
        (
            "tiny_gpt2_lora_v1",
            _set(("lora_merge", "adapter_modules_after_merge"), 1),
            "adapter_modules_after_merge must be zero",
        ),
        (
            "tiny_gpt2_lora_v1",
            _set(("lora_merge", "merge_method"), "manual"),
            "merge_method must be",
        ),
    ],
)
def test_training_proof_rejects_each_corrupt_replay_binding(
    profile_id: str,
    mutation: Mutation,
    error_fragment: str,
) -> None:
    receipt = valid_training_receipt(load_training_profile(profile_id))
    proof, _, _ = _proof_for(receipt)
    mutation(proof)
    proof = with_training_evidence_proof_digest(proof)

    errors = training_evidence_proof_errors(proof, receipt)

    assert any(error_fragment in error for error in errors), errors


def test_training_proof_rejects_invalid_outer_expectations_and_digest() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    proof, _, _ = _proof_for(receipt)

    assert (
        "training evidence proof must be an object"
        in training_evidence_proof_errors([], receipt)
    )
    errors = training_evidence_proof_errors(
        proof,
        receipt,
        expected_edit_type="synthetic",
        expected_baseline_identity={"kind": "wrong"},
        expected_artifact_identity={"kind": "remote_revision", "revision": "a" * 40},
    )
    assert "expected edit type is not a training-profile edit type" in errors
    assert any("expected baseline identity" in error for error in errors)
    assert any("expected artifact identity" in error for error in errors)

    proof["proof_sha256"] = "bad"
    assert any(
        "proof_sha256 must be" in error
        for error in training_evidence_proof_errors(proof, receipt)
    )


def test_training_proof_rejects_noncanonical_and_stale_digests() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    proof, _, _ = _proof_for(receipt)
    proof["proof_sha256"] = receipt_sha("stale proof")
    assert any(
        "proof_sha256 does not bind" in error
        for error in training_evidence_proof_errors(proof, receipt)
    )

    proof["not_json"] = object()
    assert any(
        "canonicalized as JSON" in error
        for error in training_evidence_proof_errors(proof, receipt)
    )


def test_fine_tune_proof_rejects_lora_block_and_lora_proof_requires_receipt() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    proof, _, _ = _proof_for(receipt)
    proof["lora_merge"] = {}
    proof = with_training_evidence_proof_digest(proof)
    assert "fine_tune proof must not carry a LoRA merge proof" in (
        training_evidence_proof_errors(proof, receipt)
    )

    errors: list[str] = []
    _lora_proof_errors({}, receipt=receipt, errors=errors)
    assert "training proof.lora_merge requires a LoRA training receipt" in errors

    errors = []
    assert (
        common._adapter_identity(
            {"kind": "local_checkpoint_tree", "sha256": "bad"},
            label="adapter",
            errors=errors,
        )
        is None
    )
    assert errors
