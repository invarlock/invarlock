from __future__ import annotations

import copy

import pytest

from invarlock.training_evidence import (
    TrainingEvidenceProofError,
    canonical_training_evidence_proof_sha256,
    is_training_edit_type,
    require_valid_training_evidence_proof,
    training_evidence_proof_errors,
    with_training_evidence_proof_digest,
)
from scripts.evidence_packs.python.editing.training_contract import (
    LoraTrainingProfile,
    load_training_profile,
)
from scripts.evidence_packs.python.editing.training_receipt import (
    with_receipt_digest,
)
from tests.evidence_packs._support_training_evidence_proof import _identity, _proof_for
from tests.evidence_packs._support_training_receipt import (
    receipt_sha,
    valid_training_receipt,
)


@pytest.mark.parametrize("profile_id", ["tiny_gpt2_lora_v1", "tiny_gpt2_full_ft_v1"])
def test_real_profile_receipt_and_bound_proof_pass(profile_id: str) -> None:
    receipt = valid_training_receipt(load_training_profile(profile_id))
    proof, baseline, artifact = _proof_for(receipt)

    assert (
        training_evidence_proof_errors(
            proof,
            receipt,
            expected_edit_type=receipt["edit_type"],
            expected_baseline_identity=baseline,
            expected_artifact_identity=artifact,
        )
        == []
    )
    validated = require_valid_training_evidence_proof(
        proof,
        receipt,
        expected_edit_type=receipt["edit_type"],
        expected_baseline_identity=baseline,
        expected_artifact_identity=artifact,
    )
    assert validated == proof
    assert validated is not proof


def test_proof_digest_is_canonical_and_self_excluding() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    proof, _, _ = _proof_for(receipt)

    assert (
        canonical_training_evidence_proof_sha256(dict(reversed(proof.items())))
        == proof["proof_sha256"]
    )


def test_only_real_training_edit_types_are_dispatchable() -> None:
    assert is_training_edit_type("lora_merge")
    assert is_training_edit_type("fine_tune")
    assert not is_training_edit_type("synthetic_lowrank_delta")
    assert not is_training_edit_type("quant_rtn")
    assert not is_training_edit_type(None)


@pytest.mark.parametrize(
    "declared_backend",
    ("alternate_backend_v2", "full_parameter_optimizer_training"),
)
def test_producer_declaration_must_be_canonical_for_the_edit_type(
    declared_backend: str,
) -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    assert isinstance(profile, LoraTrainingProfile)
    receipt = valid_training_receipt(profile)
    proof, _, _ = _proof_for(receipt)
    provenance = proof["provenance"]
    assert isinstance(provenance, dict)
    provenance["producer_declared_training_backend"] = declared_backend
    proof = with_training_evidence_proof_digest(proof)

    assert training_evidence_proof_errors(proof, receipt) == [
        "training proof.provenance.producer_declared_training_backend must be "
        "the canonical producer declaration 'peft_lora_train_and_merge' for "
        "'lora_merge'"
    ]


def test_retired_schema_is_rejected() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_lora_v1"))
    proof, _, _ = _proof_for(receipt)
    proof["schema"] = "invarlock/training-evidence-proof-v2"
    proof = with_training_evidence_proof_digest(proof)

    errors = training_evidence_proof_errors(proof, receipt)

    assert errors == ["training evidence proof has an unrecognized schema"]


def test_history_attestation_provenance_claim_is_rejected() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_lora_v1"))
    proof, _, _ = _proof_for(receipt)
    proof["provenance"] = {
        "kind": "real_optimized_training",
        "training_backend": "peft_lora_train_and_merge",
        "synthetic": False,
    }
    proof = with_training_evidence_proof_digest(proof)

    errors = training_evidence_proof_errors(proof, receipt)

    assert any("training proof.provenance has unbound" in error for error in errors)
    assert (
        "training proof.provenance.kind must be artifact_replay_verification" in errors
    )


def test_lora_receipt_requires_adapter_state_and_merge_binding() -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    assert isinstance(profile, LoraTrainingProfile)
    receipt = valid_training_receipt(profile)
    lora = receipt["lora"]
    runtime = receipt["runtime"]
    assert isinstance(lora, dict)
    assert isinstance(runtime, dict)
    assert isinstance(runtime["toolchain"], dict)
    lora["trained_adapter_state_sha256"] = lora["initial_adapter_state_sha256"]
    lora["serialized_adapter_state_sha256"] = lora["initial_adapter_state_sha256"]
    lora["merge_method"] = "synthetic_lowrank_delta"
    del runtime["toolchain"]["peft"]
    receipt = with_receipt_digest(receipt)
    proof, _, _ = _proof_for(receipt)

    errors = training_evidence_proof_errors(proof, receipt)

    assert any("trained adapter must differ" in error for error in errors)
    assert any("merge_method" in error for error in errors)
    assert any("toolchain" in error and "peft" in error for error in errors)


def test_noop_receipt_fails_even_when_boolean_claims_are_true() -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    receipt = valid_training_receipt(profile)
    hashes = receipt["hashes"]
    changes = receipt["changes"]
    lora = receipt["lora"]
    assert isinstance(hashes, dict)
    assert isinstance(changes, dict)
    assert isinstance(lora, dict)
    hashes["post_training_state_sha256"] = hashes["baseline_state_sha256"]
    hashes["reloaded_subject_state_sha256"] = hashes["baseline_state_sha256"]
    hashes["subject_tree_sha256"] = hashes["baseline_tree_sha256"]
    changes["changed_tensors"] = 0
    changes["max_abs_delta"] = 0.0
    lora["trained_adapter_state_sha256"] = lora["initial_adapter_state_sha256"]
    lora["serialized_adapter_state_sha256"] = lora["initial_adapter_state_sha256"]
    lora["merged_state_sha256"] = hashes["baseline_state_sha256"]
    receipt = with_receipt_digest(receipt)
    proof, _, _ = _proof_for(receipt)

    errors = training_evidence_proof_errors(proof, receipt)

    expected = (
        "post-training state must differ",
        "subject tree must differ",
        "changed_tensors must be positive",
        "max_abs_delta must be positive",
        "trained adapter must differ",
    )
    assert all(any(fragment in error for error in errors) for fragment in expected)


def test_artifact_and_runtime_proofs_are_bound_to_the_receipt_and_subject() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    proof, baseline, artifact = _proof_for(receipt)
    replay = proof["artifact_replay"]
    runtime_reload = proof["runtime_reload"]
    assert isinstance(replay, dict)
    assert isinstance(runtime_reload, dict)
    replay["artifact_identity"] = _identity("wrong-replayed-artifact")
    replay["saved_artifact_verified"] = False
    runtime_reload["logits_sha256"] = receipt_sha("forged-logits")
    runtime_reload["repeat_deterministic"] = False
    proof = with_training_evidence_proof_digest(proof)

    errors = training_evidence_proof_errors(
        proof,
        receipt,
        expected_baseline_identity=baseline,
        expected_artifact_identity=artifact,
    )

    assert "training proof.artifact_replay.artifact_identity mismatch" in errors
    assert (
        "training proof.artifact_replay.saved_artifact_verified must be true" in errors
    )
    assert any("runtime_reload.logits_sha256" in error for error in errors)
    assert "training proof.runtime_reload.repeat_deterministic must be true" in errors


def test_artifact_replay_binds_source_migration_and_loss_semantics() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    proof, _, _ = _proof_for(receipt)
    replay = proof["artifact_replay"]
    assert isinstance(replay, dict)
    replay["baseline_load_diagnostics_sha256"] = receipt_sha("forged-load")
    replay["loss_function"] = "fallback"
    proof = with_training_evidence_proof_digest(proof)

    errors = training_evidence_proof_errors(proof, receipt)

    assert any("baseline_load_diagnostics_sha256" in error for error in errors)
    assert any("loss_function" in error for error in errors)


def test_training_proof_rejects_external_subjects_and_wrong_baseline_revision() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    proof, baseline, _ = _proof_for(receipt)
    replay = proof["artifact_replay"]
    runtime_reload = proof["runtime_reload"]
    assert isinstance(replay, dict)
    assert isinstance(runtime_reload, dict)
    remote_subject = {"kind": "remote_revision", "revision": "a" * 40}
    wrong_baseline = {"kind": "remote_revision", "revision": "b" * 40}
    proof["artifact_identity"] = remote_subject
    proof["baseline_identity"] = wrong_baseline
    replay["artifact_identity"] = remote_subject
    replay["baseline_identity"] = wrong_baseline
    runtime_reload["artifact_identity"] = remote_subject
    proof = with_training_evidence_proof_digest(proof)

    errors = training_evidence_proof_errors(proof, receipt)

    assert any(
        "artifact_identity must be a local_checkpoint_tree" in error for error in errors
    )
    assert (
        "training evidence proof.baseline_identity does not bind receipt revision"
        in errors
    )
    assert baseline["revision"] != wrong_baseline["revision"]


def test_fine_tune_rejects_a_lora_block_and_mismatched_expected_identity() -> None:
    lora_profile = load_training_profile("tiny_gpt2_lora_v1")
    fine_tune = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    lora_receipt = valid_training_receipt(lora_profile)
    fine_tune["lora"] = copy.deepcopy(lora_receipt["lora"])
    fine_tune = with_receipt_digest(fine_tune)
    proof, baseline, artifact = _proof_for(fine_tune)

    errors = training_evidence_proof_errors(
        proof,
        fine_tune,
        expected_baseline_identity=baseline,
        expected_artifact_identity=_identity("other-artifact"),
    )

    assert any(
        "fine_tune training receipt must not carry LoRA evidence" in error
        for error in errors
    )
    assert (
        "training evidence proof artifact_identity does not match expected artifact"
        in errors
    )
    with pytest.raises(TrainingEvidenceProofError, match="LoRA evidence"):
        require_valid_training_evidence_proof(proof, fine_tune)


def test_closed_schemas_and_self_digests_reject_rebound_forgery() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_lora_v1"))
    proof, _, _ = _proof_for(receipt)
    provenance = proof["provenance"]
    assert isinstance(provenance, dict)
    proof["provenance"] = {
        **provenance,
        "source_edit_type": "synthetic_lowrank_delta",
    }
    proof = with_training_evidence_proof_digest(proof)
    errors = training_evidence_proof_errors(proof, receipt)
    assert any("training proof.provenance has unbound" in error for error in errors)

    tampered = copy.deepcopy(proof)
    tampered["artifact_identity"] = _identity("tampered-artifact")
    errors = training_evidence_proof_errors(tampered, receipt)
    assert "training evidence proof.proof_sha256 does not bind content" in errors

    bad_receipt = copy.deepcopy(receipt)
    training = bad_receipt["training"]
    assert isinstance(training, dict)
    training["optimization_performed"] = False
    errors = training_evidence_proof_errors(proof, bad_receipt)
    assert "training receipt.training.optimization_performed must be true" in errors
    assert "training receipt.receipt_sha256 does not bind content" in errors


def test_invalid_dispatch_target_fails_closed() -> None:
    receipt = valid_training_receipt(load_training_profile("tiny_gpt2_full_ft_v1"))
    proof, _, _ = _proof_for(receipt)

    errors = training_evidence_proof_errors(
        proof, receipt, expected_edit_type="synthetic_lowrank_delta"
    )

    assert errors == ["expected edit type is not a training-profile edit type"]
