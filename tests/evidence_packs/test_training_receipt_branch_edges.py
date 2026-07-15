from __future__ import annotations

from typing import Any

import pytest

from scripts.evidence_packs.python.editing.training_contract import (
    FineTuneTrainingProfile,
    LoraTrainingProfile,
    load_training_profile,
)
from scripts.evidence_packs.python.editing.training_receipt import (
    TrainingReceiptError,
    require_valid_training_receipt,
    training_receipt_errors,
    with_receipt_digest,
)
from tests.evidence_packs._support_training_receipt import (
    receipt_sha as _sha,
)
from tests.evidence_packs._support_training_receipt import (
    valid_training_receipt as _valid_receipt,
)


def _errors(receipt: dict[str, Any], profile: Any) -> list[str]:
    try:
        receipt = with_receipt_digest(receipt)
    except (TypeError, ValueError):
        pass
    return training_receipt_errors(receipt, profile=profile)


def test_receipt_rejects_non_object_sections_and_unknown_fields() -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    receipt = _valid_receipt(profile)
    receipt["unknown"] = True
    for key in (
        "model",
        "training_data",
        "optimizer",
        "training",
        "seed",
        "runtime",
        "hashes",
        "changes",
        "reload_smoke",
        "lora",
    ):
        receipt[key] = None

    errors = _errors(receipt, profile)

    assert any("unsupported field" in error for error in errors)
    for key in (
        "model",
        "training_data",
        "optimizer",
        "training",
        "seed",
        "runtime",
        "hashes",
        "changes",
        "reload_smoke",
        "lora",
    ):
        assert f"{key} must be an object" in errors


def test_receipt_reports_identity_model_and_data_tampering_together() -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    assert isinstance(profile, FineTuneTrainingProfile)
    receipt = _valid_receipt(profile)
    receipt.update(
        {
            "schema": "wrong",
            "profile_id": "wrong",
            "profile_sha256": _sha("wrong"),
            "edit_type": "lora_merge",
            "unknown": True,
        }
    )
    receipt["model"].update(
        {
            "model_id": "wrong",
            "model_revision": "wrong",
            "tokenizer_sha256": "bad",
            "extra": True,
        }
    )
    receipt["training_data"].update(
        {
            "path": "wrong",
            "sha256": "bad",
            "rows": 0,
            "text_field": "wrong",
            "token_count": True,
            "preprocessing_sha256": "bad",
            "extra": True,
        }
    )

    errors = _errors(receipt, profile)

    expected = (
        "unknown schema",
        "profile_id does not match",
        "profile_sha256 does not match",
        "edit_type does not match",
        "model contains unsupported",
        "model.model_id",
        "model.model_revision",
        "model.tokenizer_sha256",
        "training_data contains unsupported",
        "training_data.path",
        "training_data.sha256",
        "training_data.rows",
        "training_data.text_field",
        "training_data.token_count",
        "training_data.preprocessing_sha256",
    )
    assert all(any(fragment in error for error in errors) for fragment in expected)


def test_receipt_reports_optimizer_training_seed_and_runtime_tampering() -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    receipt = _valid_receipt(profile)
    receipt["optimizer"] = {
        "name": "sgd",
        "learning_rate": True,
        "betas": [0.1],
        "eps": 0.0,
        "weight_decay": -1.0,
        "extra": True,
    }
    receipt["training"].update(
        {
            "requested_steps": 999,
            "completed_steps": True,
            "micro_batch_size": 999,
            "gradient_accumulation_steps": 999,
            "max_sequence_length": 999,
            "losses": [],
            "initial_loss": True,
            "final_loss": float("inf"),
            "extra": True,
        }
    )
    receipt["seed"].update(
        {
            "python": -1,
            "torch_cpu": -1,
            "torch_cuda": -1,
            "deterministic_algorithms": False,
            "extra": True,
        }
    )
    receipt["runtime"].update(
        {
            "device": "cuda",
            "dtype": "float16",
            "toolchain": None,
            "container_image_digest": "latest",
            "extra": True,
        }
    )

    errors = training_receipt_errors(receipt, profile=profile)

    expected = (
        "optimizer contains unsupported",
        "optimizer.name",
        "optimizer.learning_rate",
        "optimizer.betas must contain",
        "optimizer.eps",
        "optimizer.weight_decay",
        "training contains unsupported",
        "training.requested_steps",
        "training.completed_steps",
        "training.losses must be a non-empty",
        "training.initial_loss",
        "training.final_loss",
        "seed contains unsupported",
        "seed.python",
        "seed.deterministic_algorithms",
        "runtime contains unsupported",
        "runtime.device",
        "runtime.dtype",
        "runtime.toolchain must be an object",
        "runtime.container_image_digest",
    )
    assert all(any(fragment in error for error in errors) for fragment in expected)
    assert any("not canonical JSON" in error for error in errors)


def test_receipt_rejects_loss_history_and_toolchain_version_inconsistency() -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    receipt = _valid_receipt(profile)
    receipt["optimizer"]["betas"] = [0.1, 0.2]
    receipt["training"].update(
        {
            "completed_steps": 3,
            "losses": [9.0, "bad"],
            "initial_loss": 1.0,
            "final_loss": 2.0,
        }
    )
    receipt["runtime"]["toolchain"] = {
        "python": "0.0.0",
        "torch": "",
        "transformers": 5,
        "peft": "0.0.0",
        "unknown": "x",
    }

    errors = _errors(receipt, profile)

    assert "optimizer.betas do not match the profile" in errors
    assert "training.losses[1] must be finite" in errors
    assert "training.losses must contain one value per completed step" in errors
    assert "training.initial_loss disagrees with the first step loss" in errors
    assert "training.final_loss disagrees with the final step loss" in errors
    assert any("runtime.toolchain.torch must be a version" in error for error in errors)
    assert any(
        "runtime.toolchain.transformers must be a version" in error for error in errors
    )
    assert any("runtime.toolchain.python does not match" in error for error in errors)
    assert any("runtime.toolchain.peft does not match" in error for error in errors)
    assert any("runtime.toolchain contains unsupported" in error for error in errors)


def test_receipt_rejects_hash_reload_and_lora_cross_binding_tampering() -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    assert isinstance(profile, LoraTrainingProfile)
    receipt = _valid_receipt(profile)
    receipt["hashes"].update(
        {
            "delta_sha256": "bad",
            "reloaded_subject_state_sha256": _sha("other"),
            "extra": True,
        }
    )
    receipt["changes"].update(
        {"changed_tensors": True, "max_abs_delta": float("nan"), "extra": True}
    )
    receipt["reload_smoke"].update(
        {
            "passed": False,
            "state_hash_matches": False,
            "inference_performed": False,
            "all_logits_finite": False,
            "repeat_runs": 1,
            "input_sha256": "bad",
            "logits_sha256": "bad",
            "logits_shape": [0],
            "device": "wrong",
            "extra": True,
        }
    )
    receipt["lora"].update(
        {
            "base_state_before_adapter_sha256": _sha("wrong-base"),
            "merged_state_sha256": _sha("wrong-merge"),
            "merge_method": "",
            "extra": True,
        }
    )

    errors = training_receipt_errors(receipt, profile=profile)

    expected = (
        "hashes contains unsupported",
        "hashes.delta_sha256",
        "reloaded subject state",
        "changes contains unsupported",
        "changed_tensors",
        "max_abs_delta",
        "reload_smoke contains unsupported",
        "reload_smoke.passed",
        "reload_smoke.state_hash_matches",
        "reload_smoke.inference_performed",
        "reload_smoke.all_logits_finite",
        "reload_smoke.repeat_runs",
        "reload_smoke.input_sha256",
        "reload_smoke.logits_sha256",
        "reload_smoke.logits_shape",
        "reload_smoke.device",
        "lora contains unsupported",
        "LoRA base state",
        "LoRA merged state",
        "merge_method",
    )
    assert all(any(fragment in error for error in errors) for fragment in expected)
    assert any("not canonical JSON" in error for error in errors)

    missing_hash_context = _valid_receipt(profile)
    missing_hash_context["hashes"] = None
    errors = _errors(missing_hash_context, profile)
    assert "hashes must be an object" in errors
    assert not any("LoRA base state must match" in error for error in errors)


def test_receipt_digest_and_public_validator_fail_closed() -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    assert training_receipt_errors([], profile=profile) == [
        "training receipt must be an object"
    ]

    invalid = _valid_receipt(profile)
    invalid["receipt_sha256"] = "bad"
    assert (
        "receipt_sha256 must be a canonical sha256 digest"
        in training_receipt_errors(invalid, profile=profile)
    )

    invalid = _valid_receipt(profile)
    invalid["receipt_sha256"] = _sha("wrong")
    assert "receipt_sha256 does not match canonical receipt content" in (
        training_receipt_errors(invalid, profile=profile)
    )
    with pytest.raises(TrainingReceiptError, match="does not match canonical"):
        require_valid_training_receipt(invalid, profile=profile)

    noncanonical = _valid_receipt(profile)
    noncanonical["model"]["extra"] = object()
    errors = training_receipt_errors(noncanonical, profile=profile)
    assert any("not canonical JSON" in error for error in errors)
