from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.evidence_packs.python.editing.training_contract import (
    LoraTrainingProfile,
    TrainingProfile,
    load_training_profile,
)
from scripts.evidence_packs.python.editing.training_receipt import (
    TRAINING_RECEIPT_SCHEMA,
    TrainingReceiptError,
    canonical_receipt_digest,
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


def test_training_receipt_schema_is_repaired_v1() -> None:
    assert TRAINING_RECEIPT_SCHEMA == "invarlock/evidence-pack-training-receipt-v1"


def test_training_receipt_rejects_retired_v3_schema() -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    receipt = _valid_receipt(profile)
    receipt["schema"] = "invarlock/evidence-pack-training-receipt-v3"
    receipt = with_receipt_digest(receipt)

    assert "training receipt has an unknown schema" in training_receipt_errors(
        receipt, profile=profile
    )


@pytest.fixture(params=["tiny_gpt2_lora_v1", "tiny_gpt2_full_ft_v1"])
def profile(request) -> TrainingProfile:
    return load_training_profile(request.param)


def test_valid_real_training_receipts_pass(profile: TrainingProfile) -> None:
    receipt = _valid_receipt(profile)

    assert training_receipt_errors(receipt, profile=profile) == []
    validated = require_valid_training_receipt(receipt, profile=profile)
    assert validated == receipt
    assert validated is not receipt


def test_receipt_digest_is_canonical_and_ignores_its_own_field(
    profile: TrainingProfile,
) -> None:
    receipt = _valid_receipt(profile)
    reversed_receipt = dict(reversed(list(receipt.items())))

    assert canonical_receipt_digest(reversed_receipt) == receipt["receipt_sha256"]
    assert with_receipt_digest(receipt) == receipt


def _tamper_common(receipt: dict[str, Any], case: str) -> None:
    if case == "profile_digest":
        receipt["profile_sha256"] = _sha("wrong-profile")
    elif case == "data_digest":
        receipt["training_data"]["sha256"] = _sha("wrong-data")
    elif case == "optimizer":
        receipt["optimizer"]["learning_rate"] = 0.0
    elif case == "zero_steps":
        receipt["training"]["completed_steps"] = 0
    elif case == "seed":
        receipt["seed"]["python"] += 1
    elif case == "device":
        receipt["runtime"]["device"] = "cuda"
    elif case == "toolchain":
        del receipt["runtime"]["toolchain"]["torch"]
    elif case == "loss_function":
        receipt["model"]["baseline_load"]["loss_function"] = "fallback"
    elif case == "load_diagnostics":
        receipt["model"]["baseline_load"]["diagnostics"]["unexpected_keys"].append(
            "injected.weight"
        )
    elif case == "baseline_mismatch":
        receipt["hashes"]["pre_training_state_sha256"] = _sha("other-baseline")
    elif case == "unchanged_state":
        receipt["hashes"]["post_training_state_sha256"] = receipt["hashes"][
            "pre_training_state_sha256"
        ]
        receipt["hashes"]["reloaded_subject_state_sha256"] = receipt["hashes"][
            "post_training_state_sha256"
        ]
        if "lora" in receipt:
            receipt["lora"]["merged_state_sha256"] = receipt["hashes"][
                "post_training_state_sha256"
            ]
    elif case == "copied_tree":
        receipt["hashes"]["subject_tree_sha256"] = receipt["hashes"][
            "baseline_tree_sha256"
        ]
    elif case == "no_changed_tensors":
        receipt["changes"]["changed_tensors"] = 0
    elif case == "zero_delta":
        receipt["changes"]["max_abs_delta"] = 0.0
    elif case == "reload_flag":
        receipt["reload_smoke"]["passed"] = False
    elif case == "optimization_flag":
        receipt["training"]["optimization_performed"] = False
    elif case == "data_used_flag":
        receipt["training"]["training_data_used"] = False
    else:  # pragma: no cover - test table is exhaustive
        raise AssertionError(case)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("profile_digest", "profile_sha256"),
        ("data_digest", "training_data.sha256"),
        ("optimizer", "optimizer.learning_rate"),
        ("zero_steps", "completed_steps"),
        ("seed", "seed.python"),
        ("device", "runtime.device"),
        ("toolchain", "runtime.toolchain.torch"),
        ("loss_function", "loss_function"),
        ("load_diagnostics", "unexpected_keys"),
        ("baseline_mismatch", "pre-training state"),
        ("unchanged_state", "post-training state must differ"),
        ("copied_tree", "subject checkpoint tree must differ"),
        ("no_changed_tensors", "changed_tensors"),
        ("zero_delta", "max_abs_delta"),
        ("reload_flag", "reload_smoke.passed"),
        ("optimization_flag", "optimization_performed"),
        ("data_used_flag", "training_data_used"),
    ],
)
def test_common_tamper_matrix_fails_after_attacker_rebinds_receipt_digest(
    profile: TrainingProfile,
    case: str,
    message: str,
) -> None:
    receipt = _valid_receipt(profile)
    _tamper_common(receipt, case)
    receipt = with_receipt_digest(receipt)

    errors = training_receipt_errors(receipt, profile=profile)

    assert any(message in error for error in errors), errors


def test_nonfinite_loss_fails_closed(profile: TrainingProfile) -> None:
    receipt = _valid_receipt(profile)
    receipt["training"]["losses"][0] = float("nan")

    errors = training_receipt_errors(receipt, profile=profile)

    assert any("losses[0] must be finite" in error for error in errors)
    assert any("not canonical JSON" in error for error in errors)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("adapter_unchanged", "trained adapter state must differ"),
        ("serialized_adapter_mismatch", "serialized adapter state must match"),
        ("base_mutated", "base model state must remain frozen"),
        ("training_flag", "adapter_training_performed"),
        ("adapter_steps", "adapter_optimizer_steps"),
        ("trainable_count", "trainable_parameter_count"),
        ("merge_flag", "adapter_merge_performed"),
        ("before_modules", "adapter_modules_before_merge"),
        ("after_modules", "adapter_modules_after_merge"),
        ("config_digest", "profile_lora_config_sha256"),
        ("merged_hash", "merged state must match"),
    ],
)
def test_lora_tamper_matrix_rejects_flag_only_merge_proof(
    case: str,
    message: str,
) -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    assert isinstance(profile, LoraTrainingProfile)
    receipt = _valid_receipt(profile)
    lora = receipt["lora"]
    if case == "adapter_unchanged":
        lora["trained_adapter_state_sha256"] = lora["initial_adapter_state_sha256"]
    elif case == "serialized_adapter_mismatch":
        lora["serialized_adapter_state_sha256"] = _sha("stale-serialized-adapter")
    elif case == "base_mutated":
        lora["base_state_after_training_sha256"] = _sha("mutated-base")
    elif case == "training_flag":
        lora["adapter_training_performed"] = False
    elif case == "adapter_steps":
        lora["adapter_optimizer_steps"] = 0
    elif case == "trainable_count":
        lora["trainable_parameter_count"] = 0
    elif case == "merge_flag":
        lora["adapter_merge_performed"] = False
    elif case == "before_modules":
        lora["adapter_modules_before_merge"] = 0
    elif case == "after_modules":
        lora["adapter_modules_after_merge"] = 1
    elif case == "config_digest":
        lora["profile_lora_config_sha256"] = _sha("wrong-config")
    elif case == "merged_hash":
        lora["merged_state_sha256"] = _sha("wrong-merged")
    receipt = with_receipt_digest(receipt)

    errors = training_receipt_errors(receipt, profile=profile)

    assert any(message in error for error in errors), errors


def test_fake_unchanged_lora_receipt_fails_despite_all_boolean_flags() -> None:
    profile = load_training_profile("tiny_gpt2_lora_v1")
    receipt = _valid_receipt(profile)
    baseline = receipt["hashes"]["baseline_state_sha256"]
    receipt["hashes"]["post_training_state_sha256"] = baseline
    receipt["hashes"]["reloaded_subject_state_sha256"] = baseline
    receipt["hashes"]["subject_tree_sha256"] = receipt["hashes"]["baseline_tree_sha256"]
    receipt["changes"] = {"changed_tensors": 0, "max_abs_delta": 0.0}
    receipt["lora"]["trained_adapter_state_sha256"] = receipt["lora"][
        "initial_adapter_state_sha256"
    ]
    receipt["lora"]["merged_state_sha256"] = baseline
    receipt = with_receipt_digest(receipt)

    with pytest.raises(TrainingReceiptError) as exc_info:
        require_valid_training_receipt(receipt, profile=profile)

    message = str(exc_info.value)
    assert "post-training state must differ" in message
    assert "trained adapter state must differ" in message
    assert "changed_tensors" in message


def test_fine_tune_receipt_rejects_lora_flag_block() -> None:
    fine_tune = load_training_profile("tiny_gpt2_full_ft_v1")
    lora = load_training_profile("tiny_gpt2_lora_v1")
    receipt = _valid_receipt(fine_tune)
    receipt["lora"] = copy.deepcopy(_valid_receipt(lora)["lora"])
    receipt = with_receipt_digest(receipt)

    errors = training_receipt_errors(receipt, profile=fine_tune)

    assert any("must not contain LoRA merge evidence" in error for error in errors)


def test_fine_tune_receipt_rejects_unprofiled_peft_toolchain() -> None:
    profile = load_training_profile("tiny_gpt2_full_ft_v1")
    receipt = _valid_receipt(profile)
    receipt["runtime"]["toolchain"]["peft"] = "0.19.1"
    receipt = with_receipt_digest(receipt)

    errors = training_receipt_errors(receipt, profile=profile)

    assert "runtime.toolchain.peft is only valid for LoRA receipts" in errors
