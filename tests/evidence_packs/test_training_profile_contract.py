from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.evidence_packs.python.editing.training_contract import (
    DEFAULT_TRAINING_PROFILES_PATH,
    TRAINING_PROFILES_SCHEMA,
    FineTuneTrainingProfile,
    LoraTrainingProfile,
    TrainingProfileError,
    canonical_profile_digest,
    file_sha256,
    load_training_profile,
    training_profile_errors,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _raw_profiles() -> dict[str, dict[str, object]]:
    payload = json.loads(DEFAULT_TRAINING_PROFILES_PATH.read_text(encoding="utf-8"))
    return payload["profiles"]


def _rebind(profile: dict[str, object]) -> dict[str, object]:
    profile["profile_sha256"] = canonical_profile_digest(profile)
    return profile


def test_training_profile_schema_records_toolchain_binding_upgrade() -> None:
    assert TRAINING_PROFILES_SCHEMA == "invarlock/evidence-pack-training-profiles-v1"


def test_bundled_profiles_load_as_typed_immutable_contracts() -> None:
    lora = load_training_profile("tiny_gpt2_lora_v1", expected_edit_type="lora_merge")
    fine_tune = load_training_profile(
        "tiny_gpt2_full_ft_v1", expected_edit_type="fine_tune"
    )
    cuda_lora = load_training_profile(
        "tiny_gpt2_lora_cuda_v1", expected_edit_type="lora_merge"
    )
    cuda_fine_tune = load_training_profile(
        "tiny_gpt2_full_ft_cuda_v1", expected_edit_type="fine_tune"
    )

    assert isinstance(lora, LoraTrainingProfile)
    assert lora.lora.rank == 2
    assert lora.lora.target_modules == ("c_attn",)
    assert isinstance(fine_tune, FineTuneTrainingProfile)
    assert fine_tune.steps == 2
    assert fine_tune.optimizer.name == "adamw"
    assert lora.toolchain.python == "3.12.13"
    assert lora.toolchain.torch == "2.11.0"
    assert lora.toolchain.peft == "0.19.1"
    assert fine_tune.toolchain.transformers == "5.12.0"
    assert fine_tune.toolchain.peft is None
    assert lora.model_load.loss_function == "ForCausalLM"
    assert lora.model_load.expected_unexpected_keys == (
        "transformer.h.0.attn.masked_bias",
        "transformer.h.1.attn.masked_bias",
    )
    assert cuda_lora.device == "cuda"
    assert cuda_fine_tune.device == "cuda"
    assert lora.training_data.sha256 == file_sha256(
        lora.training_data.resolve(REPO_ROOT)
    )
    assert fine_tune.training_data.sha256 == lora.training_data.sha256


def test_profile_loader_rejects_unknown_id_and_edit_mismatch() -> None:
    with pytest.raises(TrainingProfileError, match="unknown training profile"):
        load_training_profile("missing")
    with pytest.raises(TrainingProfileError, match="edit_type mismatch"):
        load_training_profile("tiny_gpt2_lora_v1", expected_edit_type="fine_tune")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(edit_type="synthetic_lowrank_delta"), "edit_type"),
        (lambda value: value.update(steps=0), "steps must be a positive integer"),
        (
            lambda value: value["optimizer"].update(learning_rate=0.0),
            "learning_rate must be finite and positive",
        ),
        (
            lambda value: value["optimizer"].update(name="unknown"),
            "optimizer.name",
        ),
        (
            lambda value: value["lora"].update(rank=0),
            "lora.rank must be a positive integer",
        ),
        (
            lambda value: value["lora"].update(target_modules=[]),
            "lora.target_modules",
        ),
        (
            lambda value: value["lora"].update(bias="all"),
            "bias must be none",
        ),
        (
            lambda value: value.update(model_revision="main"),
            "model_revision must be a pinned",
        ),
        (
            lambda value: value.update(deterministic_algorithms=False),
            "deterministic_algorithms must be true",
        ),
        (
            lambda value: value["toolchain"].update(torch="latest"),
            "toolchain.torch must be an exact",
        ),
        (
            lambda value: value["model_load"].update(loss_function="fallback"),
            "model_load.loss_function",
        ),
        (
            lambda value: value["model_load"].update(
                expected_unexpected_keys=["z", "a"]
            ),
            "sorted and unique",
        ),
        (
            lambda value: value.update(extra_field=True),
            "unsupported field",
        ),
    ],
)
def test_lora_profile_semantic_tamper_fails_even_with_recomputed_digest(
    mutate,
    message: str,
) -> None:
    profile = copy.deepcopy(_raw_profiles()["tiny_gpt2_lora_v1"])
    mutate(profile)
    _rebind(profile)

    errors = training_profile_errors("tiny_gpt2_lora_v1", profile, repo_root=REPO_ROOT)

    assert any(message in error for error in errors), errors


def test_profile_digest_tamper_is_detected() -> None:
    profile = copy.deepcopy(_raw_profiles()["tiny_gpt2_full_ft_v1"])
    profile["profile_sha256"] = "sha256:" + "0" * 64

    errors = training_profile_errors(
        "tiny_gpt2_full_ft_v1", profile, repo_root=REPO_ROOT
    )

    assert any("does not match canonical profile content" in error for error in errors)


def test_fine_tune_profile_rejects_lora_configuration() -> None:
    profiles = _raw_profiles()
    profile = copy.deepcopy(profiles["tiny_gpt2_full_ft_v1"])
    profile["lora"] = copy.deepcopy(profiles["tiny_gpt2_lora_v1"]["lora"])
    _rebind(profile)

    errors = training_profile_errors(
        "tiny_gpt2_full_ft_v1", profile, repo_root=REPO_ROOT
    )

    assert any("must not contain a lora configuration" in error for error in errors)


def test_vendored_training_data_digest_and_rows_are_recomputed(tmp_path: Path) -> None:
    data_path = tmp_path / "training.jsonl"
    data_path.write_text('{"text":"one"}\n{"text":"two"}\n', encoding="utf-8")
    profile = copy.deepcopy(_raw_profiles()["tiny_gpt2_full_ft_v1"])
    profile["training_data"] = {
        "path": "training.jsonl",
        "sha256": file_sha256(data_path),
        "rows": 2,
        "text_field": "text",
    }
    _rebind(profile)
    assert training_profile_errors("local", profile, repo_root=tmp_path) == []

    data_path.write_text('{"text":"tampered"}\n', encoding="utf-8")
    errors = training_profile_errors("local", profile, repo_root=tmp_path)

    assert any("sha256 does not match" in error for error in errors)
    assert any("rows=2 does not match observed rows=1" in error for error in errors)


def test_training_data_path_cannot_escape_repository() -> None:
    profile = copy.deepcopy(_raw_profiles()["tiny_gpt2_full_ft_v1"])
    profile["training_data"]["path"] = "../private.jsonl"
    _rebind(profile)

    errors = training_profile_errors(
        "tiny_gpt2_full_ft_v1", profile, repo_root=REPO_ROOT
    )

    assert any("repository-relative" in error for error in errors)
