from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.evidence_packs.python.editing import training_contract as contract
from scripts.evidence_packs.python.editing.training_contract import (
    DEFAULT_TRAINING_PROFILES_PATH,
    TRAINING_PROFILES_SCHEMA,
    TrainingProfileError,
    canonical_profile_digest,
    file_sha256,
    load_training_profile,
    lora_config_digest,
    training_profile_errors,
)


def _raw_profile(profile_id: str = "tiny_gpt2_lora_v1") -> dict[str, Any]:
    payload = json.loads(DEFAULT_TRAINING_PROFILES_PATH.read_text(encoding="utf-8"))
    return copy.deepcopy(payload["profiles"][profile_id])


def _rebind(profile: dict[str, Any]) -> dict[str, Any]:
    profile["profile_sha256"] = canonical_profile_digest(profile)
    return profile


def test_profile_contract_reports_all_malformed_scalar_and_nested_inputs() -> None:
    profile = _raw_profile()
    profile.update(
        {
            "profile_sha256": "bad",
            "model_id": "",
            "model_revision": "main",
            "training_data": None,
            "optimizer": None,
            "steps": True,
            "micro_batch_size": 0,
            "gradient_accumulation_steps": -1,
            "max_sequence_length": "8",
            "seed": -1,
            "deterministic_algorithms": False,
            "device": "tpu",
            "dtype": "float8",
            "toolchain": None,
            "lora": None,
        }
    )

    errors = training_profile_errors("fixture", profile, verify_data_file=False)

    expected = (
        "profile_sha256 must be a canonical",
        "model_id must be a non-empty",
        "model_revision must be a pinned",
        "training_data must be an object",
        "optimizer must be an object",
        "steps must be a positive",
        "seed must be a non-negative",
        "deterministic_algorithms must be true",
        "device must be one of",
        "dtype must be one of",
        "toolchain must be an object",
        "lora must be an object",
    )
    assert all(any(fragment in error for error in errors) for fragment in expected)
    assert training_profile_errors("", {}) == ["profile_id must be a non-empty string"]
    assert "profile must be an object" in training_profile_errors("fixture", None)[0]
    assert lora_config_digest({"rank": 2}) == contract.canonical_sha256({"rank": 2})

    missing_path = _raw_profile("tiny_gpt2_full_ft_v1")
    missing_path["training_data"]["path"] = ""
    missing_path["profile_sha256"] = "bad"
    assert any(
        "path must be a non-empty" in error
        for error in training_profile_errors(
            "fixture", missing_path, verify_data_file=False
        )
    )


def test_profile_contract_rejects_semantically_invalid_optimizer_lora_and_toolchain() -> (
    None
):
    profile = _raw_profile()
    profile["optimizer"] = {
        "name": "sgd",
        "learning_rate": True,
        "betas": [0.9, 1.0],
        "eps": True,
        "weight_decay": -1.0,
        "extra": True,
    }
    profile["lora"] = {
        "rank": True,
        "alpha": 0,
        "dropout": 1.0,
        "target_modules": ["c_attn", "c_attn"],
        "bias": "all",
        "task_type": "SEQ_CLS",
        "fan_in_fan_out": "false",
        "extra": True,
    }
    profile["toolchain"] = {
        "python": "latest",
        "torch": 2,
        "transformers": "5.12",
        "peft": "0.19",
        "extra": "x",
    }
    _rebind(profile)

    errors = training_profile_errors("fixture", profile, verify_data_file=False)

    expected = (
        "optimizer contains unsupported",
        "optimizer.name",
        "optimizer.learning_rate",
        "optimizer.betas values",
        "optimizer.eps",
        "optimizer.weight_decay",
        "lora contains unsupported",
        "lora.rank",
        "lora.alpha",
        "lora.dropout",
        "lora.target_modules must be unique",
        "lora.bias",
        "lora.task_type",
        "lora.fan_in_fan_out",
        "toolchain contains unsupported",
        "toolchain.python",
        "toolchain.torch",
        "toolchain.transformers",
        "toolchain.peft",
    )
    assert all(any(fragment in error for error in errors) for fragment in expected)

    profile = _raw_profile()
    profile["optimizer"]["betas"] = [0.9]
    profile["lora"]["target_modules"] = [""]
    _rebind(profile)
    errors = training_profile_errors("fixture", profile, verify_data_file=False)
    assert any("betas must contain exactly two" in error for error in errors)
    assert any("target_modules must be a non-empty" in error for error in errors)

    assert "optimizer.eps must be finite and positive" in contract._validate_optimizer(
        {
            "name": "adamw",
            "learning_rate": 0.1,
            "betas": [0.9, 0.99],
            "eps": float("inf"),
            "weight_decay": 0.0,
        }
    )
    assert "lora.dropout must be finite in [0, 1)" in contract._validate_lora(
        {
            "rank": 1,
            "alpha": 1,
            "dropout": float("nan"),
            "target_modules": ["c_attn"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "fan_in_fan_out": False,
        }
    )


def test_fine_tune_toolchain_rejects_peft_even_when_version_is_exact() -> None:
    profile = _raw_profile("tiny_gpt2_full_ft_v1")
    profile["toolchain"]["peft"] = "0.19.1"
    _rebind(profile)

    errors = training_profile_errors("fixture", profile, verify_data_file=False)

    assert any("peft is only valid for lora_merge" in error for error in errors)


def test_training_data_validation_rejects_bad_rows_content_and_escaped_symlink(
    tmp_path: Path,
) -> None:
    profile = _raw_profile("tiny_gpt2_full_ft_v1")
    data = tmp_path / "training.jsonl"
    data.write_text('\n[]\n{"wrong":"field"}\n{\n', encoding="utf-8")
    profile["training_data"] = {
        "path": data.name,
        "sha256": file_sha256(data),
        "rows": 8,
        "text_field": "text",
        "extra": True,
    }
    _rebind(profile)

    errors = training_profile_errors("fixture", profile, repo_root=tmp_path)

    assert any("contains unsupported" in error for error in errors)
    assert any("blank row at line 1" in error for error in errors)
    assert any("line 2 must be a JSON object" in error for error in errors)
    assert any("line 3 lacks non-empty 'text'" in error for error in errors)
    assert any("is not valid UTF-8 JSONL" in error for error in errors)
    assert any("rows=8 does not match observed" in error for error in errors)

    missing = _raw_profile("tiny_gpt2_full_ft_v1")
    missing["training_data"].update(
        {"path": "missing.jsonl", "sha256": "bad", "rows": True, "text_field": ""}
    )
    _rebind(missing)
    missing_errors = training_profile_errors("fixture", missing, repo_root=tmp_path)
    assert any("path does not exist" in error for error in missing_errors)
    assert any("sha256 must be" in error for error in missing_errors)
    assert any("rows must be" in error for error in missing_errors)
    assert any("text_field must be" in error for error in missing_errors)

    outside = tmp_path.parent / "outside-training.jsonl"
    outside.write_text('{"text":"outside"}\n', encoding="utf-8")
    link = tmp_path / "linked.jsonl"
    link.symlink_to(outside)
    escaped = _raw_profile("tiny_gpt2_full_ft_v1")
    escaped["training_data"].update(
        {"path": link.name, "sha256": file_sha256(outside), "rows": 1}
    )
    _rebind(escaped)
    escaped_errors = training_profile_errors("fixture", escaped, repo_root=tmp_path)
    assert any("resolves outside the repository" in error for error in escaped_errors)


def test_training_data_rejects_duplicate_jsonl_fields_from_one_snapshot(
    tmp_path: Path,
) -> None:
    profile = _raw_profile("tiny_gpt2_full_ft_v1")
    data = tmp_path / "training.jsonl"
    data.write_text('{"text":"first","text":"second"}\n', encoding="utf-8")
    profile["training_data"] = {
        "path": data.name,
        "sha256": file_sha256(data),
        "rows": 1,
        "text_field": "text",
    }
    _rebind(profile)

    errors = training_profile_errors("fixture", profile, repo_root=tmp_path)

    assert any("not valid UTF-8 JSONL" in error for error in errors)
    assert any("duplicate key" in error for error in errors)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "unable to load training profiles"),
        (
            '{"schema":"first","schema":"second","profiles":{}}',
            "unable to load training profiles",
        ),
        ("[]", "document must be an object"),
        (
            json.dumps(
                {
                    "schema": "invarlock/evidence-pack-training-profiles-v2",
                    "profiles": {},
                }
            ),
            "unknown schema",
        ),
        (json.dumps({"schema": "wrong", "profiles": {}}), "unknown schema"),
        (
            json.dumps(
                {"schema": TRAINING_PROFILES_SCHEMA, "profiles": {}, "extra": True}
            ),
            "contains unknown fields",
        ),
        (
            json.dumps({"schema": TRAINING_PROFILES_SCHEMA, "profiles": []}),
            "has no profiles",
        ),
    ],
)
def test_profile_loader_fails_closed_on_malformed_documents(
    payload: str, message: str, tmp_path: Path
) -> None:
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(payload, encoding="utf-8")

    with pytest.raises(TrainingProfileError, match=message):
        load_training_profile(
            "fixture", profiles_path=profiles_path, repo_root=tmp_path
        )
