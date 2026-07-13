"""Input and resource-bound validation for training evidence runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .training_contract import training_profile_errors


def profile_mapping(profile: Any) -> dict[str, Any]:
    value = asdict(profile)
    value.pop("profile_id")
    value["optimizer"]["betas"] = list(value["optimizer"]["betas"])
    if value["toolchain"].get("peft") is None:
        value["toolchain"].pop("peft")
    value["model_load"]["expected_unexpected_keys"] = list(
        value["model_load"]["expected_unexpected_keys"]
    )
    if "lora" in value:
        value["lora"]["target_modules"] = list(value["lora"]["target_modules"])
    return value


def validate_profile(
    profile_id: str,
    profile: Mapping[str, Any],
    *,
    edit_type: str,
    repo_root: Path,
    error_type: type[Exception],
) -> None:
    errors = training_profile_errors(
        profile_id,
        profile,
        expected_edit_type=edit_type,
        repo_root=repo_root,
        verify_data_file=True,
    )
    if errors:
        raise error_type("; ".join(errors))


def require_fixture_sized_model(
    model: Any,
    *,
    max_parameters: int,
    error_type: type[Exception],
) -> int:
    count = sum(int(parameter.numel()) for parameter in model.parameters())
    if count > max_parameters:
        raise error_type(
            "the v1 training runtime is limited to tiny evidence fixtures "
            f"(maximum {max_parameters:,} parameters; observed {count:,})"
        )
    return count
