"""Specialized backend and lifecycle summary validation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from scripts.checks.public_evidence_checks.artifacts import _require_path
from scripts.checks.public_evidence_checks.common import (
    REPO_ROOT,
    _load_json,
    _relative,
)

RUNTIME_BACKEND_COMPATIBILITY_SUMMARY_SCHEMA = (
    "invarlock.runtime_backend_compatibility.cuda128.summary.v1"
)
RUNTIME_BACKEND_COMPATIBILITY_HASH_SCHEMA = (
    "invarlock.runtime_backend_compatibility.cuda128.hash_inventory.v1"
)
ATTENTION_BACKEND_SUMMARY_SCHEMA = (
    "invarlock.attention_backend_compatibility.summary.v1"
)
ATTENTION_BACKEND_HASH_SCHEMA = (
    "invarlock.attention_backend_compatibility.hash_inventory.v1"
)
RUNTIME_BACKEND_FAMILIES = {
    "cuda-bnb": ("hf_bnb",),
    "cuda-compressed-tensors": ("hf_ct",),
    "cuda-gptqmodel": ("hf_awq", "hf_gptq"),
    "cuda-hqq": ("hf_hqq",),
    "cuda-quanto": ("hf_quanto",),
    "cuda-torchao": ("hf_torchao",),
}
RUNTIME_BACKEND_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _check_runtime_backend_hash_inventory(
    errors: list[str],
    base: Path,
    inventory_path: Path,
) -> None:
    inventory, error = _load_json(inventory_path)
    if error:
        errors.append(error)
        return
    assert inventory is not None
    if inventory.get("schema") != RUNTIME_BACKEND_COMPATIBILITY_HASH_SCHEMA:
        errors.append(
            f"{_relative(inventory_path)}: schema must be {RUNTIME_BACKEND_COMPATIBILITY_HASH_SCHEMA}"
        )
    if inventory.get("status") != "completed":
        errors.append(f"{_relative(inventory_path)}: status must be completed")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{_relative(inventory_path)}: artifacts must be non-empty")
        return
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}] must be object"
            )
            continue
        rel_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path required"
            )
            continue
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path must be relative"
            )
            continue
        path = base / rel_path
        if not path.is_file():
            errors.append(
                f"{_relative(base)}: hash inventory file missing {rel_path!r}"
            )
            continue
        content = path.read_bytes()
        actual_sha = "sha256:" + hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"{_relative(base)}: hash inventory mismatch for {rel_path!r}"
            )
        if len(content) != expected_bytes:
            errors.append(
                f"{_relative(base)}: hash inventory byte mismatch for {rel_path!r}"
            )


def _check_runtime_backend_compatibility(
    errors: list[str],
    base: Path,
    artifact_paths: dict[str, Any],
) -> None:
    summary_path = _require_path(errors, base, artifact_paths, "compatibility_summary")
    inventory_path = _require_path(errors, base, artifact_paths, "hash_inventory")
    if inventory_path is not None:
        _check_runtime_backend_hash_inventory(errors, base, inventory_path)
    if summary_path is None:
        return
    summary, error = _load_json(summary_path)
    if error:
        errors.append(error)
        return
    assert summary is not None
    if summary.get("schema") != RUNTIME_BACKEND_COMPATIBILITY_SUMMARY_SCHEMA:
        errors.append(
            f"{_relative(summary_path)}: schema must be {RUNTIME_BACKEND_COMPATIBILITY_SUMMARY_SCHEMA}"
        )
    if summary.get("status") != "completed":
        errors.append(f"{_relative(summary_path)}: status must be completed")
    if summary.get("validation_environment") != "CUDA-capable validation host":
        errors.append(
            f"{_relative(summary_path)}: validation_environment must be generic"
        )
    if summary.get("raw_logs_published") is not False:
        errors.append(f"{_relative(summary_path)}: raw_logs_published must be false")
    if summary.get("weights_vendored") is not False:
        errors.append(f"{_relative(summary_path)}: weights_vendored must be false")

    families = summary.get("families")
    if not isinstance(families, list) or not families:
        errors.append(f"{_relative(summary_path)}: families must be non-empty")
        return
    observed: set[str] = set()
    for index, family in enumerate(families):
        if not isinstance(family, dict):
            errors.append(
                f"{_relative(summary_path)}: families[{index}] must be object"
            )
            continue
        family_name = family.get("family")
        if family_name not in RUNTIME_BACKEND_FAMILIES:
            errors.append(f"{_relative(summary_path)}: unknown family {family_name!r}")
            continue
        observed.add(str(family_name))
        expected_adapters = list(RUNTIME_BACKEND_FAMILIES[str(family_name)])
        if family.get("adapter_smoke") != expected_adapters:
            errors.append(
                f"{_relative(summary_path)}: {family_name} adapter_smoke mismatch"
            )
        if family.get("build_rc") != 0 or family.get("smoke_rc") != 0:
            errors.append(f"{_relative(summary_path)}: {family_name} rc must be zero")
        if family.get("gpu_required") is not True:
            errors.append(f"{_relative(summary_path)}: {family_name} must require GPU")
        requirements_lock = family.get("requirements_lock")
        if not isinstance(requirements_lock, str) or not requirements_lock:
            errors.append(
                f"{_relative(summary_path)}: {family_name} requirements_lock required"
            )
        elif (
            requirements_lock.startswith("/")
            or not (REPO_ROOT / requirements_lock).is_file()
        ):
            errors.append(
                f"{_relative(summary_path)}: {family_name} requirements_lock invalid"
            )
        for command_key in ("build_command", "smoke_command"):
            command = family.get(command_key)
            if not isinstance(command, str) or command.startswith("/"):
                errors.append(
                    f"{_relative(summary_path)}: {family_name} {command_key} invalid"
                )
        image_id = family.get("image_id")
        if not isinstance(image_id, str) or not RUNTIME_BACKEND_IMAGE_ID_RE.match(
            image_id
        ):
            errors.append(f"{_relative(summary_path)}: {family_name} image_id invalid")
        image_size = family.get("image_size_bytes")
        if not isinstance(image_size, int) or image_size <= 0:
            errors.append(
                f"{_relative(summary_path)}: {family_name} image_size_bytes invalid"
            )
        smoke_result = family.get("smoke_result")
        if not isinstance(smoke_result, str) or not smoke_result.startswith(
            "quant runtime image imports ok:"
        ):
            errors.append(
                f"{_relative(summary_path)}: {family_name} smoke_result invalid"
            )
    if observed != set(RUNTIME_BACKEND_FAMILIES):
        missing = sorted(set(RUNTIME_BACKEND_FAMILIES) - observed)
        extra = sorted(observed - set(RUNTIME_BACKEND_FAMILIES))
        errors.append(
            f"{_relative(summary_path)}: family coverage mismatch "
            f"missing={missing} extra={extra}"
        )


def _check_attention_backend_hash_inventory(
    errors: list[str],
    base: Path,
    inventory_path: Path,
) -> None:
    inventory, error = _load_json(inventory_path)
    if error:
        errors.append(error)
        return
    assert inventory is not None
    if inventory.get("schema") != ATTENTION_BACKEND_HASH_SCHEMA:
        errors.append(
            f"{_relative(inventory_path)}: schema must be {ATTENTION_BACKEND_HASH_SCHEMA}"
        )
    if inventory.get("status") != "completed":
        errors.append(f"{_relative(inventory_path)}: status must be completed")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{_relative(inventory_path)}: artifacts must be non-empty")
        return
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}] must be object"
            )
            continue
        rel_path = artifact.get("path")
        expected_sha = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if not isinstance(rel_path, str) or not rel_path:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path required"
            )
            continue
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            errors.append(
                f"{_relative(inventory_path)}: artifacts[{index}].path must be relative"
            )
            continue
        path = base / rel_path
        if not path.is_file():
            errors.append(
                f"{_relative(base)}: hash inventory file missing {rel_path!r}"
            )
            continue
        content = path.read_bytes()
        actual_sha = "sha256:" + hashlib.sha256(content).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"{_relative(base)}: hash inventory mismatch for {rel_path!r}"
            )
        if len(content) != expected_bytes:
            errors.append(
                f"{_relative(base)}: hash inventory byte mismatch for {rel_path!r}"
            )


def _check_attention_backend_compatibility(
    errors: list[str],
    base: Path,
    artifact_paths: dict[str, Any],
) -> None:
    summary_path = _require_path(errors, base, artifact_paths, "compatibility_summary")
    inventory_path = _require_path(errors, base, artifact_paths, "hash_inventory")
    if inventory_path is not None:
        _check_attention_backend_hash_inventory(errors, base, inventory_path)
    if summary_path is None:
        return
    summary, error = _load_json(summary_path)
    if error:
        errors.append(error)
        return
    assert summary is not None
    if summary.get("schema") != ATTENTION_BACKEND_SUMMARY_SCHEMA:
        errors.append(
            f"{_relative(summary_path)}: schema must be {ATTENTION_BACKEND_SUMMARY_SCHEMA}"
        )
    if summary.get("status") != "completed":
        errors.append(f"{_relative(summary_path)}: status must be completed")
    if summary.get("validation_environment") != "CUDA-capable validation host":
        errors.append(
            f"{_relative(summary_path)}: validation_environment must be generic"
        )
    if summary.get("raw_logs_published") is not False:
        errors.append(f"{_relative(summary_path)}: raw_logs_published must be false")
    if summary.get("weights_vendored") is not False:
        errors.append(f"{_relative(summary_path)}: weights_vendored must be false")
    if summary.get("optimized_attention_success_claimed") is not False:
        errors.append(
            f"{_relative(summary_path)}: "
            "optimized_attention_success_claimed must be false"
        )

    probe = summary.get("cuda_probe")
    if not isinstance(probe, dict):
        errors.append(f"{_relative(summary_path)}: cuda_probe must be object")
    else:
        if probe.get("rc") != 0:
            errors.append(f"{_relative(summary_path)}: cuda_probe rc must be zero")
        if probe.get("torch_cuda_available") is not True:
            errors.append(f"{_relative(summary_path)}: CUDA must be available")
        device_count = probe.get("torch_cuda_device_count")
        if not isinstance(device_count, int) or device_count < 1:
            errors.append(f"{_relative(summary_path)}: CUDA device count invalid")
        if probe.get("flash_attn_importable") is not False:
            errors.append(
                f"{_relative(summary_path)}: flash_attn_importable must be false"
            )
        if probe.get("transformers_flash_attn_2_available") is not False:
            errors.append(
                f"{_relative(summary_path)}: transformers optimized attention availability must be false"
            )
        command = probe.get("command")
        if not isinstance(command, str) or command.startswith("/"):
            errors.append(f"{_relative(summary_path)}: cuda_probe command invalid")

    checks = summary.get("checks")
    if not isinstance(checks, list) or len(checks) < 2:
        errors.append(f"{_relative(summary_path)}: checks must include both checks")
        return
    observed = {check.get("name"): check for check in checks if isinstance(check, dict)}
    expected = {
        "flash_attention_dependency_paths": 3,
        "attention_config_selection": 1,
    }
    for name, expected_passed in expected.items():
        check = observed.get(name)
        if not isinstance(check, dict):
            errors.append(f"{_relative(summary_path)}: missing check {name}")
            continue
        if check.get("rc") != 0 or check.get("tests_failed") != 0:
            errors.append(f"{_relative(summary_path)}: {name} must pass cleanly")
        if check.get("tests_passed") != expected_passed:
            errors.append(
                f"{_relative(summary_path)}: {name} tests_passed must be {expected_passed}"
            )
        command = check.get("command")
        if not isinstance(command, str) or command.startswith("/"):
            errors.append(f"{_relative(summary_path)}: {name} command invalid")
        surface = check.get("coverage_surface")
        if not isinstance(surface, list) or not surface:
            errors.append(
                f"{_relative(summary_path)}: {name} coverage_surface required"
            )
