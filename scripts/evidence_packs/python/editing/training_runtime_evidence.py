"""State and artifact evidence helpers for the real-training runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from invarlock.peft_runtime import PeftRuntimeError
from invarlock.peft_runtime import peft_base_state as _package_peft_base_state
from invarlock.peft_runtime import (
    peft_merge_target_names as _package_peft_merge_target_names,
)
from invarlock.training_state_evidence import TrainingStateEvidenceError
from invarlock.training_state_evidence import (
    directory_sha256 as _package_directory_sha256,
)
from invarlock.training_state_evidence import (
    full_delta_evidence as _package_full_delta_evidence,
)
from invarlock.training_state_evidence import (
    require_state_manifest as _package_require_state_manifest,
)
from invarlock.training_state_evidence import state_manifest as _package_state_manifest
from invarlock.training_state_evidence import (
    streaming_lora_delta_evidence as _package_streaming_lora_delta_evidence,
)
from invarlock.training_state_evidence import (
    tensor_state_sha256 as _package_tensor_state_sha256,
)

from .training_runtime_errors import TrainingRuntimeError
from .training_runtime_validation import require_fixture_sized_model

_MAX_MODEL_PARAMETERS = 100_000_000


def _tensor_bytes(tensor: Any, torch: Any) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous().reshape(-1)
    try:
        return bytes(value.numpy().tobytes(order="C"))
    except TypeError:
        return bytes(value.view(torch.uint8).numpy().tobytes(order="C"))


def tensor_state_sha256(state: Mapping[str, Any], *, torch: Any) -> str:
    """Hash a tensor mapping by sorted name, dtype, shape, and raw bytes."""

    try:
        return cast(str, _package_tensor_state_sha256(state, torch=torch))
    except TrainingStateEvidenceError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def directory_sha256(path: Path, *, exclude: frozenset[str] = frozenset()) -> str:
    """Hash a directory tree by relative POSIX path and file contents."""

    try:
        return cast(str, _package_directory_sha256(path, exclude=exclude))
    except TrainingStateEvidenceError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _snapshot(model: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _state_manifest(
    state: Mapping[str, Any], *, torch: Any
) -> dict[str, dict[str, Any]]:
    """Record exact tensor identities without retaining duplicate values."""

    try:
        return cast(
            dict[str, dict[str, Any]], _package_state_manifest(state, torch=torch)
        )
    except TrainingStateEvidenceError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _peft_base_state(model: Any) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], _package_peft_base_state(model))
    except PeftRuntimeError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _require_state_manifest(
    state: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    torch: Any,
    label: str,
) -> str:
    try:
        return cast(
            str,
            _package_require_state_manifest(state, expected, torch=torch, label=label),
        )
    except TrainingStateEvidenceError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _peft_merge_target_names(
    model: Any,
    baseline_state: Mapping[str, Any],
) -> frozenset[str]:
    try:
        return cast(
            frozenset[str], _package_peft_merge_target_names(model, baseline_state)
        )
    except PeftRuntimeError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _streaming_lora_delta_evidence(
    *,
    baseline_manifest: Mapping[str, Mapping[str, Any]],
    baseline_targets: Mapping[str, Any],
    after: Mapping[str, Any],
    torch: Any,
) -> tuple[str, int, float, set[str]]:
    try:
        return cast(
            tuple[str, int, float, set[str]],
            _package_streaming_lora_delta_evidence(
                baseline_manifest=baseline_manifest,
                baseline_targets=baseline_targets,
                after=after,
                torch=torch,
            ),
        )
    except TrainingStateEvidenceError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _delta_evidence(
    before: Mapping[str, Any], after: Mapping[str, Any], *, torch: Any
) -> tuple[str, int, float, set[str]]:
    try:
        return cast(
            tuple[str, int, float, set[str]],
            _package_full_delta_evidence(before, after, torch=torch),
        )
    except TrainingStateEvidenceError as exc:
        raise TrainingRuntimeError(str(exc)) from exc


def _require_fixture_sized_model(model: Any) -> int:
    return cast(
        int,
        require_fixture_sized_model(
            model,
            max_parameters=_MAX_MODEL_PARAMETERS,
            error_type=TrainingRuntimeError,
        ),
    )


__all__ = [
    "_delta_evidence",
    "_peft_base_state",
    "_peft_merge_target_names",
    "_require_fixture_sized_model",
    "_require_state_manifest",
    "_snapshot",
    "_state_manifest",
    "_streaming_lora_delta_evidence",
    "_tensor_bytes",
    "directory_sha256",
    "tensor_state_sha256",
]
