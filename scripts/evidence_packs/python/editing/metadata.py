from __future__ import annotations

import json
from pathlib import Path

EDIT_METADATA_SCHEMA = "invarlock/evidence-pack-edit-metadata-v1"

VALIDATION_SUBJECT_CHECKPOINT = "validation_subject_checkpoint"
DEPLOYABLE_OPTIMIZED_SUBJECT = "deployable_optimized_subject"
FAULT_INJECTION_FIXTURE = "fault_injection_fixture"
EVIDENCE_ONLY_PACK = "evidence_only_pack"

ALLOWED_ARTIFACT_CLASSES = {
    VALIDATION_SUBJECT_CHECKPOINT,
    DEPLOYABLE_OPTIMIZED_SUBJECT,
    FAULT_INJECTION_FIXTURE,
    EVIDENCE_ONLY_PACK,
}

VALIDATION_STORAGE_FORMATS = {
    "quant_rtn": "float_dequantized",
    "fp8_quant": "float_dequantized",
    "magnitude_prune": "dense_float_with_zeros",
    "lowrank_svd": "dense_float_lowrank_approximated",
}

EDIT_SEMANTICS_EXTERNAL_SUBJECT = "external_subject_validation_edit"
EDIT_SEMANTICS_DEPLOYABLE = "backend_deployable_edit"


def _as_nonnegative_int(value: object, *, default: int = 0) -> int:
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return max(coerced, 0)


def _as_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default


def normalize_coverage(coverage: dict[str, object] | None) -> dict[str, object]:
    coverage = coverage if isinstance(coverage, dict) else {}
    edited_tensors = _as_nonnegative_int(
        coverage.get("edited_tensors", coverage.get("edited_count", 0))
    )
    edited_params = _as_nonnegative_int(coverage.get("edited_params", 0))
    total_params = _as_nonnegative_int(coverage.get("total_params", 0))
    ratio = _as_float(coverage.get("coverage_ratio", 0.0))
    if total_params > 0:
        ratio = edited_params / total_params
    ratio = max(0.0, min(ratio, 1.0))
    return {
        "edited_tensors": edited_tensors,
        "edited_params": edited_params,
        "total_params": total_params,
        "coverage_ratio": ratio,
    }


def storage_format_for_edit(edit_type: str) -> str:
    return VALIDATION_STORAGE_FORMATS.get(edit_type, "float_dequantized")


def build_edit_metadata(
    *,
    edit_type: str,
    scope: str,
    parameters: dict[str, object] | None = None,
    coverage: dict[str, object] | None = None,
    artifact_class: str = VALIDATION_SUBJECT_CHECKPOINT,
    edit_semantics: str = EDIT_SEMANTICS_EXTERNAL_SUBJECT,
    deployable_as_hf_checkpoint: bool = True,
    optimized_deployment_backend: bool = False,
    backend: str | None = None,
    storage_format: str | None = None,
    actual_storage_format: str | None = None,
    packed_quantized_storage: bool = False,
    runtime_memory_reduction: bool = False,
    runtime_memory_reduction_expected: bool | None = None,
    run_directory_contains_edit_artifacts: bool = True,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    resolved_storage = storage_format or storage_format_for_edit(edit_type)
    metadata: dict[str, object] = {
        "schema": EDIT_METADATA_SCHEMA,
        "artifact_class": artifact_class,
        "edit_type": edit_type,
        "edit_semantics": edit_semantics,
        "deployable_as_hf_checkpoint": bool(deployable_as_hf_checkpoint),
        "optimized_deployment_backend": bool(optimized_deployment_backend),
        "backend": backend,
        "storage_format": resolved_storage,
        "actual_storage_format": actual_storage_format or resolved_storage,
        "packed_quantized_storage": bool(packed_quantized_storage),
        "runtime_memory_reduction": bool(runtime_memory_reduction),
        "runtime_memory_reduction_expected": (
            bool(runtime_memory_reduction_expected)
            if runtime_memory_reduction_expected is not None
            else bool(runtime_memory_reduction)
        ),
        "run_directory_contains_edit_artifacts": bool(
            run_directory_contains_edit_artifacts
        ),
        "scope": scope,
        "parameters": dict(parameters or {}),
        "coverage": normalize_coverage(coverage),
    }
    if extra:
        metadata.update(extra)
    return metadata


def build_validation_edit_metadata(
    *,
    edit_type: str,
    scope: str,
    parameters: dict[str, object] | None = None,
    coverage: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_edit_metadata(
        edit_type=edit_type,
        scope=scope,
        parameters=parameters,
        coverage=coverage,
        artifact_class=VALIDATION_SUBJECT_CHECKPOINT,
        edit_semantics=EDIT_SEMANTICS_EXTERNAL_SUBJECT,
        deployable_as_hf_checkpoint=True,
        optimized_deployment_backend=False,
        backend=None,
        storage_format=storage_format_for_edit(edit_type),
        actual_storage_format=storage_format_for_edit(edit_type),
        packed_quantized_storage=False,
        runtime_memory_reduction=False,
        runtime_memory_reduction_expected=False,
        extra=extra,
    )


def write_edit_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def read_edit_metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("edit metadata must be a JSON object")
    return payload


def validate_edit_metadata(
    metadata: dict[str, object],
    *,
    expected_edit_type: str | None = None,
    expected_artifact_class: str | None = None,
) -> list[str]:
    errors: list[str] = []

    schema = metadata.get("schema")
    if schema != EDIT_METADATA_SCHEMA:
        errors.append(f"unknown edit metadata schema: {schema!r}")

    artifact_class = metadata.get("artifact_class")
    if artifact_class not in ALLOWED_ARTIFACT_CLASSES:
        errors.append(f"invalid artifact_class: {artifact_class!r}")
    if expected_artifact_class and artifact_class != expected_artifact_class:
        errors.append(
            f"artifact_class mismatch: expected {expected_artifact_class!r}, "
            f"got {artifact_class!r}"
        )

    edit_type = metadata.get("edit_type")
    if not isinstance(edit_type, str) or not edit_type:
        errors.append("edit_type must be a non-empty string")
    if expected_edit_type and edit_type != expected_edit_type:
        errors.append(
            f"edit_type mismatch: expected {expected_edit_type!r}, got {edit_type!r}"
        )

    coverage = metadata.get("coverage")
    if not isinstance(coverage, dict):
        errors.append("coverage must be an object")
    else:
        for field in (
            "edited_tensors",
            "edited_params",
            "total_params",
            "coverage_ratio",
        ):
            if field not in coverage:
                errors.append(f"coverage.{field} missing")

    optimized = metadata.get("optimized_deployment_backend")
    packed = metadata.get("packed_quantized_storage")
    runtime_reduction = metadata.get("runtime_memory_reduction")
    backend = metadata.get("backend")

    if artifact_class == VALIDATION_SUBJECT_CHECKPOINT:
        if optimized is not False:
            errors.append(
                "validation artifacts must set optimized_deployment_backend=false"
            )
        if packed is not False:
            errors.append(
                "validation artifacts must set packed_quantized_storage=false"
            )
        if runtime_reduction is not False:
            errors.append(
                "validation artifacts must set runtime_memory_reduction=false"
            )
        if backend is not None:
            errors.append("validation artifacts must set backend=null")
        expected_storage = storage_format_for_edit(str(edit_type or ""))
        if metadata.get("storage_format") != expected_storage:
            errors.append(
                f"storage_format mismatch for {edit_type!r}: expected {expected_storage!r}"
            )
        if metadata.get("actual_storage_format") != expected_storage:
            errors.append(
                f"actual_storage_format mismatch for {edit_type!r}: expected {expected_storage!r}"
            )

    if artifact_class == DEPLOYABLE_OPTIMIZED_SUBJECT:
        if optimized is not True:
            errors.append(
                "deployable artifacts must set optimized_deployment_backend=true"
            )
        if packed is not True:
            errors.append("deployable artifacts must set packed_quantized_storage=true")
        if not isinstance(backend, str) or not backend:
            errors.append("deployable artifacts must record a backend")

    if metadata.get("deployable_as_hf_checkpoint") is not True and artifact_class in {
        VALIDATION_SUBJECT_CHECKPOINT,
        DEPLOYABLE_OPTIMIZED_SUBJECT,
    }:
        errors.append(
            "subject checkpoint artifacts must set deployable_as_hf_checkpoint=true"
        )

    return errors


__all__ = [
    "ALLOWED_ARTIFACT_CLASSES",
    "DEPLOYABLE_OPTIMIZED_SUBJECT",
    "EDIT_METADATA_SCHEMA",
    "EVIDENCE_ONLY_PACK",
    "FAULT_INJECTION_FIXTURE",
    "VALIDATION_STORAGE_FORMATS",
    "VALIDATION_SUBJECT_CHECKPOINT",
    "build_edit_metadata",
    "build_validation_edit_metadata",
    "read_edit_metadata",
    "storage_format_for_edit",
    "validate_edit_metadata",
    "write_edit_metadata",
]
