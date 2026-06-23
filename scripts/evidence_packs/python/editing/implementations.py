from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from . import tensor_ops as _tensor_ops
except ImportError:  # pragma: no cover - direct script-path loading in tests
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tensor_ops as _tensor_ops

_SCOPE_KEYWORDS = _tensor_ops._SCOPE_KEYWORDS
_EXCLUDED_PATH_SEGMENTS = _tensor_ops._EXCLUDED_PATH_SEGMENTS

_COERCE_ERRORS = (TypeError, ValueError, OverflowError)
_FILE_READ_ERRORS = (OSError, TypeError, ValueError)

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
EDIT_PROVENANCE_FAMILIES = {
    "custom",
    "deployable_backend_quantization",
    "dynamic_adapter",
    "fault_injection",
    "fine_tune",
    "knowledge_edit",
    "lora_merge",
    "lowrank_approximation",
    "magnitude_prune",
    "noop",
    "pruning",
    "quantization",
    "quantization_dequantized",
    "self_edit",
}
EDIT_IMPACT_SCENARIO_TYPES = {
    "target_success",
    "near_neighbor",
    "near_confuser",
    "unrelated_locality",
    "general_ability_sentinel",
    "multilingual_portability",
    "sequential_edit_stress",
}
_SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


@dataclass(frozen=True)
class ResolvedEditSpec:
    status: str
    edit_type: str
    param1: str = ""
    param2: str = ""
    scope: str = ""
    version: str = ""
    edit_dir_name: str = ""
    reason: str = ""

    @property
    def skip(self) -> bool:
        return self.status == "skipped"

    @property
    def selected(self) -> bool:
        return self.status == "selected"

    def to_shell_payload(self) -> dict[str, str]:
        return {
            "status": self.status,
            "reason": self.reason,
            "edit_type": self.edit_type,
            "param1": self.param1,
            "param2": self.param2,
            "scope": self.scope,
            "version": self.version,
            "edit_dir_name": self.edit_dir_name,
        }

    def to_batch_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": self.edit_type,
            "status": self.status,
            "reason": self.reason,
            "scope": self.scope,
            "edit_dir_name": self.edit_dir_name,
            "version": self.version,
        }
        if self.edit_type == "quant_rtn":
            payload["bits"] = (
                int(self.param1) if _safe_int(self.param1) is not None else 0
            )
            payload["group_size"] = (
                int(self.param2) if _safe_int(self.param2) is not None else 0
            )
        elif self.edit_type == "fp8_quant":
            payload["format"] = self.param1
        elif self.edit_type == "magnitude_prune":
            payload["ratio"] = (
                float(self.param1) if _safe_float(self.param1) is not None else 0.0
            )
        elif self.edit_type == "lowrank_svd":
            payload["rank"] = (
                int(self.param1) if _safe_int(self.param1) is not None else 0
            )
        return payload


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except _COERCE_ERRORS:
        return None


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except _COERCE_ERRORS:
        return None


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except _FILE_READ_ERRORS:
        return None
    return payload if isinstance(payload, dict) else None


def _model_id_for(model_output_dir: Path) -> str:
    model_id_path = model_output_dir / ".model_id"
    if not model_id_path.exists():
        return ""
    try:
        return model_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _load_tuned_entry(
    *,
    tuned_path: str,
    model_key: str,
    model_id: str,
    model_output_dir_name: str,
    edit_type: str,
) -> tuple[dict[str, Any], str, str]:
    if not tuned_path:
        return {}, "missing", "missing_tuned_edit_params_file"

    payload = _load_json_object(Path(tuned_path))
    if payload is None:
        if Path(tuned_path).exists():
            return {}, "invalid", "invalid_tuned_edit_params_file"
        return {}, "missing", "missing_tuned_edit_params_file"

    entry_map: dict[str, Any] = {}
    models = payload.get("models")
    if isinstance(models, dict):
        entry_map = (
            models.get(model_key)
            or models.get(model_id)
            or models.get(model_output_dir_name)
            or {}
        )

    if not entry_map and isinstance(payload.get(edit_type), dict):
        entry_map = payload

    defaults = payload.get("defaults")
    entry = (
        (entry_map.get(edit_type) if isinstance(entry_map, dict) else None)
        or (defaults.get(edit_type) if isinstance(defaults, dict) else None)
        or {}
    )
    if not isinstance(entry, dict):
        entry = {}

    return entry, str(entry.get("status") or "missing"), str(entry.get("reason") or "")


def _normalize_non_quant_scope(
    edit_type: str,
    param1: str,
    param2: str,
    scope: str,
) -> tuple[str, str]:
    if edit_type != "quant_rtn" and not scope:
        scope = param2
        param2 = ""
    return param2, scope


def _normalize_quant_scope(param1: str, param2: str, scope: str) -> tuple[str, str]:
    if not scope and param1 and param2:
        scope = param2
        param2 = ""
    return param2, scope


def _default_edit_dir_name(
    *,
    edit_type: str,
    param1: str,
    param2: str,
    version: str,
) -> str:
    if not version:
        return ""
    if edit_type == "quant_rtn":
        return f"quant_{param1}bit_{version}"
    if edit_type == "fp8_quant":
        return f"fp8_{param1}_{version}"
    if edit_type == "magnitude_prune":
        try:
            pct = int(float(param1) * 100)
        except _COERCE_ERRORS:
            pct = 0
        return f"prune_{pct}pct_{version}"
    if edit_type == "lowrank_svd":
        return f"svd_rank{param1}_{version}"
    return f"{edit_type}_{version}"


def resolve_edit_spec(
    *,
    model_output_dir: Path,
    edit_spec: str,
    version_hint: str = "",
    tuned_path: str | None = None,
) -> ResolvedEditSpec:
    parts = edit_spec.split(":") if edit_spec else []
    edit_type = parts[0] if parts else ""
    param1 = parts[1] if len(parts) > 1 else ""
    param2 = parts[2] if len(parts) > 2 else ""
    scope = parts[3] if len(parts) > 3 else ""

    param2, scope = _normalize_non_quant_scope(edit_type, param1, param2, scope)
    if edit_type == "quant_rtn":
        param2, scope = _normalize_quant_scope(param1, param2, scope)

    clean_spec = param1 == "clean"
    status = "selected"
    reason = ""
    edit_dir_name = ""

    if clean_spec:
        resolved_tuned_path = (
            tuned_path or os.environ.get("PACK_TUNED_EDIT_PARAMS_FILE") or ""
        ).strip()
        model_id = _model_id_for(model_output_dir)
        model_key = model_id or model_output_dir.name
        entry, status, reason = _load_tuned_entry(
            tuned_path=resolved_tuned_path,
            model_key=model_key,
            model_id=model_id,
            model_output_dir_name=model_output_dir.name,
            edit_type=edit_type,
        )
        if status == "selected":
            if edit_type == "quant_rtn":
                param1 = str(entry.get("bits", ""))
                param2 = str(entry.get("group_size", ""))
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "fp8_quant":
                param1 = str(entry.get("format", ""))
                param2 = ""
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "magnitude_prune":
                param1 = str(entry.get("sparsity", ""))
                param2 = ""
                scope = str(entry.get("scope") or scope or "")
            elif edit_type == "lowrank_svd":
                param1 = str(entry.get("rank", ""))
                param2 = ""
                scope = str(entry.get("scope") or scope or "")
            edit_dir_name = str(entry.get("edit_dir_name") or "")
    else:
        if edit_type == "quant_rtn":
            if _safe_int(param1) is None or _safe_int(param2) is None:
                status = "invalid"
                reason = "invalid_quant_params"
        elif edit_type == "magnitude_prune":
            if _safe_float(param1) is None:
                status = "invalid"
                reason = "invalid_prune_sparsity"
        elif edit_type == "lowrank_svd":
            if _safe_int(param1) is None:
                status = "invalid"
                reason = "invalid_lowrank_rank"
        elif edit_type == "fp8_quant":
            if not param1:
                status = "invalid"
                reason = "invalid_fp_format"

    version = version_hint or ("clean" if clean_spec else "")
    if status == "selected" and not edit_dir_name:
        edit_dir_name = _default_edit_dir_name(
            edit_type=edit_type,
            param1=param1,
            param2=param2,
            version=version,
        )

    return ResolvedEditSpec(
        status=status,
        reason=reason,
        edit_type=edit_type,
        param1=param1,
        param2=param2,
        scope=scope,
        version=version,
        edit_dir_name=edit_dir_name,
    )


def parse_edit_specs_json(raw_payload: str) -> list[object]:
    try:
        edit_specs = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid edit_specs JSON: {exc}") from exc

    if not isinstance(edit_specs, list):
        raise ValueError("edit_specs_json must be a JSON list")
    return edit_specs


def resolve_batch_entry(
    *,
    spec_entry: object,
    model_output_dir: Path,
    tuned_path: str | None = None,
) -> ResolvedEditSpec | None:
    if not isinstance(spec_entry, dict):
        return None
    spec_str = str(spec_entry.get("spec", ""))
    version = str(spec_entry.get("version", "clean"))
    return resolve_edit_spec(
        model_output_dir=model_output_dir,
        edit_spec=spec_str,
        version_hint=version,
        tuned_path=tuned_path,
    )


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
    edit_provenance: dict[str, object] | None = None,
    edit_impact: dict[str, object] | None = None,
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
    if edit_provenance is not None:
        metadata["edit_provenance"] = dict(edit_provenance)
    if edit_impact is not None:
        metadata["edit_impact"] = dict(edit_impact)
    if extra:
        metadata.update(extra)
    return metadata


def build_validation_edit_metadata(
    *,
    edit_type: str,
    scope: str,
    parameters: dict[str, object] | None = None,
    coverage: dict[str, object] | None = None,
    edit_provenance: dict[str, object] | None = None,
    edit_impact: dict[str, object] | None = None,
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
        edit_provenance=edit_provenance,
        edit_impact=edit_impact,
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

    errors.extend(_validate_optional_edit_provenance(metadata))
    errors.extend(_validate_optional_edit_impact(metadata))
    return errors


def _validate_optional_edit_provenance(metadata: dict[str, object]) -> list[str]:
    provenance = metadata.get("edit_provenance")
    if provenance is None:
        return []
    if not isinstance(provenance, dict):
        return ["edit_provenance must be an object when present"]

    errors: list[str] = []
    family = provenance.get("edit_family")
    if family is not None and (
        not isinstance(family, str) or family not in EDIT_PROVENANCE_FAMILIES
    ):
        errors.append(f"edit_provenance.edit_family unsupported: {family!r}")

    method = provenance.get("edit_method")
    if method is not None and (not isinstance(method, str) or not method.strip()):
        errors.append("edit_provenance.edit_method must be a non-empty string")

    edit_count = provenance.get("edit_count")
    if edit_count is not None and (
        not isinstance(edit_count, int)
        or isinstance(edit_count, bool)
        or edit_count < 1
    ):
        errors.append("edit_provenance.edit_count must be a positive integer")

    for key in (
        "target_set_digest",
        "editor_artifact_digest",
        "self_edit_data_digest",
    ):
        digest = provenance.get(key)
        if digest is not None and (
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
        ):
            errors.append(
                f"edit_provenance.{key} must be a sha256:<64 lowercase hex> digest"
            )

    dynamic_required = provenance.get("dynamic_runtime_required")
    if dynamic_required is not None and not isinstance(dynamic_required, bool):
        errors.append("edit_provenance.dynamic_runtime_required must be boolean")
    return errors


def _validate_optional_edit_impact(metadata: dict[str, object]) -> list[str]:
    impact = metadata.get("edit_impact")
    if impact is None:
        return []
    if not isinstance(impact, dict):
        return ["edit_impact must be an object when present"]

    scenario_types = impact.get("scenario_types")
    if scenario_types is None:
        return []
    if not isinstance(scenario_types, list):
        return ["edit_impact.scenario_types must be a list when present"]

    errors: list[str] = []
    for index, scenario_type in enumerate(scenario_types):
        if (
            not isinstance(scenario_type, str)
            or scenario_type not in EDIT_IMPACT_SCENARIO_TYPES
        ):
            errors.append(
                f"edit_impact.scenario_types[{index}] unsupported: {scenario_type!r}"
            )
    return errors


def _path_segments(name: str) -> tuple[str, ...]:
    return _tensor_ops._path_segments(name)


def _is_excluded_multimodal_path(name: str) -> bool:
    return _tensor_ops._is_excluded_multimodal_path(name)


def matches_edit_scope(name: str, scope: str) -> bool:
    return _tensor_ops.matches_edit_scope(name, scope)


EditStats = _tensor_ops.EditStats


def total_model_params(model: Any) -> int:
    return _tensor_ops.total_model_params(model)


def _matches_scope(name: str, scope: str) -> bool:
    return _tensor_ops._matches_scope(name, scope)


def round_to_nearest_dequantized(
    tensor: Any,
    *,
    bits: int,
    group_size: int,
) -> Any:
    return _tensor_ops.round_to_nearest_dequantized(
        tensor,
        bits=bits,
        group_size=group_size,
    )


def apply_rtn_dequantized_simulation(
    model: Any,
    *,
    bits: int,
    group_size: int,
    scope: str,
) -> EditStats:
    return _tensor_ops.apply_rtn_dequantized_simulation(
        model,
        bits=bits,
        group_size=group_size,
        scope=scope,
    )


def fp8_dtype(format_type: str) -> Any:
    return _tensor_ops.fp8_dtype(format_type)


def apply_fp8_dequantized_simulation(
    model: Any,
    *,
    format_type: str,
    scope: str,
) -> EditStats:
    return _tensor_ops.apply_fp8_dequantized_simulation(
        model,
        format_type=format_type,
        scope=scope,
    )


def magnitude_prune_tensor(weight: Any, sparsity: float) -> Any:
    return _tensor_ops.magnitude_prune_tensor(weight, sparsity)


def apply_dense_magnitude_prune(
    model: Any,
    *,
    sparsity: float,
    scope: str,
) -> EditStats:
    return _tensor_ops.apply_dense_magnitude_prune(
        model,
        sparsity=sparsity,
        scope=scope,
    )


def parse_scope_layers(raw_scope: str) -> tuple[str, int | None, int | None]:
    return _tensor_ops.parse_scope_layers(raw_scope)


def extract_layer_index(name: str) -> int | None:
    return _tensor_ops.extract_layer_index(name)


def _layer_selected(
    name: str,
    *,
    layer_limit: int | None,
    layer_exact: int | None,
) -> bool:
    return _tensor_ops._layer_selected(
        name,
        layer_limit=layer_limit,
        layer_exact=layer_exact,
    )


def truncated_svd(weight: Any, rank: int) -> Any:
    return _tensor_ops.truncated_svd(weight, rank)


def apply_dense_lowrank_approximation(
    model: Any,
    *,
    rank: int,
    scope: str,
) -> EditStats:
    return _tensor_ops.apply_dense_lowrank_approximation(
        model,
        rank=rank,
        scope=scope,
    )


__all__ = [
    "ALLOWED_ARTIFACT_CLASSES",
    "DEPLOYABLE_OPTIMIZED_SUBJECT",
    "EDIT_METADATA_SCHEMA",
    "EDIT_IMPACT_SCENARIO_TYPES",
    "EDIT_PROVENANCE_FAMILIES",
    "EDIT_SEMANTICS_DEPLOYABLE",
    "EDIT_SEMANTICS_EXTERNAL_SUBJECT",
    "EditStats",
    "EVIDENCE_ONLY_PACK",
    "FAULT_INJECTION_FIXTURE",
    "ResolvedEditSpec",
    "VALIDATION_STORAGE_FORMATS",
    "VALIDATION_SUBJECT_CHECKPOINT",
    "apply_dense_lowrank_approximation",
    "apply_dense_magnitude_prune",
    "apply_fp8_dequantized_simulation",
    "apply_rtn_dequantized_simulation",
    "build_edit_metadata",
    "build_validation_edit_metadata",
    "fp8_dtype",
    "magnitude_prune_tensor",
    "normalize_coverage",
    "parse_edit_specs_json",
    "parse_scope_layers",
    "read_edit_metadata",
    "resolve_batch_entry",
    "resolve_edit_spec",
    "round_to_nearest_dequantized",
    "storage_format_for_edit",
    "total_model_params",
    "truncated_svd",
    "validate_edit_metadata",
    "write_edit_metadata",
]
