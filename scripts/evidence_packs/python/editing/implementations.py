from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - metadata/spec parsing can run without torch

    class _TorchUnavailable:
        def no_grad(self):
            return lambda func: func

        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError("No module named 'torch'")

    torch = _TorchUnavailable()  # type: ignore[assignment]

_SCOPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "all": (
        "linear",
        "dense",
        "proj",
        "fc",
        "mlp",
        "attn",
        "wqkv",
        "query_key_value",
    ),
    "ffn": (
        "mlp",
        "fc",
        "dense",
        "gate",
        "up_proj",
        "down_proj",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ),
    "attn": (
        "attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "wqkv",
        "out_proj",
        "query_key_value",
    ),
}

_EXCLUDED_PATH_SEGMENTS = frozenset(
    {
        "connector",
        "mm_projector",
        "multi_modal_projector",
        "vision_encoder",
        "vision_model",
        "vision_resampler",
        "vision_tower",
    }
)

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


def _path_segments(name: str) -> tuple[str, ...]:
    return tuple(
        segment for segment in re.split(r"[^a-z0-9_]+", name.lower()) if segment
    )


def _is_excluded_multimodal_path(name: str) -> bool:
    segments = _path_segments(name)
    return any(segment in _EXCLUDED_PATH_SEGMENTS for segment in segments)


def matches_edit_scope(name: str, scope: str) -> bool:
    if _is_excluded_multimodal_path(name):
        return False
    name_lower = name.lower()
    keywords = _SCOPE_KEYWORDS.get(scope)
    if not keywords:
        return False
    return any(keyword in name_lower for keyword in keywords)


@dataclass
class EditStats:
    edited_tensors: int = 0
    edited_params: int = 0
    total_params: int = 0
    details: dict[str, object] = field(default_factory=dict)

    @property
    def coverage_ratio(self) -> float:
        if self.total_params <= 0:
            return 0.0
        return self.edited_params / self.total_params

    def coverage_payload(self) -> dict[str, object]:
        return {
            "edited_tensors": self.edited_tensors,
            "edited_params": self.edited_params,
            "total_params": self.total_params,
            "coverage_ratio": self.coverage_ratio,
        }


def total_model_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _matches_scope(name: str, scope: str) -> bool:
    return matches_edit_scope(name, scope)


@torch.no_grad()
def round_to_nearest_dequantized(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    qmin = -(2 ** (bits - 1))
    qmax = max((2 ** (bits - 1)) - 1, 1)
    orig_shape = tensor.shape
    flat = tensor.reshape(orig_shape[0], -1)
    in_features = flat.shape[1]
    eff_group_size = group_size if group_size > 0 else in_features
    if eff_group_size >= in_features:
        eff_group_size = in_features
    num_groups = (in_features + eff_group_size - 1) // eff_group_size
    pad = (num_groups * eff_group_size) - in_features
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(orig_shape[0], num_groups, eff_group_size)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / qmax, min=1e-10)
    quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
    quantized = quantized.reshape(orig_shape[0], num_groups * eff_group_size)
    if pad > 0:
        quantized = quantized[:, :in_features]
    return quantized.reshape(orig_shape).to(tensor.dtype)


@torch.no_grad()
def apply_rtn_dequantized_simulation(
    model: Any,
    *,
    bits: int,
    group_size: int,
    scope: str,
) -> EditStats:
    stats = EditStats(total_params=total_model_params(model))
    for name, param in model.named_parameters():
        if _matches_scope(name, scope) and param.dim() >= 2:
            param.data = round_to_nearest_dequantized(
                param.data,
                bits=bits,
                group_size=group_size,
            )
            stats.edited_tensors += 1
            stats.edited_params += param.numel()
            if stats.edited_tensors <= 3:
                print(f"  Quantized: {name} ({tuple(param.shape)})")
    stats.details.update({"bits": bits, "group_size": group_size})
    return stats


def fp8_dtype(format_type: str) -> torch.dtype | None:
    if format_type in {"e4m3", "e4m3fn", "e4m3fnuz"}:
        return getattr(torch, "float8_e4m3fn", None)
    if format_type in {"e5m2", "e5m2fn", "e5m2fnuz"}:
        return getattr(torch, "float8_e5m2", None)
    return None


@torch.no_grad()
def apply_fp8_dequantized_simulation(
    model: Any,
    *,
    format_type: str,
    scope: str,
) -> EditStats:
    dtype = fp8_dtype(format_type)
    stats = EditStats(total_params=total_model_params(model))
    rel_error_total = 0.0

    for name, param in model.named_parameters():
        if not _matches_scope(name, scope) or param.dim() < 2:
            continue
        original = param.data.clone()
        if dtype is None:
            param.data = param.data.to(torch.float16).to(param.dtype)
        else:
            param.data = param.data.to(dtype).to(param.dtype)
        stats.edited_tensors += 1
        stats.edited_params += param.numel()
        denom = original.abs().mean() + 1e-10
        rel_error_total += float((param.data - original).abs().mean() / denom)
        if stats.edited_tensors <= 3:
            print(f"  FP8: {name}")

    avg_error = rel_error_total / max(stats.edited_tensors, 1)
    stats.details.update(
        {
            "format": format_type,
            "avg_relative_error": avg_error,
            "torch_fp8_dtype_available": dtype is not None,
        }
    )
    return stats


@torch.no_grad()
def magnitude_prune_tensor(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    flat = weight.abs().flatten()
    k = int(flat.numel() * sparsity)
    if k == 0:
        return weight
    threshold = torch.kthvalue(flat, k).values
    mask = weight.abs() >= threshold
    return weight * mask.to(weight.dtype)


@torch.no_grad()
def apply_dense_magnitude_prune(
    model: Any,
    *,
    sparsity: float,
    scope: str,
) -> EditStats:
    stats = EditStats(total_params=total_model_params(model))
    total_zeros = 0

    for name, param in model.named_parameters():
        if _matches_scope(name, scope) and param.dim() >= 2:
            original_zeros = int((param == 0).sum().item())
            param.data = magnitude_prune_tensor(param.data, sparsity)
            new_zeros = int((param == 0).sum().item())
            stats.edited_tensors += 1
            stats.edited_params += param.numel()
            total_zeros += new_zeros
            if stats.edited_tensors <= 3:
                print(f"  Pruned: {name} ({original_zeros} -> {new_zeros} zeros)")

    actual_sparsity = total_zeros / stats.edited_params if stats.edited_params else 0.0
    stats.details.update(
        {
            "target_sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
        }
    )
    return stats


def parse_scope_layers(raw_scope: str) -> tuple[str, int | None, int | None]:
    base = (raw_scope or "").strip()
    layer_limit: int | None = None
    layer_exact: int | None = None
    if "@" in base:
        base, rest = base.split("@", 1)
        base = base.strip()
        for item in (s.strip() for s in rest.split(",") if s.strip()):
            if item.startswith("layers="):
                try:
                    layer_limit = int(item.split("=", 1)[1])
                except (TypeError, ValueError):
                    layer_limit = None
            elif item.startswith("layer="):
                try:
                    layer_exact = int(item.split("=", 1)[1])
                except (TypeError, ValueError):
                    layer_exact = None
    return base, layer_limit, layer_exact


def extract_layer_index(name: str) -> int | None:
    marker = ".layers."
    pos = name.find(marker)
    if pos < 0:
        return None
    start = pos + len(marker)
    end = start
    while end < len(name) and name[end].isdigit():
        end += 1
    if end == start:
        return None
    try:
        return int(name[start:end])
    except (TypeError, ValueError):
        return None


def _layer_selected(
    name: str,
    *,
    layer_limit: int | None,
    layer_exact: int | None,
) -> bool:
    if layer_limit is None and layer_exact is None:
        return True
    idx = extract_layer_index(name)
    if idx is None:
        return False
    if layer_exact is not None and idx != layer_exact:
        return False
    if layer_limit is not None and idx >= layer_limit:
        return False
    return True


@torch.no_grad()
def truncated_svd(weight: torch.Tensor, rank: int) -> torch.Tensor:
    if weight.dim() < 2:
        return weight

    original_shape = weight.shape
    weight_2d = weight.view(weight.shape[0], -1).float()
    max_rank = min(weight_2d.shape)
    effective_rank = min(rank, max_rank)
    u, s, v = torch.svd_lowrank(weight_2d, q=effective_rank, niter=2)
    lowrank = (u * s) @ v.T
    return lowrank.to(weight.dtype).view(original_shape)


@torch.no_grad()
def apply_dense_lowrank_approximation(
    model: Any,
    *,
    rank: int,
    scope: str,
) -> EditStats:
    base_scope, layer_limit, layer_exact = parse_scope_layers(scope)
    if base_scope != scope:
        print(
            "Parsed scope="
            f"{scope} -> base_scope={base_scope}, "
            f"layer_limit={layer_limit}, layer={layer_exact}"
        )

    stats = EditStats(total_params=total_model_params(model))
    total_energy_retained = 0.0

    for name, param in model.named_parameters():
        if not _layer_selected(
            name,
            layer_limit=layer_limit,
            layer_exact=layer_exact,
        ):
            continue
        if _matches_scope(name, base_scope) and param.dim() >= 2:
            original_norm = param.data.norm()
            param.data = truncated_svd(param.data, rank)
            new_norm = param.data.norm()
            energy_retained = (
                (new_norm / original_norm).item() if original_norm > 0 else 1.0
            )
            stats.edited_tensors += 1
            stats.edited_params += param.numel()
            total_energy_retained += energy_retained
            if stats.edited_tensors <= 3:
                print(f"  Low-rank: {name}, energy retained: {energy_retained:.4f}")

    avg_energy = (
        total_energy_retained / stats.edited_tensors if stats.edited_tensors else 1.0
    )
    stats.details.update(
        {
            "rank": rank,
            "avg_energy_retained": avg_energy,
            "base_scope": base_scope,
            "layer_limit": layer_limit,
            "layer": layer_exact,
        }
    )
    return stats


__all__ = [
    "ALLOWED_ARTIFACT_CLASSES",
    "DEPLOYABLE_OPTIMIZED_SUBJECT",
    "EDIT_METADATA_SCHEMA",
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
