"""Fail-closed validation for runtime quantization proof sidecars.

The producer observes live Python objects in ``runtime_quantization_proof``.
This module owns the persisted v1 contract and cross-family checks that can be
performed after those observations have been serialized.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from invarlock.core.backend_inventory import BACKEND_INVENTORY_SCHEMA

GPTQMODEL_ADAPTERS = frozenset({"hf_awq", "hf_gptq"})
GPTQMODEL_QLINEAR_PREFIX = "gptqmodel.nn_modules.qlinear."
BITSANDBYTES_RUNTIME_TYPES = frozenset(
    {
        "bitsandbytes.nn.modules.linear4bit",
        "bitsandbytes.nn.modules.linear8bitlt",
    }
)
TORCHAO_RUNTIME_TYPES = frozenset(
    {
        "torchao.dtypes.affine_quantized_tensor.affinequantizedtensor",
        "torchao.quantization.int8tensor",
        "torchao.quantization.quantize_.workflows.int8.int8_tensor.int8tensor",
    }
)

# The pinned GPTQModel runtime exposes family-specific wrapper modules.  The
# families share a package and generic QLinear base, so package membership
# alone cannot prove which quantization family produced an observation.
GPTQMODEL_AWQ_MODULES = frozenset(
    {
        "bitblas_awq",
        "exllamav2_awq",
        "gemm_awq",
        "gemm_awq_triton",
        "gemv_awq",
        "gemv_fast_awq",
        "machete_awq",
        "marlin_awq",
        "torch_aten_kernel_awq",
        "torch_awq",
        "torch_fused_awq",
        "torch_int8_awq",
    }
)
GPTQMODEL_GPTQ_MODULES = frozenset(
    {
        "bitblas",
        "exllamav2",
        "machete",
        "marlin",
        "torch",
        "torch_aten_kernel",
        "torch_fused",
        "torch_int8",
        "tritonv2",
    }
)

_RUNTIME_PROOF_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "proof_kind",
        "adapter",
        "backend",
        "backend_version",
        "ok",
        "status",
        "reason",
        "live_model_observed",
        "module_inventory_observed",
        "recognized_quantized_runtime_type_count",
        "recognized_quantized_runtime_types",
        "recognized_quantized_runtime_observation_kinds",
        "live_model_quantization_method",
        "backend_runtime_importable",
        "backend_runtime_import_error_type",
        "backend_runtime_version",
        "backend_runtime_compatibility_bridge_required",
        "backend_runtime_compatibility_bridge_applied",
        "backend_runtime_compatibility_bridge_error_type",
        "packed_storage_artifact_proof_required",
        "artifact_binding",
    }
)
_BACKEND_INVENTORY_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "adapter",
        "backend",
        "backend_version",
        "quantized_module_count",
        "quantized_module_types",
        "quantized_observation_kinds",
        "transformers_version",
        "quantization_config",
        "device_map",
        "memory_footprint",
        "load_smoke",
        "inference_smoke",
    }
)
_GPTQMODEL_RUNTIME_FIELDS = (
    "backend_runtime_importable",
    "backend_runtime_import_error_type",
    "backend_runtime_version",
    "backend_runtime_compatibility_bridge_required",
    "backend_runtime_compatibility_bridge_applied",
    "backend_runtime_compatibility_bridge_error_type",
)


def _is_quantized_linear_type(type_name: str) -> bool:
    return "linear" in type_name or "qlinear" in type_name


def runtime_type_name_matches_adapter(
    *,
    adapter: str,
    type_name: str,
    quantization_method: str | None,
) -> bool:
    """Recognize the v1 sidecar's allowed persisted type-name surface."""

    normalized = type_name.casefold()
    if adapter == "hf_bnb":
        return normalized in BITSANDBYTES_RUNTIME_TYPES
    if adapter == "hf_torchao":
        return normalized in TORCHAO_RUNTIME_TYPES
    if adapter == "hf_hqq":
        return normalized.startswith("hqq.") and "hqqlinear" in normalized
    if adapter == "hf_quanto":
        return normalized == "optimum.quanto.nn.qlinear.qlinear"
    if adapter not in GPTQMODEL_ADAPTERS:
        return False
    if not normalized.startswith(GPTQMODEL_QLINEAR_PREFIX):
        return False
    remainder = normalized.removeprefix(GPTQMODEL_QLINEAR_PREFIX)
    module_name, separator, _class_name = remainder.partition(".")
    if not separator:
        expected_method = "awq" if adapter == "hf_awq" else "gptq"
        return _is_quantized_linear_type(normalized) and (
            quantization_method == expected_method
        )
    family_modules = (
        GPTQMODEL_AWQ_MODULES if adapter == "hf_awq" else GPTQMODEL_GPTQ_MODULES
    )
    return module_name in family_modules and _is_quantized_linear_type(normalized)


def _runtime_contract_identity_errors(
    *,
    payload: Mapping[str, Any],
    expected_adapter: str,
    expected_backend: str,
    expected_schema: str,
    expected_proof_kind: str,
) -> list[str]:
    errors: list[str] = []
    payload_keys = set(payload)
    missing_fields = sorted(_RUNTIME_PROOF_REQUIRED_FIELDS - payload_keys)
    unexpected_fields = sorted(payload_keys - _RUNTIME_PROOF_REQUIRED_FIELDS)
    if missing_fields:
        errors.append(
            "runtime quantization proof is missing required fields: "
            + ", ".join(missing_fields)
        )
    if unexpected_fields:
        errors.append(
            "runtime quantization proof has unsupported v1 fields: "
            + ", ".join(unexpected_fields)
        )
    if payload.get("schema") != expected_schema:
        errors.append("runtime quantization proof schema does not match v1")
    if payload.get("proof_kind") != expected_proof_kind:
        errors.append(
            "runtime quantization proof kind is not a live runtime type inventory"
        )
    if payload.get("adapter") != expected_adapter:
        errors.append(
            "runtime quantization proof adapter does not match the selected subject "
            "adapter"
        )
    if payload.get("backend") != expected_backend:
        errors.append(
            "runtime quantization proof backend does not match the selected subject "
            "adapter"
        )
    backend_version = payload.get("backend_version")
    if not isinstance(backend_version, str) or not backend_version.strip():
        errors.append("runtime quantization proof backend_version must be non-empty")
    return errors


def _positive_runtime_status_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_values = (
        ("ok", True, "runtime quantization proof must record ok: true"),
        (
            "status",
            "verified_live_runtime_types",
            "runtime quantization proof status must be verified_live_runtime_types",
        ),
        (
            "reason",
            "recognized_live_quantized_runtime_types",
            "runtime quantization proof reason must be "
            "recognized_live_quantized_runtime_types",
        ),
        (
            "live_model_observed",
            True,
            "runtime quantization proof must record a live model observation",
        ),
        (
            "module_inventory_observed",
            True,
            "runtime quantization proof must record a module inventory observation",
        ),
        (
            "packed_storage_artifact_proof_required",
            False,
            "runtime quantization proof cannot stand in for packed-storage evidence",
        ),
        (
            "artifact_binding",
            "not_attempted",
            "runtime quantization proof artifact_binding must be not_attempted",
        ),
    )
    for field, expected, message in expected_values:
        actual = payload.get(field)
        matches = (
            actual is expected if isinstance(expected, bool) else actual == expected
        )
        if not matches:
            errors.append(message)
    return errors


def _runtime_inventory_errors(
    payload: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    count = payload.get("recognized_quantized_runtime_type_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        errors.append(
            "runtime quantization proof recognized runtime type count must be positive"
        )
    runtime_types = payload.get("recognized_quantized_runtime_types")
    if not isinstance(runtime_types, list) or not runtime_types:
        errors.append(
            "runtime quantization proof must contain recognized runtime type names"
        )
        runtime_types = []
    elif not all(
        isinstance(type_name, str) and type_name and type_name.strip() == type_name
        for type_name in runtime_types
    ):
        errors.append("runtime quantization proof runtime type names must be strings")
        runtime_types = []
    elif runtime_types != sorted(set(runtime_types)):
        errors.append(
            "runtime quantization proof runtime type names must be sorted and unique"
        )
    if (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count < len(runtime_types)
    ):
        errors.append(
            "runtime quantization proof type count is smaller than its runtime "
            "type inventory"
        )
    observation_kinds = payload.get("recognized_quantized_runtime_observation_kinds")
    if (
        not isinstance(observation_kinds, list)
        or not observation_kinds
        or not all(
            isinstance(kind, str) and kind in {"module", "direct_weight"}
            for kind in observation_kinds
        )
    ):
        errors.append(
            "runtime quantization proof observation kinds must be a non-empty "
            "supported list"
        )
    elif observation_kinds != sorted(set(observation_kinds)):
        errors.append(
            "runtime quantization proof observation kinds must be sorted and unique"
        )
    return errors, runtime_types


def _quantization_family_errors(
    *,
    payload: Mapping[str, Any],
    expected_adapter: str,
    runtime_types: list[str],
) -> list[str]:
    errors: list[str] = []
    raw_quantization_method = payload.get("live_model_quantization_method")
    quantization_method = (
        raw_quantization_method if isinstance(raw_quantization_method, str) else None
    )
    if raw_quantization_method is not None and quantization_method not in {
        "awq",
        "gptq",
    }:
        errors.append(
            "runtime quantization proof live quantization method must be awq, gptq, "
            "or null"
        )
    expected_method = (
        "awq"
        if expected_adapter == "hf_awq"
        else "gptq"
        if expected_adapter == "hf_gptq"
        else None
    )
    if expected_method is None and raw_quantization_method is not None:
        errors.append(
            "non-GPTQModel runtime proof must record a null live quantization method"
        )
    if expected_method is not None and quantization_method not in {
        None,
        expected_method,
    }:
        errors.append(
            "runtime quantization proof live quantization method does not match "
            "the selected subject adapter"
        )
    for type_name in runtime_types:
        if not runtime_type_name_matches_adapter(
            adapter=expected_adapter,
            type_name=type_name,
            quantization_method=quantization_method,
        ):
            errors.append(
                "runtime quantization proof has an unrecognized or cross-family "
                f"runtime type: {type_name!r}"
            )
    return errors


def _gptqmodel_runtime_errors(
    *, payload: Mapping[str, Any], expected_adapter: str
) -> list[str]:
    errors: list[str] = []
    if expected_adapter not in GPTQMODEL_ADAPTERS:
        for field in _GPTQMODEL_RUNTIME_FIELDS:
            if payload.get(field) is not None:
                errors.append(
                    f"non-GPTQModel runtime proof must record {field} as null"
                )
        return errors
    if payload.get("backend_runtime_importable") is not True:
        errors.append(
            "GPTQModel runtime proof must record backend_runtime_importable: true"
        )
    if payload.get("backend_runtime_import_error_type") is not None:
        errors.append("GPTQModel runtime proof records a runtime import error")
    runtime_version = payload.get("backend_runtime_version")
    if not isinstance(runtime_version, str) or not runtime_version.strip():
        errors.append("GPTQModel runtime proof must record a non-empty runtime version")
    bridge_required = payload.get("backend_runtime_compatibility_bridge_required")
    bridge_applied = payload.get("backend_runtime_compatibility_bridge_applied")
    if not isinstance(bridge_required, bool):
        errors.append("GPTQModel runtime proof bridge-required field must be boolean")
    if not isinstance(bridge_applied, bool):
        errors.append("GPTQModel runtime proof bridge-applied field must be boolean")
    if bridge_required is True and bridge_applied is not True:
        errors.append("GPTQModel runtime proof required bridge was not applied")
    if payload.get("backend_runtime_compatibility_bridge_error_type") is not None:
        errors.append("GPTQModel runtime proof records a compatibility bridge error")
    return errors


def validate_runtime_quantization_proof_payload(
    *,
    payload: Mapping[str, Any],
    expected_adapter: str,
    expected_backend: str,
    expected_schema: str,
    expected_proof_kind: str,
) -> list[str]:
    """Validate the complete original positive-runtime proof contract."""

    errors = _runtime_contract_identity_errors(
        payload=payload,
        expected_adapter=expected_adapter,
        expected_backend=expected_backend,
        expected_schema=expected_schema,
        expected_proof_kind=expected_proof_kind,
    )
    errors.extend(_positive_runtime_status_errors(payload))
    inventory_errors, runtime_types = _runtime_inventory_errors(payload)
    errors.extend(inventory_errors)
    errors.extend(
        _quantization_family_errors(
            payload=payload,
            expected_adapter=expected_adapter,
            runtime_types=runtime_types,
        )
    )
    errors.extend(
        _gptqmodel_runtime_errors(
            payload=payload,
            expected_adapter=expected_adapter,
        )
    )
    return errors


def validate_backend_inventory_payload(
    *,
    payload: Mapping[str, Any],
    expected_adapter: str,
    expected_backend: str,
) -> list[str]:
    """Validate the inventory shape needed to cross-bind an original proof."""

    errors: list[str] = []
    payload_keys = set(payload)
    missing_fields = sorted(_BACKEND_INVENTORY_REQUIRED_FIELDS - payload_keys)
    unexpected_fields = sorted(payload_keys - _BACKEND_INVENTORY_REQUIRED_FIELDS)
    if missing_fields:
        errors.append(
            "backend inventory is missing required fields: " + ", ".join(missing_fields)
        )
    if unexpected_fields:
        errors.append(
            "backend inventory has unsupported v1 fields: "
            + ", ".join(unexpected_fields)
        )
    if payload.get("schema") != BACKEND_INVENTORY_SCHEMA:
        errors.append("backend inventory schema does not match v1")
    if payload.get("adapter") != expected_adapter:
        errors.append(
            "backend inventory adapter does not match the selected subject adapter"
        )
    if payload.get("backend") != expected_backend:
        errors.append(
            "backend inventory backend does not match the selected subject adapter"
        )
    version = payload.get("backend_version")
    if not isinstance(version, str) or not version.strip():
        errors.append("backend inventory backend_version must be non-empty")
    count = payload.get("quantized_module_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        errors.append(
            "backend inventory quantized runtime observation count must be non-negative"
        )
    module_types = payload.get("quantized_module_types")
    if not isinstance(module_types, list) or not all(
        isinstance(type_name, str) and type_name and type_name.strip() == type_name
        for type_name in module_types
    ):
        errors.append(
            "backend inventory quantized runtime observation types must be a string list"
        )
        module_types = []
    elif module_types != sorted(set(module_types)):
        errors.append(
            "backend inventory quantized runtime observation types must be sorted and unique"
        )
    if (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count < len(module_types)
    ):
        errors.append(
            "backend inventory observation count is smaller than its type inventory"
        )
    observation_kinds = payload.get("quantized_observation_kinds")
    if (
        not isinstance(observation_kinds, list)
        or not observation_kinds
        or not all(
            isinstance(kind, str) and kind in {"module", "direct_weight"}
            for kind in observation_kinds
        )
    ):
        errors.append(
            "backend inventory observation kinds must be a non-empty supported list"
        )
    elif observation_kinds != sorted(set(observation_kinds)):
        errors.append("backend inventory observation kinds must be sorted and unique")
    if payload.get("load_smoke") is not True:
        errors.append("backend inventory must record load_smoke: true")
    if payload.get("inference_smoke") is not True:
        errors.append("backend inventory must record inference_smoke: true")
    return errors


__all__ = [
    "BITSANDBYTES_RUNTIME_TYPES",
    "GPTQMODEL_ADAPTERS",
    "GPTQMODEL_AWQ_MODULES",
    "GPTQMODEL_GPTQ_MODULES",
    "GPTQMODEL_QLINEAR_PREFIX",
    "TORCHAO_RUNTIME_TYPES",
    "runtime_type_name_matches_adapter",
    "validate_backend_inventory_payload",
    "validate_runtime_quantization_proof_payload",
]
