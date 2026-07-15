"""Fail-closed runtime observations for optional quantized model backends.

These sidecars record only what the live loaded model exposes through its
module tree.  They are deliberately not artifact proofs: a strict evidence
lane must additionally bind a verified artifact and any backend-specific
packed-storage facts it requires.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

from invarlock.core.backend_inventory import quantized_adapter_backend
from invarlock.core.installed_distribution import installed_distribution_version
from invarlock.core.runtime_observation import observe_model_runtime
from invarlock.core.runtime_quantization.validation import (
    BITSANDBYTES_RUNTIME_TYPES,
    GPTQMODEL_ADAPTERS,
    GPTQMODEL_AWQ_MODULES,
    GPTQMODEL_GPTQ_MODULES,
    GPTQMODEL_QLINEAR_PREFIX,
    TORCHAO_RUNTIME_TYPES,
    validate_backend_inventory_payload,
    validate_runtime_quantization_proof_payload,
)
from invarlock.gptqmodel_runtime import (
    GPTQModelRuntimeStatus,
    prepare_gptqmodel_runtime,
)

RUNTIME_QUANTIZATION_PROOF_SCHEMA = "invarlock/runtime-quantization-proof-v1"
RUNTIME_QUANTIZATION_PROOF_FILENAME = "runtime_quantization_proof.json"
RUNTIME_QUANTIZATION_PROOF_KIND = "live_loaded_model_runtime_type_inventory"

_SUPPORTED_ADAPTERS = frozenset(
    {
        "hf_bnb",
        "hf_awq",
        "hf_gptq",
        "hf_torchao",
        "hf_hqq",
        "hf_quanto",
        "hf_ct",
    }
)
_COMPRESSED_TENSORS_ADAPTER = "hf_ct"


def _backend_version(backend: str | None) -> str | None:
    if not backend:
        return None
    return installed_distribution_version(backend)


def _runtime_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _resolve_imported_runtime_type(value: Any) -> type[Any] | None:
    """Resolve ``type(value)`` through its imported backend module.

    A Python class can forge ``__module__`` and ``__qualname__``.  The proof
    producer must therefore verify that importing that exact qualified name
    returns the same class object before accepting a backend runtime type.
    This is deliberately a process-local observation; a persisted FQCN alone
    cannot recreate that identity check and is validated separately below.
    """

    runtime_type = type(value)
    module_name = runtime_type.__module__
    qualified_name = runtime_type.__qualname__
    if not isinstance(module_name, str) or not module_name:
        return None
    if not isinstance(qualified_name, str) or not qualified_name:
        return None
    parts = qualified_name.split(".")
    if any(not part.isidentifier() for part in parts):
        return None
    try:
        imported: object = import_module(module_name)
        for part in parts:
            imported = getattr(imported, part)
    except (AttributeError, ImportError, ModuleNotFoundError, TypeError, ValueError):
        return None
    return imported if isinstance(imported, type) else None


def _is_imported_runtime_type_identity(value: Any) -> bool:
    return _resolve_imported_runtime_type(value) is type(value)


def _is_bitsandbytes_module(
    value: Any,
    type_name: str,
    _quantization_method: str | None,
) -> bool:
    return (
        type_name in BITSANDBYTES_RUNTIME_TYPES
        and _is_imported_runtime_type_identity(value)
    )


def _is_quantized_linear_type(type_name: str) -> bool:
    return "linear" in type_name or "qlinear" in type_name


def _gptqmodel_qlinear_module_name(type_name: str) -> str | None:
    """Return a GPTQModel QLinear module stem, or ``None`` when ambiguous.

    For example, ``...qlinear.marlin.MarlinLinear`` yields ``"marlin"``.
    A class directly in ``...qlinear`` has no family-specific module stem and
    must be disambiguated by the live model's quantization configuration.
    """

    if not type_name.startswith(GPTQMODEL_QLINEAR_PREFIX):
        return None
    remainder = type_name.removeprefix(GPTQMODEL_QLINEAR_PREFIX)
    module_name, separator, _class_name = remainder.partition(".")
    if not separator:
        return None
    return module_name


def _is_generic_gptqmodel_qlinear(type_name: str) -> bool:
    if not type_name.startswith(GPTQMODEL_QLINEAR_PREFIX):
        return False
    remainder = type_name.removeprefix(GPTQMODEL_QLINEAR_PREFIX)
    return "." not in remainder and _is_quantized_linear_type(type_name)


def _live_quantization_method(model: Any) -> str | None:
    """Read only an explicit live config family used to resolve base wrappers.

    A configuration can never make a dense module succeed: it is consulted
    only after the module inventory sees an otherwise ambiguous GPTQModel
    ``QLinear`` type.  This keeps AWQ and GPTQ labels from cross-validating one
    another while preserving support for configurations represented as either
    dicts or Transformers config objects.
    """

    try:
        model_config = getattr(model, "config", None)
        if isinstance(model_config, Mapping):
            quantization_config = model_config.get("quantization_config")
        else:
            quantization_config = getattr(model_config, "quantization_config", None)
    except Exception:  # noqa: BLE001 - optional observation must fail closed
        return None
    if quantization_config is None:
        return None
    try:
        if isinstance(quantization_config, Mapping):
            raw_method = quantization_config.get(
                "quant_method",
                quantization_config.get("quantization_method"),
            )
        else:
            raw_method = getattr(
                quantization_config,
                "quant_method",
                getattr(quantization_config, "quantization_method", None),
            )
            if raw_method is None:
                class_name = type(quantization_config).__name__.casefold()
                if class_name == "awqconfig":
                    return "awq"
                if class_name == "gptqconfig":
                    return "gptq"
    except Exception:  # noqa: BLE001 - optional observation must fail closed
        return None
    if not isinstance(raw_method, str):
        return None
    normalized_method = raw_method.strip().casefold().replace("-", "_")
    if normalized_method in {"awq", "gptq"}:
        return normalized_method
    return None


def _is_awq_module(
    value: Any,
    type_name: str,
    quantization_method: str | None,
) -> bool:
    if not _is_quantized_linear_type(type_name):
        return False
    module_name = _gptqmodel_qlinear_module_name(type_name)
    if module_name in GPTQMODEL_AWQ_MODULES and _is_imported_runtime_type_identity(
        value
    ):
        return True
    return (
        _is_generic_gptqmodel_qlinear(type_name)
        and quantization_method == "awq"
        and _is_imported_runtime_type_identity(value)
    )


def _is_gptq_module(
    value: Any,
    type_name: str,
    quantization_method: str | None,
) -> bool:
    if not _is_quantized_linear_type(type_name):
        return False
    module_name = _gptqmodel_qlinear_module_name(type_name)
    if module_name in GPTQMODEL_GPTQ_MODULES and _is_imported_runtime_type_identity(
        value
    ):
        return True
    return (
        _is_generic_gptqmodel_qlinear(type_name)
        and quantization_method == "gptq"
        and _is_imported_runtime_type_identity(value)
    )


def _is_torchao_module(
    value: Any,
    type_name: str,
    _quantization_method: str | None,
) -> bool:
    return type_name in TORCHAO_RUNTIME_TYPES and _is_imported_runtime_type_identity(
        value
    )


def _is_hqq_module(
    value: Any,
    type_name: str,
    _quantization_method: str | None,
) -> bool:
    return (
        type_name.startswith("hqq.")
        and "hqqlinear" in type_name
        and _is_imported_runtime_type_identity(value)
    )


def _is_quanto_module(
    value: Any,
    type_name: str,
    _quantization_method: str | None,
) -> bool:
    return (
        type_name == "optimum.quanto.nn.qlinear.qlinear"
        and _is_imported_runtime_type_identity(value)
    )


_MODULE_RECOGNIZERS: dict[str, Callable[[Any, str, str | None], bool]] = {
    "hf_bnb": _is_bitsandbytes_module,
    "hf_awq": _is_awq_module,
    "hf_gptq": _is_gptq_module,
    "hf_torchao": _is_torchao_module,
    "hf_hqq": _is_hqq_module,
    "hf_quanto": _is_quanto_module,
}


def _recognized_runtime_type_inventory(
    model: Any,
    *,
    recognizer: Callable[[Any, str, str | None], bool],
    quantization_method: str | None,
) -> tuple[bool, int, list[str], list[str]]:
    observed, observations = observe_model_runtime(model)
    if not observed:
        return False, 0, [], []
    recognized = [
        observation
        for observation in observations
        if recognizer(
            observation.value,
            observation.fqcn.casefold(),
            quantization_method,
        )
    ]
    return (
        True,
        len(recognized),
        sorted({observation.fqcn for observation in recognized}),
        sorted({observation.kind for observation in recognized}),
    )


def _proof_payload(
    *,
    adapter: str,
    backend: str | None,
    ok: bool,
    status: str,
    reason: str,
    live_model_observed: bool,
    module_inventory_observed: bool,
    recognized_quantized_runtime_type_count: int | None,
    recognized_quantized_runtime_types: list[str],
    recognized_quantized_runtime_observation_kinds: list[str] | None = None,
    packed_storage_artifact_proof_required: bool,
    live_model_quantization_method: str | None = None,
    backend_runtime_importable: bool | None = None,
    backend_runtime_import_error_type: str | None = None,
    backend_runtime_version: str | None = None,
    backend_runtime_compatibility_bridge_required: bool | None = None,
    backend_runtime_compatibility_bridge_applied: bool | None = None,
    backend_runtime_compatibility_bridge_error_type: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": RUNTIME_QUANTIZATION_PROOF_SCHEMA,
        "proof_kind": RUNTIME_QUANTIZATION_PROOF_KIND,
        "adapter": adapter,
        "backend": backend,
        "backend_version": _backend_version(backend),
        "ok": ok,
        "status": status,
        "reason": reason,
        "live_model_observed": live_model_observed,
        "module_inventory_observed": module_inventory_observed,
        "recognized_quantized_runtime_type_count": (
            recognized_quantized_runtime_type_count
        ),
        "recognized_quantized_runtime_types": recognized_quantized_runtime_types,
        "recognized_quantized_runtime_observation_kinds": (
            recognized_quantized_runtime_observation_kinds or []
        ),
        "live_model_quantization_method": live_model_quantization_method,
        "backend_runtime_importable": backend_runtime_importable,
        "backend_runtime_import_error_type": backend_runtime_import_error_type,
        "backend_runtime_version": backend_runtime_version,
        "backend_runtime_compatibility_bridge_required": (
            backend_runtime_compatibility_bridge_required
        ),
        "backend_runtime_compatibility_bridge_applied": (
            backend_runtime_compatibility_bridge_applied
        ),
        "backend_runtime_compatibility_bridge_error_type": (
            backend_runtime_compatibility_bridge_error_type
        ),
        "packed_storage_artifact_proof_required": (
            packed_storage_artifact_proof_required
        ),
        "artifact_binding": "not_attempted",
    }


def build_runtime_quantization_proof(
    *,
    adapter: str | None,
    model: Any | None,
) -> dict[str, Any] | None:
    """Observe recognized runtime types on a live loaded supported model.

    ``None`` means the adapter is not one of the optional quantized backends.
    Every supported adapter returns a proof payload.  Only a positive inventory
    of backend-recognized live runtime types yields ``ok: true``.
    """

    adapter_name = str(adapter or "").strip().lower()
    if adapter_name not in _SUPPORTED_ADAPTERS:
        return None
    backend = quantized_adapter_backend(adapter_name)
    if backend is None:
        return _proof_payload(
            adapter=adapter_name,
            backend=None,
            ok=False,
            status="unverified",
            reason="quantized_backend_unrecognized",
            live_model_observed=model is not None,
            module_inventory_observed=False,
            recognized_quantized_runtime_type_count=0,
            recognized_quantized_runtime_types=[],
            packed_storage_artifact_proof_required=False,
        )
    if adapter_name == _COMPRESSED_TENSORS_ADAPTER:
        return _proof_payload(
            adapter=adapter_name,
            backend=backend,
            ok=False,
            status="unsupported",
            reason="packed_storage_artifact_proof_required",
            live_model_observed=model is not None,
            module_inventory_observed=False,
            recognized_quantized_runtime_type_count=None,
            recognized_quantized_runtime_types=[],
            packed_storage_artifact_proof_required=True,
        )
    if model is None:
        return _proof_payload(
            adapter=adapter_name,
            backend=backend,
            ok=False,
            status="unverified",
            reason="live_model_missing",
            live_model_observed=False,
            module_inventory_observed=False,
            recognized_quantized_runtime_type_count=0,
            recognized_quantized_runtime_types=[],
            packed_storage_artifact_proof_required=False,
        )

    runtime_status: GPTQModelRuntimeStatus | None = None
    runtime_importable: bool | None = None
    runtime_import_error_type: str | None = None
    if adapter_name in GPTQMODEL_ADAPTERS:
        runtime_status = prepare_gptqmodel_runtime()
        runtime_importable = runtime_status.ready
        runtime_import_error_type = (
            runtime_status.import_error_type
            or runtime_status.compatibility_bridge_error_type
        )
        if not runtime_importable:
            return _proof_payload(
                adapter=adapter_name,
                backend=backend,
                ok=False,
                status="unavailable",
                reason="gptqmodel_runtime_import_failed",
                live_model_observed=True,
                module_inventory_observed=False,
                recognized_quantized_runtime_type_count=0,
                recognized_quantized_runtime_types=[],
                packed_storage_artifact_proof_required=False,
                backend_runtime_importable=False,
                backend_runtime_import_error_type=runtime_import_error_type,
                backend_runtime_version=runtime_status.gptqmodel_version,
                backend_runtime_compatibility_bridge_required=(
                    runtime_status.compatibility_bridge_required
                ),
                backend_runtime_compatibility_bridge_applied=(
                    runtime_status.compatibility_bridge_applied
                ),
                backend_runtime_compatibility_bridge_error_type=(
                    runtime_status.compatibility_bridge_error_type
                ),
            )

    quantization_method = _live_quantization_method(model)
    recognizer = _MODULE_RECOGNIZERS[adapter_name]
    observed, count, type_names, observation_kinds = _recognized_runtime_type_inventory(
        model,
        recognizer=recognizer,
        quantization_method=quantization_method,
    )
    if not observed:
        return _proof_payload(
            adapter=adapter_name,
            backend=backend,
            ok=False,
            status="unverified",
            reason="module_inventory_unavailable",
            live_model_observed=True,
            module_inventory_observed=False,
            recognized_quantized_runtime_type_count=0,
            recognized_quantized_runtime_types=[],
            packed_storage_artifact_proof_required=False,
            live_model_quantization_method=quantization_method,
            backend_runtime_importable=runtime_importable,
            backend_runtime_import_error_type=runtime_import_error_type,
            backend_runtime_version=(
                runtime_status.gptqmodel_version if runtime_status else None
            ),
            backend_runtime_compatibility_bridge_required=(
                runtime_status.compatibility_bridge_required if runtime_status else None
            ),
            backend_runtime_compatibility_bridge_applied=(
                runtime_status.compatibility_bridge_applied if runtime_status else None
            ),
            backend_runtime_compatibility_bridge_error_type=(
                runtime_status.compatibility_bridge_error_type
                if runtime_status
                else None
            ),
        )
    if count <= 0:
        return _proof_payload(
            adapter=adapter_name,
            backend=backend,
            ok=False,
            status="unverified",
            reason="no_recognized_quantized_runtime_types",
            live_model_observed=True,
            module_inventory_observed=True,
            recognized_quantized_runtime_type_count=0,
            recognized_quantized_runtime_types=[],
            packed_storage_artifact_proof_required=False,
            live_model_quantization_method=quantization_method,
            backend_runtime_importable=runtime_importable,
            backend_runtime_import_error_type=runtime_import_error_type,
            backend_runtime_version=(
                runtime_status.gptqmodel_version if runtime_status else None
            ),
            backend_runtime_compatibility_bridge_required=(
                runtime_status.compatibility_bridge_required if runtime_status else None
            ),
            backend_runtime_compatibility_bridge_applied=(
                runtime_status.compatibility_bridge_applied if runtime_status else None
            ),
            backend_runtime_compatibility_bridge_error_type=(
                runtime_status.compatibility_bridge_error_type
                if runtime_status
                else None
            ),
        )
    return _proof_payload(
        adapter=adapter_name,
        backend=backend,
        ok=True,
        status="verified_live_runtime_types",
        reason="recognized_live_quantized_runtime_types",
        live_model_observed=True,
        module_inventory_observed=True,
        recognized_quantized_runtime_type_count=count,
        recognized_quantized_runtime_types=type_names,
        recognized_quantized_runtime_observation_kinds=observation_kinds,
        packed_storage_artifact_proof_required=False,
        live_model_quantization_method=quantization_method,
        backend_runtime_importable=runtime_importable,
        backend_runtime_import_error_type=runtime_import_error_type,
        backend_runtime_version=(
            runtime_status.gptqmodel_version if runtime_status else None
        ),
        backend_runtime_compatibility_bridge_required=(
            runtime_status.compatibility_bridge_required if runtime_status else None
        ),
        backend_runtime_compatibility_bridge_applied=(
            runtime_status.compatibility_bridge_applied if runtime_status else None
        ),
        backend_runtime_compatibility_bridge_error_type=(
            runtime_status.compatibility_bridge_error_type if runtime_status else None
        ),
    )


def _read_json_object(
    path: str | Path,
    *,
    artifact_name: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Read one sidecar as a JSON object without ambiguous keys or constants."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key {key!r}")
            payload[key] = value
        return payload

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value!r}")

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return None, [f"{artifact_name} is not valid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"{artifact_name} must contain one JSON object"]
    return payload, []


def validate_runtime_quantization_proof_sidecars(
    *,
    proof_path: str | Path,
    expected_adapter: str,
    backend_inventory_path: str | Path,
) -> list[str]:
    """Validate a strict v1 runtime proof and its matching backend inventory.

    This consumes ordinary JSON sidecars at a package boundary.  It does not
    claim to independently observe the original model process; that happened
    only when the producer compared live class identities before writing the
    receipt.  The validator prevents a wrapper from promoting an incomplete,
    cross-adapter, or internally inconsistent positive receipt.
    """

    errors: list[str] = []
    adapter = str(expected_adapter or "").strip().lower()
    backend = quantized_adapter_backend(adapter)
    if adapter not in _MODULE_RECOGNIZERS or backend is None:
        return [
            "runtime quantization proof requires an explicit supported "
            "module-backed quantized subject adapter"
        ]

    proof, proof_errors = _read_json_object(
        proof_path,
        artifact_name="runtime quantization proof",
    )
    errors.extend(proof_errors)
    inventory, inventory_errors = _read_json_object(
        backend_inventory_path,
        artifact_name="backend inventory",
    )
    errors.extend(inventory_errors)
    if proof is None or inventory is None:
        return errors

    errors.extend(
        validate_runtime_quantization_proof_payload(
            payload=proof,
            expected_adapter=adapter,
            expected_backend=backend,
            expected_schema=RUNTIME_QUANTIZATION_PROOF_SCHEMA,
            expected_proof_kind=RUNTIME_QUANTIZATION_PROOF_KIND,
        )
    )
    errors.extend(
        validate_backend_inventory_payload(
            payload=inventory,
            expected_adapter=adapter,
            expected_backend=backend,
        )
    )
    if proof.get("adapter") != inventory.get("adapter"):
        errors.append(
            "runtime quantization proof adapter does not match backend inventory"
        )
    if proof.get("backend") != inventory.get("backend"):
        errors.append(
            "runtime quantization proof backend does not match backend inventory"
        )
    if proof.get("backend_version") != inventory.get("backend_version"):
        errors.append(
            "runtime quantization proof backend_version does not match backend inventory"
        )

    runtime_types = proof.get("recognized_quantized_runtime_types")
    inventory_types = inventory.get("quantized_module_types")
    if runtime_types != inventory_types:
        errors.append(
            "runtime quantization proof runtime types do not exactly match backend inventory"
        )
    if proof.get("recognized_quantized_runtime_type_count") != inventory.get(
        "quantized_module_count"
    ):
        errors.append(
            "runtime quantization proof observation count does not match backend inventory"
        )
    if proof.get("recognized_quantized_runtime_observation_kinds") != inventory.get(
        "quantized_observation_kinds"
    ):
        errors.append(
            "runtime quantization proof observation kinds do not match backend inventory"
        )
    return errors


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an InvarLock runtime quantization proof sidecar."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-sidecars",
        help="validate one v1 runtime proof and its matching backend inventory",
    )
    validate.add_argument("--proof", required=True, help="runtime proof JSON path")
    validate.add_argument(
        "--backend-inventory",
        required=True,
        help="matching backend inventory JSON path",
    )
    validate.add_argument(
        "--adapter",
        required=True,
        help="explicit module-backed quantized subject adapter",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the package-owned sidecar validation command used by integrations."""

    args = _build_argument_parser().parse_args(argv)
    if args.command != "validate-sidecars":  # pragma: no cover - argparse guards it
        return 2
    errors = validate_runtime_quantization_proof_sidecars(
        proof_path=args.proof,
        expected_adapter=args.adapter,
        backend_inventory_path=args.backend_inventory,
    )
    if not errors:
        return 0
    for error in errors:
        sys.stderr.write(f"runtime quantization proof validation failed: {error}\n")
    return 1


def write_runtime_quantization_proof_sidecar(
    output_dir: str | Path,
    proof: Mapping[str, Any] | None,
) -> Path | None:
    """Persist a runtime proof payload next to its run report when available."""

    if proof is None:
        return None
    payload = dict(proof)
    if payload.get("schema") != RUNTIME_QUANTIZATION_PROOF_SCHEMA:
        raise ValueError("runtime quantization proof schema is invalid")
    if not isinstance(payload.get("ok"), bool):
        raise ValueError("runtime quantization proof ok must be boolean")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_path / RUNTIME_QUANTIZATION_PROOF_FILENAME
    sidecar_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return sidecar_path


__all__ = [
    "RUNTIME_QUANTIZATION_PROOF_FILENAME",
    "RUNTIME_QUANTIZATION_PROOF_KIND",
    "RUNTIME_QUANTIZATION_PROOF_SCHEMA",
    "build_runtime_quantization_proof",
    "validate_runtime_quantization_proof_sidecars",
    "write_runtime_quantization_proof_sidecar",
]


if __name__ == "__main__":  # pragma: no cover - exercised through integration shell
    raise SystemExit(main())
