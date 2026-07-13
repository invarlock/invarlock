"""Contracts for canonical deployable coverage of packed bitsandbytes checkpoints."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from typing import Any, cast

LOGICAL_COVERAGE_BASIS = "dense_baseline_unique_parameters"
_LOGICAL_COVERAGE_FIELDS = {
    "basis",
    "weight_tensor_names",
    "weight_tensor_names_sha256",
    "weight_tensor_count",
    "parameter_elements",
    "total_unique_parameter_elements",
}


def canonical_names_sha256(names: list[str]) -> str:
    encoded = json.dumps(
        names,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DenseParameterCatalog:
    by_name: dict[str, tuple[tuple[object, ...], int]]
    aliases: dict[tuple[object, ...], tuple[str, ...]]
    total_unique_elements: int


def _storage_identity(value: object) -> tuple[object, ...]:
    """Identify shared tensor storage even when wrappers are distinct objects."""

    untyped_storage = getattr(value, "untyped_storage", None)
    if callable(untyped_storage):
        try:
            storage = untyped_storage()
            storage_handle = int(getattr(storage, "_cdata", 0))
            if storage_handle > 0:
                return ("torch_storage", storage_handle)
            data_ptr = int(storage.data_ptr())
            storage_bytes = int(storage.nbytes())
            if data_ptr > 0 and storage_bytes > 0:
                return (
                    "torch_storage_address",
                    str(getattr(value, "device", "")),
                    data_ptr,
                    storage_bytes,
                )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return ("python_object", id(value))


def _bitsandbytes_type_contract(bits: int) -> tuple[type[Any], type[Any]]:
    try:
        bnb_nn = importlib.import_module("bitsandbytes.nn")
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is required to authenticate packed module types"
        ) from exc
    if bits == 4:
        names = ("Linear4bit", "Params4bit")
    elif bits == 8:
        names = ("Linear8bitLt", "Int8Params")
    else:
        raise ValueError("bits must be 4 or 8")
    module_type = getattr(bnb_nn, names[0], None)
    weight_type = getattr(bnb_nn, names[1], None)
    if not isinstance(module_type, type) or not isinstance(weight_type, type):
        raise RuntimeError("bitsandbytes packed type contract is unavailable")
    return module_type, weight_type


def dense_parameter_catalog(model: Any) -> DenseParameterCatalog:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("dense baseline does not expose named parameters")
    try:
        entries = list(named_parameters(remove_duplicate=False))
    except TypeError as exc:
        raise RuntimeError(
            "dense baseline cannot expose aliases for logical coverage"
        ) from exc
    if not entries:
        raise RuntimeError("dense baseline has no named parameters")

    by_name: dict[str, tuple[tuple[object, ...], int]] = {}
    aliases: dict[tuple[object, ...], list[str]] = {}
    unique_elements: dict[tuple[object, ...], int] = {}
    for name, parameter in entries:
        if not isinstance(name, str) or not name or name in by_name:
            raise RuntimeError("dense baseline parameter names are not canonical")
        numel = int(parameter.numel())
        if numel <= 0:
            raise RuntimeError(f"dense baseline parameter is empty: {name}")
        marker = _storage_identity(parameter)
        by_name[name] = (marker, numel)
        aliases.setdefault(marker, []).append(name)
        previous = unique_elements.setdefault(marker, numel)
        if previous != numel:
            raise RuntimeError("tied dense parameters disagree on logical size")
    total = sum(unique_elements.values())
    if total <= 0:
        raise RuntimeError("dense baseline logical parameter total is not positive")
    return DenseParameterCatalog(
        by_name=by_name,
        aliases={marker: tuple(sorted(names)) for marker, names in aliases.items()},
        total_unique_elements=total,
    )


def inspect_bitsandbytes_modules(
    model: Any, *, bits: int | None = None
) -> dict[str, Any]:
    if bits not in {4, 8}:
        raise ValueError("bits must be 4 or 8")
    assert bits is not None
    expected_module_type, expected_weight_type = _bitsandbytes_type_contract(bits)
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise RuntimeError("model does not expose named modules")

    module_names: list[str] = []
    types: set[str] = set()
    storage_parameter_ids: set[tuple[object, ...]] = set()
    storage_elements = 0
    for module_name, module in named_modules():
        if not isinstance(module, expected_module_type):
            continue
        module_type = type(module)
        fqcn = f"{module_type.__module__}.{module_type.__name__}"
        if not isinstance(module_name, str) or not module_name:
            raise RuntimeError("packed bitsandbytes module has no canonical name")
        module_names.append(module_name)
        types.add(fqcn)
        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, expected_weight_type):
            raise RuntimeError(
                f"packed bitsandbytes module has no packed backend weight: {module_name}"
            )
        marker = _storage_identity(weight)
        if marker in storage_parameter_ids:
            raise RuntimeError("packed bitsandbytes modules share one storage weight")
        storage_parameter_ids.add(marker)
        storage_elements += int(weight.numel())
    module_names = sorted(module_names)
    if not module_names:
        raise RuntimeError("no bitsandbytes packed linear modules were observed")
    if len(module_names) != len(set(module_names)):
        raise RuntimeError("packed bitsandbytes module names are ambiguous")
    if storage_elements <= 0:
        raise RuntimeError("packed bitsandbytes storage element count is not positive")
    return {
        "count": len(module_names),
        "names": module_names,
        "names_sha256": canonical_names_sha256(module_names),
        "types": sorted(types),
        "packed_weight_storage_elements": storage_elements,
    }


def logical_coverage_from_inventory(
    catalog: DenseParameterCatalog,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    module_names = inventory.get("names")
    if (
        not isinstance(module_names, list)
        or not module_names
        or any(not isinstance(name, str) or not name for name in module_names)
        or module_names != sorted(set(module_names))
    ):
        raise RuntimeError("packed module names are not canonical")
    weight_names = [f"{name}.weight" for name in module_names]
    logical_elements = 0
    observed_markers: set[tuple[object, ...]] = set()
    for name in weight_names:
        record = catalog.by_name.get(name)
        if record is None:
            raise RuntimeError(f"packed module has no dense baseline weight: {name}")
        marker, numel = record
        aliases = catalog.aliases[marker]
        if len(aliases) != 1:
            raise RuntimeError(
                f"packed module maps to tied or ambiguous dense weight: {name}"
            )
        if marker in observed_markers:
            raise RuntimeError("packed modules map to the same dense baseline weight")
        observed_markers.add(marker)
        logical_elements += numel
    payload = {
        "basis": LOGICAL_COVERAGE_BASIS,
        "weight_tensor_names": weight_names,
        "weight_tensor_names_sha256": canonical_names_sha256(weight_names),
        "weight_tensor_count": len(weight_names),
        "parameter_elements": logical_elements,
        "total_unique_parameter_elements": catalog.total_unique_elements,
    }
    require_logical_coverage(payload)
    return payload


def require_logical_coverage(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _LOGICAL_COVERAGE_FIELDS:
        raise ValueError("logical coverage has missing or unsupported fields")
    names = value.get("weight_tensor_names")
    if (
        value.get("basis") != LOGICAL_COVERAGE_BASIS
        or not isinstance(names, list)
        or not names
        or any(
            not isinstance(name, str) or not name.endswith(".weight") for name in names
        )
        or names != sorted(set(names))
    ):
        raise ValueError("logical coverage weight tensor names are invalid")
    if value.get("weight_tensor_names_sha256") != canonical_names_sha256(names):
        raise ValueError("logical coverage weight tensor names digest mismatch")
    count = value.get("weight_tensor_count")
    elements = value.get("parameter_elements")
    total = value.get("total_unique_parameter_elements")
    for field, item in (
        ("weight_tensor_count", count),
        ("parameter_elements", elements),
        ("total_unique_parameter_elements", total),
    ):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"logical coverage {field} must be a positive integer")
    count = cast(int, count)
    elements = cast(int, elements)
    total = cast(int, total)
    if count != len(names):
        raise ValueError("logical coverage weight tensor count mismatch")
    if elements > total:
        raise ValueError("logical coverage elements exceed dense parameter total")
    return value


def require_inventory_runtime_facts(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("packed runtime facts must be an object")
    names = value.get("quantized_module_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or names != sorted(set(names))
    ):
        raise ValueError("quantized module names are invalid")
    if value.get("quantized_module_names_sha256") != canonical_names_sha256(names):
        raise ValueError("quantized module names digest mismatch")
    count = value.get("quantized_module_count")
    storage = value.get("packed_weight_storage_elements")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(names):
        raise ValueError("quantized module count does not match module names")
    if isinstance(storage, bool) or not isinstance(storage, int) or storage <= 0:
        raise ValueError("packed weight storage elements must be positive")
    types = value.get("quantized_module_types")
    if (
        not isinstance(types, list)
        or not types
        or any(not isinstance(name, str) or not name for name in types)
        or types != sorted(set(types))
    ):
        raise ValueError("quantized module types are invalid")
    return value


def require_inventory_logical_binding(
    runtime_facts: object, logical_coverage: object
) -> None:
    runtime = require_inventory_runtime_facts(runtime_facts)
    logical = require_logical_coverage(logical_coverage)
    expected_names = [f"{name}.weight" for name in runtime["quantized_module_names"]]
    if logical["weight_tensor_names"] != expected_names:
        raise ValueError("logical weight tensors do not match packed module names")
    if logical["weight_tensor_count"] != runtime["quantized_module_count"]:
        raise ValueError("logical weight tensor count does not match packed modules")
