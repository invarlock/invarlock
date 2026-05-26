"""Backend inventory sidecars for optional quantized adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

BACKEND_INVENTORY_SCHEMA = "invarlock/backend-inventory-v1"
BACKEND_INVENTORY_FILENAME = "backend_inventory.json"

_QUANTIZED_ADAPTER_BACKENDS = {
    "hf_bnb": "bitsandbytes",
    "hf_awq": "gptqmodel",
    "hf_gptq": "gptqmodel",
}

_VERSION_ERRORS = (PackageNotFoundError, OSError, RuntimeError, TypeError, ValueError)


def quantized_adapter_backend(adapter_name: str | None) -> str | None:
    return _QUANTIZED_ADAPTER_BACKENDS.get(str(adapter_name or "").strip().lower())


def _package_version(name: str) -> str | None:
    try:
        return pkg_version(name)
    except _VERSION_ERRORS:
        return None


def _quantization_config_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    plugins = report.get("plugins")
    adapter_meta = plugins.get("adapter") if isinstance(plugins, Mapping) else None
    if isinstance(adapter_meta, Mapping):
        quant_cfg = adapter_meta.get("quantization_config")
        if isinstance(quant_cfg, Mapping):
            return dict(quant_cfg)
    meta = report.get("meta")
    model_profile = meta.get("model_profile") if isinstance(meta, Mapping) else None
    quant_cfg = (
        model_profile.get("quantization_config")
        if isinstance(model_profile, Mapping)
        else None
    )
    return dict(quant_cfg) if isinstance(quant_cfg, Mapping) else {}


def build_backend_inventory_from_report(
    report: Mapping[str, Any],
    *,
    model: Any | None = None,
) -> dict[str, Any] | None:
    if not isinstance(report, Mapping):
        return None
    meta = report.get("meta")
    adapter_name = meta.get("adapter") if isinstance(meta, Mapping) else None
    adapter = str(adapter_name or "").strip()
    backend = quantized_adapter_backend(adapter)
    if backend is None:
        return None

    plugins = report.get("plugins")
    adapter_meta = plugins.get("adapter") if isinstance(plugins, Mapping) else None
    provenance = (
        adapter_meta.get("provenance") if isinstance(adapter_meta, Mapping) else None
    )
    backend_version = (
        provenance.get("version")
        if isinstance(provenance, Mapping) and provenance.get("version")
        else _package_version(backend)
    )

    return build_backend_inventory_for_adapter(
        adapter=adapter,
        backend_version=backend_version,
        quantization_config=_quantization_config_from_report(report),
        model=model,
    )


def build_backend_inventory_for_adapter(
    *,
    adapter: str | None,
    backend_version: str | None = None,
    quantization_config: Mapping[str, Any] | None = None,
    model: Any | None = None,
) -> dict[str, Any] | None:
    adapter_name = str(adapter or "").strip()
    backend = quantized_adapter_backend(adapter_name)
    if backend is None:
        return None

    module_inventory = _quantized_module_inventory(model, adapter=adapter_name)
    memory_footprint = _memory_footprint(model)

    return {
        "schema": BACKEND_INVENTORY_SCHEMA,
        "adapter": adapter_name,
        "backend": backend,
        "backend_version": backend_version or _package_version(backend),
        "transformers_version": _package_version("transformers"),
        "quantization_config": dict(quantization_config or {}),
        "quantized_module_count": module_inventory["count"],
        "quantized_module_types": module_inventory["types"],
        "device_map": "unknown",
        "memory_footprint": memory_footprint,
        "load_smoke": True,
        "inference_smoke": True,
    }


def _quantized_module_inventory(
    model: Any | None,
    *,
    adapter: str,
) -> dict[str, Any]:
    if model is None:
        return {"count": 0, "types": []}

    adapter_key = str(adapter or "").strip().lower()
    type_names: set[str] = set()
    count = 0
    modules_fn = getattr(model, "modules", None)
    if not callable(modules_fn):
        return {"count": 0, "types": []}

    for module in modules_fn():
        module_type = type(module)
        fqcn = f"{module_type.__module__}.{module_type.__name__}"
        normalized = fqcn.lower()
        is_quantized = False
        if adapter_key == "hf_bnb":
            is_quantized = "bitsandbytes" in normalized
        elif adapter_key == "hf_awq":
            is_quantized = (
                "awq" in normalized
                or "gptqmodel" in normalized
                or "qlinear" in normalized
                or "wqlinear" in normalized
            )
        elif adapter_key == "hf_gptq":
            is_quantized = "gptq" in normalized or "quantlinear" in normalized
        if not is_quantized:
            continue
        count += 1
        type_names.add(fqcn)
    return {"count": count, "types": sorted(type_names)}


def _memory_footprint(model: Any | None) -> dict[str, Any]:
    if model is None:
        return {"reported_bytes": 0, "method": "unknown"}
    get_memory_footprint = getattr(model, "get_memory_footprint", None)
    if callable(get_memory_footprint):
        try:
            return {
                "reported_bytes": int(get_memory_footprint()),
                "method": "get_memory_footprint",
            }
        except (RuntimeError, TypeError, ValueError):
            pass
    return {"reported_bytes": 0, "method": "unknown"}


def write_backend_inventory_sidecar(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    model: Any | None = None,
    inventory: Mapping[str, Any] | None = None,
) -> Path | None:
    inventory_payload = (
        dict(inventory)
        if isinstance(inventory, Mapping)
        else build_backend_inventory_from_report(report, model=model)
    )
    if inventory_payload is None:
        return None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sidecar_path = output_path / BACKEND_INVENTORY_FILENAME
    sidecar_path.write_text(
        json.dumps(inventory_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sidecar_path


__all__ = [
    "BACKEND_INVENTORY_FILENAME",
    "BACKEND_INVENTORY_SCHEMA",
    "build_backend_inventory_for_adapter",
    "build_backend_inventory_from_report",
    "quantized_adapter_backend",
    "write_backend_inventory_sidecar",
]
