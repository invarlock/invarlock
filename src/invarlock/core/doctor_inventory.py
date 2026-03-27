from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any


@dataclass(frozen=True)
class DoctorInventoryRow:
    name: str
    origin: str
    mode: str
    backend: str | None
    version: str | None
    status: str
    enable: str


@dataclass(frozen=True)
class DoctorDatasetRow:
    provider: str
    network: str
    status: str
    params: str


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def build_adapter_inventory_rows(
    registry: Any,
    *,
    has_cuda: bool,
    is_linux: bool,
    find_spec_safe: Callable[[str], object | None],
    bitsandbytes_runtime_ready: bool,
) -> list[DoctorInventoryRow]:
    rows: list[DoctorInventoryRow] = []
    transformers_version = _package_version("transformers")

    for name in registry.list_adapters():
        info = registry.get_plugin_info(name, "adapters")
        module = str(info.get("module") or "")
        support = (
            "auto"
            if module.startswith("invarlock.adapters") and name in {"hf_auto"}
            else ("core" if module.startswith("invarlock.adapters") else "optional")
        )
        origin = "core" if support in {"core", "auto"} else "plugin"
        mode = "auto-matcher" if support == "auto" else "adapter"

        backend: str | None = None
        version: str | None = None
        status = "ready"
        enable = ""

        if name in {"hf_causal", "hf_mlm", "hf_seq2seq", "hf_auto"}:
            backend = "transformers"
            version = transformers_version
        elif name == "hf_gptq":
            backend = "auto-gptq"
        elif name == "hf_awq":
            backend = "autoawq"
        elif name == "hf_bnb":
            backend = "bitsandbytes"

        if support == "optional":
            present = (
                find_spec_safe((backend or "").replace("-", "_")) is not None
                if backend
                else False
            )
            if not present:
                status = "needs_extra"
                hint = {
                    "hf_gptq": "invarlock[gptq]",
                    "hf_awq": "invarlock[awq]",
                    "hf_bnb": "invarlock[gpu]",
                }.get(name)
                if hint:
                    enable = f"pip install '{hint}'"

        if backend in {"auto-gptq", "autoawq"} and not is_linux:
            status = "unsupported"
            enable = "Linux-only"
        if (
            backend == "bitsandbytes"
            and find_spec_safe("bitsandbytes") is not None
            and not bitsandbytes_runtime_ready
        ):
            status = "unsupported"
            if has_cuda:
                enable = "bitsandbytes unavailable on this host"
            else:
                enable = "Requires CUDA or a compatible bitsandbytes runtime"

        rows.append(
            DoctorInventoryRow(
                name=name,
                origin=origin,
                mode=mode,
                backend=backend,
                version=version,
                status=status,
                enable=enable,
            )
        )
    return rows


def build_generic_inventory_rows(
    registry: Any,
    *,
    kind: str,
    check_plugin_extras: Callable[[str, str], str],
) -> list[DoctorInventoryRow]:
    names = registry.list_guards() if kind == "guards" else registry.list_edits()
    rows: list[DoctorInventoryRow] = []
    for name in names:
        info = registry.get_plugin_info(name, kind)
        module = str(info.get("module") or "")
        origin = "core" if module.startswith(f"invarlock.{kind}") else "plugin"
        mode = "guard" if kind == "guards" else "edit"
        status = "ready"
        enable = ""
        try:
            extras = check_plugin_extras(name, kind)
        except Exception:
            extras = ""
        if isinstance(extras, str) and extras.startswith("⚠️") and "missing" in extras:
            status = "needs_extra"
            hint = extras.split("missing", 1)[-1].strip()
            if hint:
                enable = f"pip install '{hint}'"
        rows.append(
            DoctorInventoryRow(
                name=name,
                origin=origin,
                mode=mode,
                backend=None,
                version=None,
                status=status,
                enable=enable,
            )
        )
    return rows


def summarize_inventory_rows(rows: list[DoctorInventoryRow]) -> dict[str, int]:
    return {
        "total": len(rows),
        "ready": sum(1 for row in rows if row.status == "ready"),
        "needs_extra": sum(1 for row in rows if row.status == "needs_extra"),
        "unsupported": sum(1 for row in rows if row.status == "unsupported"),
        "auto": sum(1 for row in rows if row.mode == "auto-matcher"),
    }


def build_dataset_inventory_rows(
    providers: list[str],
    *,
    provider_network: Mapping[str, str],
    provider_params: Mapping[str, str],
) -> list[DoctorDatasetRow]:
    def _network_label(name: str) -> str:
        value = (provider_network.get(name, "") or "").lower()
        if value == "cache":
            return "Cache/Net"
        if value == "yes":
            return "Yes"
        if value == "no":
            return "No"
        return "Unknown"

    return [
        DoctorDatasetRow(
            provider=name,
            network=_network_label(name),
            status="✓ Available",
            params=provider_params.get(name, "-"),
        )
        for name in providers
    ]


__all__ = [
    "DoctorDatasetRow",
    "DoctorInventoryRow",
    "build_adapter_inventory_rows",
    "build_dataset_inventory_rows",
    "build_generic_inventory_rows",
    "summarize_inventory_rows",
]
