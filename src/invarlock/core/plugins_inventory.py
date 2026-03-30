from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

PluginCategory = Literal["adapters", "guards", "edits"]
InventoryRow = dict[str, Any]


def is_minimal_plugins_view(env_value: str | None) -> bool:
    value = str(env_value or "").strip().lower()
    return value not in ("", "0", "false", "no")


def detect_cuda_available(torch_module: Any) -> bool:
    try:
        cuda = getattr(torch_module, "cuda", None)
        return bool(cuda and cuda.is_available())
    except Exception:
        return False


def _module_origin(module: str) -> str:
    return "builtin" if module.startswith("invarlock.") else "third_party"


def _adapter_backend_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    backend_name = row.get("backend") or ""
    backend_ver = row.get("backend_version")
    if not backend_name:
        return None
    backend_obj: dict[str, Any] = {
        "name": backend_name,
        "present": bool(row.get("backend_present")),
    }
    if backend_ver:
        backend_obj["version"] = backend_ver
    return backend_obj


def gather_adapter_inventory_rows(
    *,
    registry: Any,
    minimal: bool,
    has_cuda: bool,
    is_linux: bool,
    extras_checker: Callable[[str, str], str],
    provenance_extractor: Callable[[str], Any],
    bitsandbytes_runtime_available: Callable[[], bool],
) -> list[InventoryRow]:
    names = list(registry.list_adapters())
    if minimal:
        names = [
            name
            for name in names
            if str(registry.get_plugin_info(name, "adapters").get("module") or "").startswith(
                "invarlock.plugins"
            )
        ]

    rows: list[InventoryRow] = []
    for name in names:
        info = registry.get_plugin_info(name, "adapters")
        module = str(info.get("module") or "")
        entry = info.get("entry_point")
        if module.startswith("invarlock.adapters"):
            support = "auto" if name in {"hf_auto"} else "core"
        else:
            support = "optional"

        origin = "core" if module.startswith("invarlock.adapters") else "plugin"
        mode = "auto-matcher" if support == "auto" else "adapter"
        backend_name = ""
        backend_version = None
        present = False
        backend_present = False
        try:
            provenance = provenance_extractor(name)
            backend_name = provenance.library or ""
            backend_version = provenance.version
            present = backend_version is not None
            backend_present = present
        except Exception:
            pass

        status = "ready"
        enable = ""
        if support == "optional" and not present:
            status = "needs_extra"

        if backend_name in {"auto-gptq", "autoawq"} and not is_linux:
            status = "unsupported"
            enable = "Linux-only"

        try:
            extras_status = extras_checker(name, "adapters")
        except Exception:
            extras_status = ""
        if (
            support == "optional"
            and extras_status.startswith("⚠️")
            and "missing" in extras_status
        ):
            status = "needs_extra"
            hint = extras_status.split("missing", 1)[-1].strip()
            if hint:
                enable = f"pip install '{hint}'"

        if backend_name == "bitsandbytes" and present:
            backend_present = bitsandbytes_runtime_available()
            if not backend_present:
                status = "unsupported"
                if has_cuda:
                    enable = "bitsandbytes unavailable on this host"
                else:
                    enable = "Requires CUDA or a compatible bitsandbytes runtime"

        extra_hint = {
            "hf_gptq": "invarlock[gptq]",
            "hf_awq": "invarlock[awq]",
            "hf_bnb": "invarlock[gpu]",
        }.get(name)
        if status == "needs_extra" and extra_hint:
            enable = f"pip install '{extra_hint}'"

        rows.append(
            {
                "name": name,
                "backend": backend_name,
                "backend_version": backend_version,
                "backend_present": backend_present,
                "support": support,
                "origin": origin,
                "mode": mode,
                "status": status,
                "enable": enable,
                "module": module,
                "entry_point": entry,
            }
        )

    rows.sort(
        key=lambda row: (
            {"needs_extra": 0, "partial": 1, "ready": 2}.get(
                str(row["status"]), 3
            ),
            {"optional": 0, "core": 1, "auto": 2}.get(str(row["support"]), 3),
            str(row["name"]),
        )
    )
    return rows


def gather_generic_inventory_rows(
    *,
    registry: Any,
    plugin_type: PluginCategory,
    extras_checker: Callable[[str, str], str],
) -> list[InventoryRow]:
    names = (
        registry.list_guards() if plugin_type == "guards" else registry.list_edits()
    )
    rows: list[InventoryRow] = []
    for name in names:
        info = registry.get_plugin_info(name, plugin_type)
        module = str(info.get("module") or "")
        entry = info.get("entry_point")
        support = "core" if module.startswith(f"invarlock.{plugin_type}") else "optional"
        origin = "core" if support == "core" else "plugin"
        mode = "guard" if plugin_type == "guards" else "edit"
        extras_status = extras_checker(name, plugin_type)
        status = "ready"
        enable = ""
        if extras_status.startswith("⚠️") and "missing" in extras_status:
            status = "needs_extra"
            hint = extras_status.split("missing", 1)[-1].strip()
            if hint:
                enable = f"pip install '{hint}'"
        rows.append(
            {
                "name": name,
                "backend": None,
                "backend_version": None,
                "support": support,
                "origin": origin,
                "mode": mode,
                "status": status,
                "enable": enable,
                "module": module,
                "entry_point": entry,
            }
        )

    rows.sort(
        key=lambda row: (
            {"core": 0, "optional": 1}.get(str(row["support"]), 2),
            {"needs_extra": 0, "ready": 1}.get(str(row["status"]), 2),
            str(row["name"]),
        )
    )
    return rows


def filter_inventory_rows(
    rows: list[InventoryRow],
    only: str | None,
) -> list[InventoryRow]:
    if not only:
        return rows
    mode = only.strip().lower()
    if mode == "missing":
        return [row for row in rows if row["status"] == "needs_extra"]
    if mode == "ready":
        return [row for row in rows if row["status"] == "ready"]
    if mode == "core":
        return [row for row in rows if row["support"] == "core"]
    if mode == "optional":
        return [row for row in rows if row["support"] == "optional"]
    return rows


def adapter_inventory_json_items(rows: list[InventoryRow]) -> list[dict[str, Any]]:
    return [
        {
            "name": row.get("name"),
            "kind": "adapter",
            "module": row.get("module"),
            "entry_point": row.get("entry_point"),
            "origin": _module_origin(str(row.get("module") or "")),
            "status": row.get("status"),
            "backend": _adapter_backend_payload(row),
            "capability": row.get("capability"),
        }
        for row in rows
    ]


def generic_inventory_json_items(
    rows: list[InventoryRow],
    *,
    kind: Literal["guards", "edits"],
) -> list[dict[str, Any]]:
    item_kind = "guard" if kind == "guards" else "edit"
    return [
        {
            "name": row.get("name"),
            "kind": item_kind,
            "module": row.get("module"),
            "entry_point": row.get("entry_point"),
            "origin": _module_origin(str(row.get("module") or "")),
        }
        for row in rows
    ]


def combined_plugins_json_items(
    *,
    adapter_rows: list[InventoryRow],
    guard_rows: list[InventoryRow],
    edit_rows: list[InventoryRow],
) -> list[dict[str, Any]]:
    return (
        [
            {
                "name": row.get("name"),
                "kind": "adapter",
                "module": row.get("module"),
                "entry_point": row.get("entry_point"),
                "origin": _module_origin(str(row.get("module") or "")),
                "backend": _adapter_backend_payload(row),
            }
            for row in adapter_rows
        ]
        + generic_inventory_json_items(guard_rows, kind="guards")
        + generic_inventory_json_items(edit_rows, kind="edits")
    )


def dataset_inventory_json_items(
    providers: list[str],
    providers_map: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for provider_name in providers:
        provider_cls = providers_map.get(provider_name)
        items.append(
            {
                "name": provider_name,
                "module": getattr(provider_cls, "__module__", "unknown"),
                "status": "available",
            }
        )
    return items


__all__ = [
    "adapter_inventory_json_items",
    "combined_plugins_json_items",
    "dataset_inventory_json_items",
    "detect_cuda_available",
    "filter_inventory_rows",
    "gather_adapter_inventory_rows",
    "gather_generic_inventory_rows",
    "generic_inventory_json_items",
    "is_minimal_plugins_view",
]
