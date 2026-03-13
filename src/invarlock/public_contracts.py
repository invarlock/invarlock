from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACTS_ROOT = Path(__file__).resolve().parents[2] / "contracts"


def contract_path(filename: str) -> Path:
    return CONTRACTS_ROOT / filename


def contract_relpath(filename: str) -> str:
    return f"contracts/{filename}"


def load_json_contract(filename: str) -> Any:
    path = contract_path(filename)
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_load(filename: str, default: Any) -> Any:
    try:
        return load_json_contract(filename)
    except Exception:
        return default


def load_support_matrix() -> dict[str, Any]:
    data = _safe_load("support_matrix.json", {"format_version": "support-matrix-v1"})
    if isinstance(data, dict):
        data.setdefault("lanes", [])
        return data
    return {"format_version": "support-matrix-v1", "lanes": []}


def load_adapter_capabilities() -> dict[str, Any]:
    data = _safe_load(
        "adapter_capabilities.json", {"format_version": "adapter-capabilities-v1"}
    )
    if isinstance(data, dict):
        data.setdefault("adapters", [])
        return data
    return {"format_version": "adapter-capabilities-v1", "adapters": []}


def load_plugin_compatibility() -> dict[str, Any]:
    data = _safe_load(
        "plugin_compatibility.json", {"format_version": "plugin-compatibility-v1"}
    )
    if isinstance(data, dict):
        return data
    return {"format_version": "plugin-compatibility-v1"}


def load_policy_pack_schema() -> dict[str, Any]:
    data = _safe_load("policy_pack.schema.json", {})
    return data if isinstance(data, dict) else {}


def load_proof_pack_manifest_schema() -> dict[str, Any]:
    data = _safe_load("proof_pack_manifest.schema.json", {})
    return data if isinstance(data, dict) else {}


def support_lanes() -> list[dict[str, Any]]:
    lanes = load_support_matrix().get("lanes", [])
    return [lane for lane in lanes if isinstance(lane, dict)]


def support_lane_by_id(lane_id: str) -> dict[str, Any] | None:
    for lane in support_lanes():
        if lane.get("lane_id") == lane_id:
            return lane
    return None


def adapter_capability_map() -> dict[str, dict[str, Any]]:
    payload = load_adapter_capabilities().get("adapters", [])
    mapping: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("adapter")
        if isinstance(name, str) and name:
            mapping[name] = item
    return mapping


def adapter_capability(adapter_name: str) -> dict[str, Any] | None:
    return adapter_capability_map().get(adapter_name)


def published_basis_lanes() -> list[dict[str, Any]]:
    return [
        lane
        for lane in support_lanes()
        if lane.get("support_tier") == "published_basis"
    ]


def contract_reference(filename: str) -> dict[str, Any]:
    ref: dict[str, Any] = {"path": contract_relpath(filename)}
    payload = _safe_load(filename, None)
    if isinstance(payload, dict):
        if isinstance(payload.get("format_version"), str):
            ref["format_version"] = payload["format_version"]
        if isinstance(payload.get("format"), str):
            ref["format"] = payload["format"]
        if isinstance(payload.get("core_abi"), str):
            ref["core_abi"] = payload["core_abi"]
        if isinstance(payload.get("match_policy"), str):
            ref["match_policy"] = payload["match_policy"]
    return ref


def contract_catalog() -> dict[str, Any]:
    return {
        "support_matrix": contract_reference("support_matrix.json"),
        "adapter_capabilities": contract_reference("adapter_capabilities.json"),
        "plugin_compatibility": contract_reference("plugin_compatibility.json"),
        "proof_pack_manifest": contract_reference("proof_pack_manifest.schema.json"),
        "policy_pack": contract_reference("policy_pack.schema.json"),
    }


__all__ = [
    "CONTRACTS_ROOT",
    "adapter_capability",
    "adapter_capability_map",
    "contract_catalog",
    "contract_path",
    "contract_reference",
    "contract_relpath",
    "load_adapter_capabilities",
    "load_json_contract",
    "load_plugin_compatibility",
    "load_policy_pack_schema",
    "load_proof_pack_manifest_schema",
    "load_support_matrix",
    "published_basis_lanes",
    "support_lane_by_id",
    "support_lanes",
]
