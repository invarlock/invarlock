from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any

CONTRACTS_ROOT = Path(__file__).resolve().parents[2] / "contracts"
PACKAGE_CONTRACTS_ROOT = importlib.resources.files("invarlock").joinpath(
    "_data", "contracts"
)


class ContractLoadError(RuntimeError):
    """Raised when a shipped contract cannot be loaded from either contract root."""

    def __init__(self, filename: str, *, reason: str) -> None:
        super().__init__(f"Failed to load contract '{filename}': {reason}")
        self.filename = filename
        self.reason = reason


def contract_path(filename: str) -> Path:
    return CONTRACTS_ROOT / filename


def contract_relpath(filename: str) -> str:
    return f"contracts/{filename}"


def load_json_contract(filename: str) -> Any:
    path = contract_path(filename)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(
        PACKAGE_CONTRACTS_ROOT.joinpath(filename).read_text(encoding="utf-8")
    )


def _load_contract_or_raise(filename: str) -> Any:
    try:
        return load_json_contract(filename)
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        NotADirectoryError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ContractLoadError(filename, reason=str(exc)) from exc


def load_support_matrix() -> dict[str, Any]:
    data = _load_contract_or_raise("support_matrix.json")
    if isinstance(data, dict):
        data.setdefault("lanes", [])
        return data
    return {"format_version": "support-matrix-v1", "lanes": []}


def load_adapter_capabilities() -> dict[str, Any]:
    data = _load_contract_or_raise("adapter_capabilities.json")
    if isinstance(data, dict):
        data.setdefault("adapters", [])
        return data
    return {"format_version": "adapter-capabilities-v1", "adapters": []}


def load_model_family_catalog() -> dict[str, Any]:
    data = _load_contract_or_raise("model_family_catalog.json")
    if isinstance(data, dict):
        data.setdefault("declared_support", [])
        data.setdefault("implemented_coverage", [])
        data.setdefault("usage_only", [])
        data.setdefault("recommended_additions", [])
        return data
    return {
        "format_version": "model-family-catalog-v1",
        "declared_support": [],
        "implemented_coverage": [],
        "usage_only": [],
        "recommended_additions": [],
    }


def load_plugin_compatibility() -> dict[str, Any]:
    data = _load_contract_or_raise("plugin_compatibility.json")
    if isinstance(data, dict):
        return data
    return {"format_version": "plugin-compatibility-v1"}


def load_policy_pack_schema() -> dict[str, Any]:
    data = _load_contract_or_raise("policy_pack.schema.json")
    return data if isinstance(data, dict) else {}


def load_proof_pack_manifest_schema() -> dict[str, Any]:
    data = _load_contract_or_raise("proof_pack_manifest.schema.json")
    return data if isinstance(data, dict) else {}


def load_runtime_manifest_schema() -> dict[str, Any]:
    data = _load_contract_or_raise("runtime_manifest.schema.json")
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
    try:
        payload = _load_contract_or_raise(filename)
    except ContractLoadError as exc:
        ref["load_error"] = exc.reason
        return ref
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
        "model_family_catalog": contract_reference("model_family_catalog.json"),
        "adapter_capabilities": contract_reference("adapter_capabilities.json"),
        "plugin_compatibility": contract_reference("plugin_compatibility.json"),
        "runtime_manifest": contract_reference("runtime_manifest.schema.json"),
        "proof_pack_manifest": contract_reference("proof_pack_manifest.schema.json"),
        "policy_pack": contract_reference("policy_pack.schema.json"),
    }


__all__ = [
    "CONTRACTS_ROOT",
    "PACKAGE_CONTRACTS_ROOT",
    "ContractLoadError",
    "adapter_capability",
    "adapter_capability_map",
    "contract_catalog",
    "contract_path",
    "contract_reference",
    "contract_relpath",
    "load_adapter_capabilities",
    "load_json_contract",
    "load_model_family_catalog",
    "load_plugin_compatibility",
    "load_policy_pack_schema",
    "load_proof_pack_manifest_schema",
    "load_runtime_manifest_schema",
    "load_support_matrix",
    "published_basis_lanes",
    "support_lane_by_id",
    "support_lanes",
]
