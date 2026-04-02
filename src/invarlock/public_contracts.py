from __future__ import annotations

import importlib.resources
import json
import os
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


def _fallback_contract_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("INVARLOCK_CONTRACTS_ROOT")
    if env_root:
        roots.append(Path(env_root))
    github_workspace = os.environ.get("GITHUB_WORKSPACE")
    if github_workspace:
        roots.append(Path(github_workspace) / "contracts")
    roots.append(Path.cwd() / "contracts")

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root in seen or root == CONTRACTS_ROOT:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def _ancestor_contract_roots(*, filename: str) -> list[Path]:
    roots: list[Path] = []
    anchors = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    seen: set[Path] = set()
    for anchor in anchors:
        for parent in (anchor, *anchor.parents):
            candidate = parent / "contracts"
            if candidate in seen or candidate == CONTRACTS_ROOT:
                continue
            seen.add(candidate)
            if (candidate / filename).is_file():
                roots.append(candidate)
    return roots


def load_json_contract(filename: str) -> Any:
    path = contract_path(filename)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        return json.loads(
            PACKAGE_CONTRACTS_ROOT.joinpath(filename).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, NotADirectoryError, OSError):
        pass

    for root in _fallback_contract_roots():
        fallback_path = root / filename
        if fallback_path.is_file():
            return json.loads(fallback_path.read_text(encoding="utf-8"))

    for root in _ancestor_contract_roots(filename=filename):
        fallback_path = root / filename
        if fallback_path.is_file():
            return json.loads(fallback_path.read_text(encoding="utf-8"))

    raise FileNotFoundError(filename)


def _load_contract_or_raise(filename: str) -> Any:
    try:
        return load_json_contract(filename)
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        NotADirectoryError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ContractLoadError(filename, reason=str(exc)) from exc


def _load_object_contract_or_raise(filename: str) -> dict[str, Any]:
    data = _load_contract_or_raise(filename)
    if not isinstance(data, dict):
        raise ContractLoadError(
            filename,
            reason=f"expected JSON object, got {type(data).__name__}",
        )
    return data


def load_support_matrix() -> dict[str, Any]:
    data = _load_object_contract_or_raise("support_matrix.json")
    data.setdefault("lanes", [])
    return data


def load_adapter_capabilities() -> dict[str, Any]:
    data = _load_object_contract_or_raise("adapter_capabilities.json")
    data.setdefault("adapters", [])
    return data


def load_model_family_catalog() -> dict[str, Any]:
    data = _load_object_contract_or_raise("model_family_catalog.json")
    data.setdefault("declared_support", [])
    data.setdefault("implemented_coverage", [])
    data.setdefault("usage_only", [])
    data.setdefault("recommended_additions", [])
    return data


def load_plugin_compatibility() -> dict[str, Any]:
    return _load_object_contract_or_raise("plugin_compatibility.json")


def load_policy_pack_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("policy_pack.schema.json")


def load_proof_pack_manifest_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("proof_pack_manifest.schema.json")


def load_runtime_manifest_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("runtime_manifest.schema.json")


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
        payload = _load_object_contract_or_raise(filename)
    except ContractLoadError as exc:
        ref["load_error"] = exc.reason
        return ref
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
