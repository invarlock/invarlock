from __future__ import annotations

import importlib.resources
import json
import os
import sys
from pathlib import Path
from typing import Any

CONTRACTS_ROOT = Path(__file__).resolve().parents[2] / "contracts"
PACKAGE_CONTRACTS_ROOT = importlib.resources.files("invarlock").joinpath(
    "_data", "contracts"
)
PACKAGE_PUBLIC_EVIDENCE_ROOT = importlib.resources.files("invarlock").joinpath(
    "_data", "public_evidence"
)

REPORT_SCHEMA_VERSION = "v1"
EVIDENCE_PACK_FORMAT_VERSION = "evidence-pack-v1"
EVIDENCE_CATALOG_FORMAT_VERSION = "invarlock/evidence-catalog-v1"
EVIDENCE_CATALOG_VALIDATE_OUTPUT_FORMAT_VERSION = "evidence-catalog-validate-v1"
PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION = "public-evidence-index-v2"
RUNTIME_MANIFEST_CONTRACT_VERSION = "runtime-manifest-v1"
DOCTOR_OUTPUT_FORMAT_VERSION = "doctor-v1"
PLUGINS_OUTPUT_FORMAT_VERSION = "plugins-v2"
VERIFY_OUTPUT_FORMAT_VERSION = "verify-v1"
RUNTIME_VERIFY_OUTPUT_FORMAT_VERSION = "runtime-verify-v1"
POLICY_PACK_VERIFY_OUTPUT_FORMAT_VERSION = "policy-pack-verify-v1"
EVIDENCE_PACK_VERIFY_OUTPUT_FORMAT_VERSION = "evidence-pack-verify-v1"
CLI_STABILITY_POLICY_VERSION = "cli-stability-v1"
ADAPTER_SUPPORT_TIER_POLICY_VERSION = "adapter-support-tiers-v2"
MODEL_CLASSIFICATION_FORMAT_VERSION = "model-classification-v2"
RUNTIME_PROVIDER_ABI_VERSION = "1"
RUNTIME_PROVIDER_CAPABILITIES_FORMAT_VERSION = "runtime-provider-capabilities-v1"
MODEL_ARTIFACT_IDENTITY_FORMAT_VERSION = "invarlock/model-artifact-identity-v1"
RUNTIME_PROVIDER_RECEIPT_FORMAT_VERSION = "invarlock/runtime-provider-receipt-v1"
RUNTIME_SCORING_OBSERVATION_FORMAT_VERSION = "invarlock/runtime-scoring-observation-v1"

STABLE_CLI_JSON_SURFACES: dict[str, str] = {
    "invarlock doctor --json": DOCTOR_OUTPUT_FORMAT_VERSION,
    "invarlock verify --json": VERIFY_OUTPUT_FORMAT_VERSION,
    "invarlock advanced runtime-verify --json": RUNTIME_VERIFY_OUTPUT_FORMAT_VERSION,
    "invarlock advanced plugins list --json": PLUGINS_OUTPUT_FORMAT_VERSION,
    "invarlock advanced plugins adapters --json": PLUGINS_OUTPUT_FORMAT_VERSION,
    "invarlock advanced policy verify --json": POLICY_PACK_VERIFY_OUTPUT_FORMAT_VERSION,
    "invarlock advanced evidence-pack verify --json": (
        EVIDENCE_PACK_VERIFY_OUTPUT_FORMAT_VERSION
    ),
    "invarlock advanced evidence-catalog validate --json": (
        EVIDENCE_CATALOG_VALIDATE_OUTPUT_FORMAT_VERSION
    ),
}


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
    bundle_root = getattr(sys, "_MEIPASS", "")
    if bundle_root:
        roots.append(Path(bundle_root) / "contracts")
        roots.append(Path(bundle_root) / "invarlock" / "_data" / "contracts")
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
    argv0 = Path(sys.argv[0]).resolve().parent if sys.argv else None
    executable = Path(sys.executable).resolve().parent if sys.executable else None
    if argv0 is not None:
        anchors.append(argv0)
    if executable is not None:
        anchors.append(executable)
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
    data.setdefault(
        "maintained_catalog_candidates_text_le_14b",
        {
            "format_version": "maintained-catalog-candidates-text-le-14b-v1",
            "candidates": [],
        },
    )
    data.setdefault("recommended_additions", [])
    return data


def load_model_classification() -> dict[str, Any]:
    data = _load_object_contract_or_raise("model_classification.json")
    data.setdefault("entries", [])
    data.setdefault("blocked_named_checkpoints", [])
    return data


def load_public_evidence_index() -> dict[str, Any]:
    try:
        payload = json.loads(
            PACKAGE_PUBLIC_EVIDENCE_ROOT.joinpath(
                "catalog_evidence_index.json"
            ).read_text(encoding="utf-8")
        )
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        NotADirectoryError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ContractLoadError("catalog_evidence_index.json", reason=str(exc)) from exc
    if not isinstance(payload, dict):
        raise ContractLoadError(
            "catalog_evidence_index.json",
            reason=f"expected JSON object, got {type(payload).__name__}",
        )
    if payload.get("format_version") != PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION:
        raise ContractLoadError(
            "catalog_evidence_index.json",
            reason=(f"format_version must be {PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION}"),
        )
    payload.setdefault("entries", [])
    payload.setdefault("carrier_policy", {})
    return payload


def load_plugin_compatibility() -> dict[str, Any]:
    return _load_object_contract_or_raise("plugin_compatibility.json")


def load_policy_pack_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("policy_pack.schema.json")


def load_evidence_pack_manifest_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("evidence_pack_manifest.schema.json")


def load_evidence_catalog() -> dict[str, Any]:
    data = _load_object_contract_or_raise("evidence_catalog_v1.json")
    if data.get("format_version") != EVIDENCE_CATALOG_FORMAT_VERSION:
        raise ContractLoadError(
            "evidence_catalog_v1.json",
            reason=f"format_version must be {EVIDENCE_CATALOG_FORMAT_VERSION}",
        )
    data.setdefault("entries", [])
    return data


def load_runtime_manifest_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("runtime_manifest.schema.json")


def load_runtime_provider_capabilities_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("runtime_provider_capabilities.json")


def load_model_artifact_identity_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("model_artifact_identity.schema.json")


def load_runtime_provider_receipt_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("runtime_provider_receipt.schema.json")


def load_runtime_scoring_observation_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("runtime_scoring_observation.schema.json")


def load_verify_output_schema() -> dict[str, Any]:
    return _load_object_contract_or_raise("verify_output.schema.json")


def support_lanes() -> list[dict[str, Any]]:
    lanes = load_support_matrix().get("lanes", [])
    return [lane for lane in lanes if isinstance(lane, dict)]


def support_tiers() -> tuple[str, ...]:
    tiers = load_support_matrix().get("support_tiers", [])
    if not isinstance(tiers, list):
        return ()
    return tuple(tier for tier in tiers if isinstance(tier, str) and tier)


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


def maintained_catalog_lanes() -> list[dict[str, Any]]:
    return [
        lane
        for lane in support_lanes()
        if lane.get("support_tier") == "maintained_catalog"
    ]


def stable_cli_json_surfaces() -> dict[str, str]:
    return dict(STABLE_CLI_JSON_SURFACES)


def public_subcontract_catalog() -> dict[str, dict[str, Any]]:
    return {
        "report_schema": {
            "version": REPORT_SCHEMA_VERSION,
            "source": "invarlock.reporting.report_schema.REPORT_JSON_SCHEMA",
            "compatibility": "additive_within_v1",
        },
        "evidence_pack_format": {
            "version": EVIDENCE_PACK_FORMAT_VERSION,
            "source": "contracts/evidence_pack_manifest.schema.json",
            "compatibility": "strict_format_match",
        },
        "evidence_catalog": {
            "version": EVIDENCE_CATALOG_FORMAT_VERSION,
            "source": "contracts/evidence_catalog_v1.json",
            "compatibility": "closed_versioned_catalog",
        },
        "verifier_output": {
            "version": VERIFY_OUTPUT_FORMAT_VERSION,
            "source": "contracts/verify_output.schema.json",
            "compatibility": "additive_within_v1",
        },
        "runtime_manifest": {
            "version": RUNTIME_MANIFEST_CONTRACT_VERSION,
            "source": "contracts/runtime_manifest.schema.json",
            "compatibility": "strict_contract_version_match",
        },
        "runtime_provider_capabilities": {
            "version": RUNTIME_PROVIDER_CAPABILITIES_FORMAT_VERSION,
            "source": "contracts/runtime_provider_capabilities.json",
            "compatibility": "strict_format_and_abi_match",
        },
        "model_artifact_identity": {
            "version": MODEL_ARTIFACT_IDENTITY_FORMAT_VERSION,
            "source": "contracts/model_artifact_identity.schema.json",
            "compatibility": "closed_discriminated_variants",
        },
        "runtime_provider_receipt": {
            "version": RUNTIME_PROVIDER_RECEIPT_FORMAT_VERSION,
            "source": "contracts/runtime_provider_receipt.schema.json",
            "compatibility": "strict_format_and_abi_match",
        },
        "runtime_scoring_observation": {
            "version": RUNTIME_SCORING_OBSERVATION_FORMAT_VERSION,
            "source": "contracts/runtime_scoring_observation.schema.json",
            "compatibility": "closed_versioned_observation",
        },
        "cli_stability_policy": {
            "version": CLI_STABILITY_POLICY_VERSION,
            "source": "docs/reference/cli.md",
            "stable_json_surfaces": stable_cli_json_surfaces(),
        },
        "adapter_support_tiers": {
            "version": ADAPTER_SUPPORT_TIER_POLICY_VERSION,
            "source": "contracts/support_matrix.json",
            "tiers": list(support_tiers()),
        },
        "public_evidence_index": {
            "version": PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION,
            "source": ("invarlock/_data/public_evidence/catalog_evidence_index.json"),
            "compatibility": "generated_from_public_evidence_source_tree",
            "carrier_policy": load_public_evidence_index().get("carrier_policy", {}),
        },
    }


def contract_reference(filename: str) -> dict[str, Any]:
    ref: dict[str, Any] = {"path": contract_relpath(filename)}
    try:
        payload = _load_contract_or_raise(filename)
    except (ContractLoadError, KeyError):
        try:
            payload = load_json_contract(filename)
        except (
            FileNotFoundError,
            ModuleNotFoundError,
            NotADirectoryError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            ref["load_error"] = str(exc)
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
    elif isinstance(payload, list):
        ref["kind"] = "array"
        ref["item_count"] = len(payload)
    else:
        ref["kind"] = type(payload).__name__
    return ref


def contract_catalog() -> dict[str, Any]:
    return {
        "support_matrix": contract_reference("support_matrix.json"),
        "model_family_catalog": contract_reference("model_family_catalog.json"),
        "model_classification": contract_reference("model_classification.json"),
        "adapter_capabilities": contract_reference("adapter_capabilities.json"),
        "plugin_compatibility": contract_reference("plugin_compatibility.json"),
        "validation_keys": contract_reference("validation_keys.json"),
        "console_labels": contract_reference("console_labels.json"),
        "metric_kinds": contract_reference("metric_kinds.json"),
        "runtime_manifest": contract_reference("runtime_manifest.schema.json"),
        "runtime_provider_capabilities": contract_reference(
            "runtime_provider_capabilities.json"
        ),
        "model_artifact_identity": contract_reference(
            "model_artifact_identity.schema.json"
        ),
        "runtime_provider_receipt": contract_reference(
            "runtime_provider_receipt.schema.json"
        ),
        "runtime_scoring_observation": contract_reference(
            "runtime_scoring_observation.schema.json"
        ),
        "verify_output": contract_reference("verify_output.schema.json"),
        "evidence_pack_manifest": contract_reference(
            "evidence_pack_manifest.schema.json"
        ),
        "evidence_catalog": contract_reference("evidence_catalog_v1.json"),
        "policy_pack": contract_reference("policy_pack.schema.json"),
    }


__all__ = [
    "CONTRACTS_ROOT",
    "EVIDENCE_CATALOG_FORMAT_VERSION",
    "EVIDENCE_CATALOG_VALIDATE_OUTPUT_FORMAT_VERSION",
    "PACKAGE_CONTRACTS_ROOT",
    "PACKAGE_PUBLIC_EVIDENCE_ROOT",
    "ADAPTER_SUPPORT_TIER_POLICY_VERSION",
    "CLI_STABILITY_POLICY_VERSION",
    "ContractLoadError",
    "DOCTOR_OUTPUT_FORMAT_VERSION",
    "EVIDENCE_PACK_FORMAT_VERSION",
    "EVIDENCE_PACK_VERIFY_OUTPUT_FORMAT_VERSION",
    "MODEL_CLASSIFICATION_FORMAT_VERSION",
    "MODEL_ARTIFACT_IDENTITY_FORMAT_VERSION",
    "PLUGINS_OUTPUT_FORMAT_VERSION",
    "POLICY_PACK_VERIFY_OUTPUT_FORMAT_VERSION",
    "PUBLIC_EVIDENCE_INDEX_FORMAT_VERSION",
    "REPORT_SCHEMA_VERSION",
    "RUNTIME_MANIFEST_CONTRACT_VERSION",
    "RUNTIME_PROVIDER_ABI_VERSION",
    "RUNTIME_PROVIDER_CAPABILITIES_FORMAT_VERSION",
    "RUNTIME_PROVIDER_RECEIPT_FORMAT_VERSION",
    "RUNTIME_SCORING_OBSERVATION_FORMAT_VERSION",
    "RUNTIME_VERIFY_OUTPUT_FORMAT_VERSION",
    "STABLE_CLI_JSON_SURFACES",
    "VERIFY_OUTPUT_FORMAT_VERSION",
    "adapter_capability",
    "adapter_capability_map",
    "contract_catalog",
    "contract_path",
    "contract_reference",
    "contract_relpath",
    "load_adapter_capabilities",
    "load_json_contract",
    "load_model_classification",
    "load_model_family_catalog",
    "load_public_evidence_index",
    "load_plugin_compatibility",
    "load_policy_pack_schema",
    "load_evidence_pack_manifest_schema",
    "load_evidence_catalog",
    "load_runtime_manifest_schema",
    "load_runtime_provider_capabilities_schema",
    "load_model_artifact_identity_schema",
    "load_runtime_provider_receipt_schema",
    "load_runtime_scoring_observation_schema",
    "load_verify_output_schema",
    "load_support_matrix",
    "maintained_catalog_lanes",
    "public_subcontract_catalog",
    "stable_cli_json_surfaces",
    "support_lane_by_id",
    "support_lanes",
    "support_tiers",
]
