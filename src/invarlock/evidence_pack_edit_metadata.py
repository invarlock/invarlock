"""Evidence-pack edit metadata consistency checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

EDIT_METADATA_SCHEMA = "invarlock/evidence-pack-edit-metadata-v1"
VALIDATION_SUBJECT_CHECKPOINT = "validation_subject_checkpoint"
DEPLOYABLE_OPTIMIZED_SUBJECT = "deployable_optimized_subject"
FAULT_INJECTION_FIXTURE = "fault_injection_fixture"
DEPLOYABLE_SIDECARS = (
    "deployable_artifact_validation.json",
    "backend_inventory.json",
    "memory_report.json",
    "load_smoke.json",
    "inference_smoke.json",
)
DEPLOYABLE_SIDECAR_SCHEMAS = {
    "deployable_artifact_validation.json": (
        "invarlock/deployable-artifact-validation-v1"
    ),
    "backend_inventory.json": "invarlock/backend-inventory-v1",
    "memory_report.json": "invarlock/deployable-memory-report-v1",
    "load_smoke.json": "invarlock/deployable-load-smoke-v1",
    "inference_smoke.json": "invarlock/deployable-inference-smoke-v1",
}
EDIT_PROVENANCE_FAMILIES = {
    "custom",
    "deployable_backend_quantization",
    "dynamic_adapter",
    "fault_injection",
    "fine_tune",
    "knowledge_edit",
    "lora_merge",
    "lowrank_approximation",
    "magnitude_prune",
    "noop",
    "pruning",
    "quantization",
    "quantization_dequantized",
    "self_edit",
}
EDIT_IMPACT_SCENARIO_TYPES = {
    "target_success",
    "near_neighbor",
    "near_confuser",
    "unrelated_locality",
    "general_ability_sentinel",
    "multilingual_portability",
    "sequential_edit_stress",
}
EDIT_TOPOLOGY_ARTIFACT_KINDS = {
    "checkpoint",
    "adapter",
    "merged_adapter",
    "memory_module",
    "dynamic_weight_module",
    "runtime_config",
    "prompt_wrapper",
}
DELTA_AVAILABILITY_VALUES = {"none", "private", "public", "hash_only"}
PRIVACY_SENSITIVITY_VALUES = {
    "public",
    "internal",
    "customer_controlled",
    "sensitive",
}
_SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_load_error_types() -> tuple[type[BaseException], ...]:
    return (OSError, TypeError, ValueError, json.JSONDecodeError)


def _infer_scenario_artifact_class(spec: dict[str, Any]) -> str:
    artifact_class = spec.get("artifact_class")
    if isinstance(artifact_class, str) and artifact_class:
        return artifact_class
    generation = spec.get("generation")
    kind = generation.get("kind") if isinstance(generation, dict) else ""
    if kind == "error":
        return FAULT_INJECTION_FIXTURE
    if kind == "deployable_edit":
        return DEPLOYABLE_OPTIMIZED_SUBJECT
    return VALIDATION_SUBJECT_CHECKPOINT if kind == "edit" else ""


def _scenario_index_from_pack(pack_dir: Path) -> dict[str, dict[str, Any]]:
    scenarios_path = pack_dir / "metadata" / "scenarios.json"
    if not scenarios_path.is_file():
        return {}
    try:
        payload = _load_json(scenarios_path)
    except _json_load_error_types():
        return {}
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        scenario_id = item.get("id")
        if isinstance(scenario_id, str) and scenario_id:
            result[scenario_id] = item
    return result


def _report_scenario_id(pack_dir: Path, report_path: Path) -> str | None:
    try:
        rel = report_path.relative_to(pack_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 4 or parts[0] != "reports":
        return None
    if parts[2] == "errors":
        return parts[3] if len(parts) > 3 else None
    return parts[2]


def _load_json_sidecar(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = _load_json(path)
    except _json_load_error_types() as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "JSON sidecar must contain an object"
    return payload, None


def _expected_edit_type(spec: dict[str, Any]) -> str:
    generation = spec.get("generation")
    edit_spec = generation.get("edit_spec") if isinstance(generation, dict) else ""
    if isinstance(edit_spec, str) and edit_spec:
        return edit_spec.split(":", 1)[0]
    failure_class = spec.get("failure_class")
    if isinstance(failure_class, str) and "." in failure_class:
        return failure_class.rsplit(".", 1)[-1]
    return ""


def _metadata_consistency_errors(
    *,
    scenario_id: str,
    spec: dict[str, Any],
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_class = _infer_scenario_artifact_class(spec)
    expected_edit = _expected_edit_type(spec)
    artifact_class = metadata.get("artifact_class")
    edit_type = metadata.get("edit_type")
    prefix = f"{scenario_id}: "

    if metadata.get("schema") != EDIT_METADATA_SCHEMA:
        errors.append(prefix + "edit_metadata.json has unrecognized schema")
    if expected_class and artifact_class != expected_class:
        errors.append(
            prefix
            + f"artifact_class mismatch (scenario={expected_class!r}, "
            + f"metadata={artifact_class!r})"
        )
    if expected_edit and edit_type != expected_edit:
        errors.append(
            prefix
            + f"edit_type mismatch (scenario={expected_edit!r}, metadata={edit_type!r})"
        )

    optimized = metadata.get("optimized_deployment_backend")
    packed = metadata.get("packed_quantized_storage")
    if expected_class == DEPLOYABLE_OPTIMIZED_SUBJECT:
        if optimized is not True:
            errors.append(
                prefix
                + "deployable metadata must set optimized_deployment_backend=true"
            )
        if packed is not True:
            errors.append(
                prefix + "deployable metadata must set packed_quantized_storage=true"
            )
    if expected_class == VALIDATION_SUBJECT_CHECKPOINT:
        if optimized is not False:
            errors.append(
                prefix
                + "validation metadata must set optimized_deployment_backend=false"
            )
        if edit_type == "quant_rtn" and packed is not False:
            errors.append(
                prefix
                + "quant_rtn validation metadata must set packed_quantized_storage=false"
            )
    errors.extend(_optional_edit_provenance_errors(prefix, metadata))
    errors.extend(_optional_edit_impact_errors(prefix, metadata))
    errors.extend(_optional_edit_topology_errors(prefix, metadata))
    errors.extend(_optional_delta_privacy_errors(prefix, metadata))
    return errors


def _optional_edit_provenance_errors(
    prefix: str, metadata: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    provenance = metadata.get("edit_provenance")
    if provenance is None:
        return errors
    if not isinstance(provenance, dict):
        return [prefix + "edit_provenance must be an object when present"]

    family = provenance.get("edit_family")
    if family is not None and (
        not isinstance(family, str) or family not in EDIT_PROVENANCE_FAMILIES
    ):
        errors.append(prefix + f"edit_provenance.edit_family unsupported: {family!r}")

    method = provenance.get("edit_method")
    if method is not None and (not isinstance(method, str) or not method.strip()):
        errors.append(prefix + "edit_provenance.edit_method must be a non-empty string")

    edit_count = provenance.get("edit_count")
    if edit_count is not None and (
        not isinstance(edit_count, int)
        or isinstance(edit_count, bool)
        or edit_count < 1
    ):
        errors.append(prefix + "edit_provenance.edit_count must be a positive integer")

    for key in (
        "target_set_digest",
        "editor_artifact_digest",
        "self_edit_data_digest",
    ):
        digest = provenance.get(key)
        if digest is not None and (
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
        ):
            errors.append(
                prefix
                + f"edit_provenance.{key} must be a sha256:<64 lowercase hex> digest"
            )

    dynamic_required = provenance.get("dynamic_runtime_required")
    if dynamic_required is not None and not isinstance(dynamic_required, bool):
        errors.append(
            prefix + "edit_provenance.dynamic_runtime_required must be boolean"
        )
    return errors


def _optional_edit_impact_errors(prefix: str, metadata: dict[str, Any]) -> list[str]:
    impact = metadata.get("edit_impact")
    if impact is None:
        return []
    if not isinstance(impact, dict):
        return [prefix + "edit_impact must be an object when present"]

    scenario_types = impact.get("scenario_types")
    if scenario_types is None:
        return []
    if not isinstance(scenario_types, list):
        return [prefix + "edit_impact.scenario_types must be a list when present"]

    errors: list[str] = []
    for index, scenario_type in enumerate(scenario_types):
        if (
            not isinstance(scenario_type, str)
            or scenario_type not in EDIT_IMPACT_SCENARIO_TYPES
        ):
            errors.append(
                prefix
                + f"edit_impact.scenario_types[{index}] unsupported: "
                + f"{scenario_type!r}"
            )
    return errors


def _optional_edit_topology_errors(prefix: str, metadata: dict[str, Any]) -> list[str]:
    topology = metadata.get("edit_topology")
    if topology is None:
        return []
    if not isinstance(topology, dict):
        return [prefix + "edit_topology must be an object when present"]

    errors: list[str] = []
    artifact_kind = topology.get("artifact_kind")
    if artifact_kind is not None and (
        not isinstance(artifact_kind, str)
        or artifact_kind not in EDIT_TOPOLOGY_ARTIFACT_KINDS
    ):
        errors.append(
            prefix + f"edit_topology.artifact_kind unsupported: {artifact_kind!r}"
        )

    module_hashes = topology.get("module_hashes")
    if module_hashes is not None:
        if not isinstance(module_hashes, dict):
            errors.append(prefix + "edit_topology.module_hashes must be an object")
        else:
            for name, digest in module_hashes.items():
                if not isinstance(name, str) or not name.strip():
                    errors.append(
                        prefix + f"edit_topology.module_hashes key invalid: {name!r}"
                    )
                    continue
                if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                    errors.append(
                        prefix
                        + f"edit_topology.module_hashes.{name} must be a "
                        + "sha256:<64 lowercase hex> digest"
                    )

    activation_policy = topology.get("runtime_activation_policy")
    if activation_policy is not None:
        valid_policy = isinstance(activation_policy, dict) or (
            isinstance(activation_policy, str) and bool(activation_policy.strip())
        )
        if not valid_policy:
            errors.append(
                prefix
                + "edit_topology.runtime_activation_policy must be a non-empty string or object"
            )

    data_ref = topology.get("training_or_edit_data_ref")
    if data_ref is not None and (not isinstance(data_ref, str) or not data_ref.strip()):
        errors.append(
            prefix
            + "edit_topology.training_or_edit_data_ref must be a non-empty string"
        )
    return errors


def _optional_delta_privacy_errors(prefix: str, metadata: dict[str, Any]) -> list[str]:
    privacy = metadata.get("delta_privacy")
    if privacy is None:
        return []
    if not isinstance(privacy, dict):
        return [prefix + "delta_privacy must be an object when present"]

    errors: list[str] = []
    delta_available = privacy.get("delta_available")
    if delta_available is not None and (
        not isinstance(delta_available, str)
        or delta_available not in DELTA_AVAILABILITY_VALUES
    ):
        errors.append(
            prefix + f"delta_privacy.delta_available unsupported: {delta_available!r}"
        )

    sensitivity = privacy.get("privacy_sensitivity")
    if sensitivity is not None and (
        not isinstance(sensitivity, str)
        or sensitivity not in PRIVACY_SENSITIVITY_VALUES
    ):
        errors.append(
            prefix + f"delta_privacy.privacy_sensitivity unsupported: {sensitivity!r}"
        )

    raw_approval = privacy.get("public_raw_delta_approved")
    if raw_approval is not None and not isinstance(raw_approval, bool):
        errors.append(
            prefix + "delta_privacy.public_raw_delta_approved must be boolean"
        )
    return errors


def _deployable_sidecar_consistency_errors(
    *,
    scenario_id: str,
    sidecar: str,
    payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    prefix = f"{scenario_id}: "
    expected_schema = DEPLOYABLE_SIDECAR_SCHEMAS.get(sidecar)
    if expected_schema and payload.get("schema") != expected_schema:
        errors.append(
            prefix
            + f"deployable sidecar schema mismatch ({sidecar}): "
            + f"expected {expected_schema!r}, got {payload.get('schema')!r}"
        )

    if sidecar == "deployable_artifact_validation.json":
        if payload.get("ok") is not True:
            errors.append(prefix + f"deployable sidecar did not pass: {sidecar}")
        if payload.get("load_smoke") is not True:
            errors.append(prefix + f"{sidecar} load_smoke must be true")
        if payload.get("inference_smoke") is not True:
            errors.append(prefix + f"{sidecar} inference_smoke must be true")
    elif sidecar == "backend_inventory.json":
        if "ok" in payload and payload.get("ok") is not True:
            errors.append(prefix + f"deployable sidecar did not pass: {sidecar}")
        if payload.get("load_smoke") is not True:
            errors.append(prefix + f"{sidecar} load_smoke must be true")
        if payload.get("inference_smoke") is not True:
            errors.append(prefix + f"{sidecar} inference_smoke must be true")
        quantized_count = payload.get("quantized_module_count")
        if not isinstance(quantized_count, int) or quantized_count < 0:
            errors.append(
                prefix + f"{sidecar} quantized_module_count must be non-negative int"
            )
        if not isinstance(payload.get("quantized_module_types"), list):
            errors.append(prefix + f"{sidecar} quantized_module_types must be a list")
        if not isinstance(payload.get("memory_footprint"), dict):
            errors.append(prefix + f"{sidecar} memory_footprint must be an object")
    elif payload.get("ok") is not True:
        errors.append(prefix + f"deployable sidecar did not pass: {sidecar}")
    return errors


def _verify_edit_metadata_consistency(pack_dir: Path) -> list[str]:
    errors: list[str] = []
    scenarios = _scenario_index_from_pack(pack_dir)
    if not scenarios:
        return errors

    deployable_scenarios = {
        scenario_id
        for scenario_id, spec in scenarios.items()
        if isinstance(spec, dict)
        and _infer_scenario_artifact_class(spec) == DEPLOYABLE_OPTIMIZED_SUBJECT
    }
    seen_deployable_reports: set[str] = set()

    for report_path in sorted(pack_dir.glob("reports/**/evaluation.report.json")):
        scenario_id = _report_scenario_id(pack_dir, report_path)
        if scenario_id is None:
            continue
        spec = scenarios.get(scenario_id)
        if not isinstance(spec, dict):
            continue
        artifact_class = _infer_scenario_artifact_class(spec)
        if artifact_class not in {
            VALIDATION_SUBJECT_CHECKPOINT,
            DEPLOYABLE_OPTIMIZED_SUBJECT,
        }:
            continue

        report_dir = report_path.parent
        metadata_path = report_dir / "edit_metadata.json"
        if not metadata_path.is_file():
            errors.append(f"{scenario_id}: edit_metadata.json missing next to report")
            continue
        metadata, metadata_error = _load_json_sidecar(metadata_path)
        if metadata_error is not None or metadata is None:
            errors.append(
                f"{scenario_id}: edit_metadata.json invalid: {metadata_error}"
            )
            continue
        errors.extend(
            _metadata_consistency_errors(
                scenario_id=scenario_id,
                spec=spec,
                metadata=metadata,
            )
        )

        if artifact_class == DEPLOYABLE_OPTIMIZED_SUBJECT:
            seen_deployable_reports.add(scenario_id)
            for sidecar in DEPLOYABLE_SIDECARS:
                sidecar_path = report_dir / sidecar
                if not sidecar_path.is_file():
                    errors.append(
                        f"{scenario_id}: deployable sidecar missing: {sidecar}"
                    )
                else:
                    payload, sidecar_error = _load_json_sidecar(sidecar_path)
                    if sidecar_error is not None:
                        errors.append(
                            f"{scenario_id}: deployable sidecar invalid "
                            f"({sidecar}): {sidecar_error}"
                        )
                    else:
                        errors.extend(
                            _deployable_sidecar_consistency_errors(
                                scenario_id=scenario_id,
                                sidecar=sidecar,
                                payload=cast(dict[str, Any], payload),
                            )
                        )

    for scenario_id in sorted(deployable_scenarios - seen_deployable_reports):
        errors.append(
            f"{scenario_id}: deployable scenario has no deployability report sidecars"
        )
    return errors


__all__ = [
    "DEPLOYABLE_OPTIMIZED_SUBJECT",
    "DEPLOYABLE_SIDECARS",
    "EDIT_METADATA_SCHEMA",
    "EDIT_IMPACT_SCENARIO_TYPES",
    "EDIT_PROVENANCE_FAMILIES",
    "FAULT_INJECTION_FIXTURE",
    "VALIDATION_SUBJECT_CHECKPOINT",
    "_expected_edit_type",
    "_infer_scenario_artifact_class",
    "_load_json_sidecar",
    "_metadata_consistency_errors",
    "_deployable_sidecar_consistency_errors",
    "_report_scenario_id",
    "_scenario_index_from_pack",
    "_verify_edit_metadata_consistency",
]
