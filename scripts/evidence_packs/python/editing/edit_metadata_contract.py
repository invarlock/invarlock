"""Schema constants and validation for evidence-pack edit metadata."""

from __future__ import annotations

import re

from invarlock.evidence_pack_edit_common import (
    DELTA_AVAILABILITY_VALUES,
    EDIT_IMPACT_SCENARIO_TYPES,
    EDIT_PROVENANCE_FAMILIES,
    EDIT_TOPOLOGY_ARTIFACT_KINDS,
    PRIVACY_SENSITIVITY_VALUES,
    edit_metadata_coverage_errors,
)

EDIT_METADATA_SCHEMA = "invarlock/evidence-pack-edit-metadata-v1"

VALIDATION_SUBJECT_CHECKPOINT = "validation_subject_checkpoint"
DEPLOYABLE_OPTIMIZED_SUBJECT = "deployable_optimized_subject"
FAULT_INJECTION_FIXTURE = "fault_injection_fixture"
EVIDENCE_ONLY_PACK = "evidence_only_pack"

SYNTHETIC_LOWRANK_DELTA = "synthetic_lowrank_delta"
SYNTHETIC_DENSE_UPDATE = "synthetic_dense_update"

# These names described mutable in-memory simulations, not checkpoint-level
# transformations with durable replay evidence.  They remain explicitly
# forbidden rather than falling through the generic metadata contract.
UNVERIFIABLE_GENERATED_EDIT_TYPES = frozenset({"fp8_quant", "lowrank_svd"})

ALLOWED_ARTIFACT_CLASSES = {
    VALIDATION_SUBJECT_CHECKPOINT,
    DEPLOYABLE_OPTIMIZED_SUBJECT,
    FAULT_INJECTION_FIXTURE,
    EVIDENCE_ONLY_PACK,
}

VALIDATION_STORAGE_FORMATS = {
    "quant_rtn": "float_dequantized",
    "magnitude_prune": "dense_float_with_zeros",
    "lora_merge": "dense_float_merged_adapter_checkpoint",
    "fine_tune": "dense_float_fine_tuned_checkpoint",
    SYNTHETIC_LOWRANK_DELTA: "dense_float_with_synthetic_lowrank_delta",
    SYNTHETIC_DENSE_UPDATE: "dense_float_with_synthetic_update",
}

EDIT_SEMANTICS_EXTERNAL_SUBJECT = "external_subject_validation_edit"
EDIT_SEMANTICS_DEPLOYABLE = "backend_deployable_edit"
_SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


def is_unverifiable_generated_edit_type(edit_type: object) -> bool:
    """Return whether a label denotes a retired mutable simulation."""

    if not isinstance(edit_type, str):
        return False
    normalized = edit_type.strip().lower().replace("-", "_")
    return normalized in UNVERIFIABLE_GENERATED_EDIT_TYPES


def reject_unverifiable_generated_edit_type(edit_type: object) -> None:
    """Reject labels that lack a checkpoint storage and replay contract."""

    if not is_unverifiable_generated_edit_type(edit_type):
        return
    normalized = str(edit_type).strip().lower().replace("-", "_")
    raise ValueError(
        f"{normalized} requires a dedicated storage and replay contract before use"
    )


def storage_format_for_edit(edit_type: str) -> str:
    """Return the validation checkpoint storage description for an edit type."""

    reject_unverifiable_generated_edit_type(edit_type)
    return VALIDATION_STORAGE_FORMATS.get(edit_type, "float_dequantized")


def validate_edit_metadata(
    metadata: dict[str, object],
    *,
    expected_edit_type: str | None = None,
    expected_artifact_class: str | None = None,
) -> list[str]:
    """Return contract violations for an edit-metadata payload."""

    errors: list[str] = []

    schema = metadata.get("schema")
    if schema != EDIT_METADATA_SCHEMA:
        errors.append(f"unknown edit metadata schema: {schema!r}")

    artifact_class = metadata.get("artifact_class")
    if artifact_class not in ALLOWED_ARTIFACT_CLASSES:
        errors.append(f"invalid artifact_class: {artifact_class!r}")
    if expected_artifact_class and artifact_class != expected_artifact_class:
        errors.append(
            f"artifact_class mismatch: expected {expected_artifact_class!r}, "
            f"got {artifact_class!r}"
        )

    edit_type = metadata.get("edit_type")
    if not isinstance(edit_type, str) or not edit_type:
        errors.append("edit_type must be a non-empty string")
    elif is_unverifiable_generated_edit_type(edit_type):
        errors.append(f"{edit_type!r} requires a dedicated storage and replay contract")
    if expected_edit_type and edit_type != expected_edit_type:
        errors.append(
            f"edit_type mismatch: expected {expected_edit_type!r}, got {edit_type!r}"
        )

    errors.extend(
        edit_metadata_coverage_errors(
            metadata,
            require_positive=artifact_class
            in {VALIDATION_SUBJECT_CHECKPOINT, DEPLOYABLE_OPTIMIZED_SUBJECT},
        )
    )

    optimized = metadata.get("optimized_deployment_backend")
    packed = metadata.get("packed_quantized_storage")
    runtime_reduction = metadata.get("runtime_memory_reduction")
    backend = metadata.get("backend")

    if artifact_class == VALIDATION_SUBJECT_CHECKPOINT:
        if optimized is not False:
            errors.append(
                "validation artifacts must set optimized_deployment_backend=false"
            )
        if packed is not False:
            errors.append(
                "validation artifacts must set packed_quantized_storage=false"
            )
        if runtime_reduction is not False:
            errors.append(
                "validation artifacts must set runtime_memory_reduction=false"
            )
        if backend is not None:
            errors.append("validation artifacts must set backend=null")
        if not is_unverifiable_generated_edit_type(edit_type):
            expected_storage = storage_format_for_edit(str(edit_type or ""))
            if metadata.get("storage_format") != expected_storage:
                errors.append(
                    f"storage_format mismatch for {edit_type!r}: expected {expected_storage!r}"
                )
            if metadata.get("actual_storage_format") != expected_storage:
                errors.append(
                    f"actual_storage_format mismatch for {edit_type!r}: expected {expected_storage!r}"
                )

    if artifact_class == DEPLOYABLE_OPTIMIZED_SUBJECT:
        if optimized is not True:
            errors.append(
                "deployable artifacts must set optimized_deployment_backend=true"
            )
        if packed is not True:
            errors.append("deployable artifacts must set packed_quantized_storage=true")
        if not isinstance(backend, str) or not backend:
            errors.append("deployable artifacts must record a backend")

    if metadata.get("deployable_as_hf_checkpoint") is not True and artifact_class in {
        VALIDATION_SUBJECT_CHECKPOINT,
        DEPLOYABLE_OPTIMIZED_SUBJECT,
    }:
        errors.append(
            "subject checkpoint artifacts must set deployable_as_hf_checkpoint=true"
        )

    if edit_type in {SYNTHETIC_DENSE_UPDATE, SYNTHETIC_LOWRANK_DELTA}:
        provenance = metadata.get("edit_provenance")
        if not isinstance(provenance, dict):
            errors.append("synthetic edits require edit_provenance")
        else:
            if provenance.get("synthetic") is not True:
                errors.append("synthetic edits require edit_provenance.synthetic=true")
            if edit_type == SYNTHETIC_LOWRANK_DELTA:
                if provenance.get("trained_adapter") is not False:
                    errors.append(
                        "synthetic low-rank deltas must declare trained_adapter=false"
                    )
                if provenance.get("adapter_merge_performed") is not False:
                    errors.append(
                        "synthetic low-rank deltas must declare "
                        "adapter_merge_performed=false"
                    )
            if edit_type == SYNTHETIC_DENSE_UPDATE:
                if provenance.get("optimization_performed") is not False:
                    errors.append(
                        "synthetic dense updates must declare optimization_performed=false"
                    )
                if provenance.get("training_data_used") is not False:
                    errors.append(
                        "synthetic dense updates must declare training_data_used=false"
                    )

    errors.extend(_validate_optional_edit_provenance(metadata))
    errors.extend(_validate_optional_edit_impact(metadata))
    errors.extend(_validate_optional_edit_topology(metadata))
    errors.extend(_validate_optional_delta_privacy(metadata))
    return errors


def _validate_optional_edit_provenance(metadata: dict[str, object]) -> list[str]:
    provenance = metadata.get("edit_provenance")
    if provenance is None:
        return []
    if not isinstance(provenance, dict):
        return ["edit_provenance must be an object when present"]

    errors: list[str] = []
    family = provenance.get("edit_family")
    if family is not None and (
        not isinstance(family, str) or family not in EDIT_PROVENANCE_FAMILIES
    ):
        errors.append(f"edit_provenance.edit_family unsupported: {family!r}")

    method = provenance.get("edit_method")
    if method is not None and (not isinstance(method, str) or not method.strip()):
        errors.append("edit_provenance.edit_method must be a non-empty string")

    edit_count = provenance.get("edit_count")
    if edit_count is not None and (
        not isinstance(edit_count, int)
        or isinstance(edit_count, bool)
        or edit_count < 1
    ):
        errors.append("edit_provenance.edit_count must be a positive integer")

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
                f"edit_provenance.{key} must be a sha256:<64 lowercase hex> digest"
            )

    dynamic_required = provenance.get("dynamic_runtime_required")
    if dynamic_required is not None and not isinstance(dynamic_required, bool):
        errors.append("edit_provenance.dynamic_runtime_required must be boolean")
    return errors


def _validate_optional_edit_impact(metadata: dict[str, object]) -> list[str]:
    impact = metadata.get("edit_impact")
    if impact is None:
        return []
    if not isinstance(impact, dict):
        return ["edit_impact must be an object when present"]

    scenario_types = impact.get("scenario_types")
    if scenario_types is None:
        return []
    if not isinstance(scenario_types, list):
        return ["edit_impact.scenario_types must be a list when present"]

    errors: list[str] = []
    for index, scenario_type in enumerate(scenario_types):
        if (
            not isinstance(scenario_type, str)
            or scenario_type not in EDIT_IMPACT_SCENARIO_TYPES
        ):
            errors.append(
                f"edit_impact.scenario_types[{index}] unsupported: {scenario_type!r}"
            )
    return errors


def _validate_optional_edit_topology(metadata: dict[str, object]) -> list[str]:
    topology = metadata.get("edit_topology")
    if topology is None:
        return []
    if not isinstance(topology, dict):
        return ["edit_topology must be an object when present"]

    errors: list[str] = []
    artifact_kind = topology.get("artifact_kind")
    if artifact_kind is not None and (
        not isinstance(artifact_kind, str)
        or artifact_kind not in EDIT_TOPOLOGY_ARTIFACT_KINDS
    ):
        errors.append(f"edit_topology.artifact_kind unsupported: {artifact_kind!r}")

    module_hashes = topology.get("module_hashes")
    if module_hashes is not None:
        if not isinstance(module_hashes, dict):
            errors.append("edit_topology.module_hashes must be an object")
        else:
            for name, digest in module_hashes.items():
                if not isinstance(name, str) or not name.strip():
                    errors.append(f"edit_topology.module_hashes key invalid: {name!r}")
                    continue
                if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                    errors.append(
                        f"edit_topology.module_hashes.{name} must be a "
                        "sha256:<64 lowercase hex> digest"
                    )

    activation_policy = topology.get("runtime_activation_policy")
    if activation_policy is not None:
        valid_policy = isinstance(activation_policy, dict) or (
            isinstance(activation_policy, str) and bool(activation_policy.strip())
        )
        if not valid_policy:
            errors.append(
                "edit_topology.runtime_activation_policy must be a non-empty string or object"
            )

    data_ref = topology.get("training_or_edit_data_ref")
    if data_ref is not None and (not isinstance(data_ref, str) or not data_ref.strip()):
        errors.append(
            "edit_topology.training_or_edit_data_ref must be a non-empty string"
        )
    return errors


def _validate_optional_delta_privacy(metadata: dict[str, object]) -> list[str]:
    privacy = metadata.get("delta_privacy")
    if privacy is None:
        return []
    if not isinstance(privacy, dict):
        return ["delta_privacy must be an object when present"]

    errors: list[str] = []
    delta_available = privacy.get("delta_available")
    if delta_available is not None and (
        not isinstance(delta_available, str)
        or delta_available not in DELTA_AVAILABILITY_VALUES
    ):
        errors.append(f"delta_privacy.delta_available unsupported: {delta_available!r}")

    sensitivity = privacy.get("privacy_sensitivity")
    if sensitivity is not None and (
        not isinstance(sensitivity, str)
        or sensitivity not in PRIVACY_SENSITIVITY_VALUES
    ):
        errors.append(f"delta_privacy.privacy_sensitivity unsupported: {sensitivity!r}")

    raw_approval = privacy.get("public_raw_delta_approved")
    if raw_approval is not None and not isinstance(raw_approval, bool):
        errors.append("delta_privacy.public_raw_delta_approved must be boolean")
    return errors
