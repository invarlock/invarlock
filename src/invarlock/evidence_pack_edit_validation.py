"""Core edit-metadata field and optional provenance validation."""

from __future__ import annotations

from typing import Any

from invarlock.evidence_pack_edit_common import (
    _SHA256_RE,
    _SYNTHETIC_EDIT_TYPES,
    DELTA_AVAILABILITY_VALUES,
    DEPLOYABLE_OPTIMIZED_SUBJECT,
    EDIT_IMPACT_SCENARIO_TYPES,
    EDIT_METADATA_SCHEMA,
    EDIT_PROVENANCE_FAMILIES,
    EDIT_TOPOLOGY_ARTIFACT_KINDS,
    PRIVACY_SENSITIVITY_VALUES,
    VALIDATION_SUBJECT_CHECKPOINT,
    _expected_edit_type,
    _expected_literal_pruning_params,
    _infer_scenario_artifact_class,
    edit_metadata_coverage_errors,
)


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
    errors.extend(
        prefix + error
        for error in edit_metadata_coverage_errors(
            metadata,
            require_positive=expected_class
            in {VALIDATION_SUBJECT_CHECKPOINT, DEPLOYABLE_OPTIMIZED_SUBJECT},
        )
    )
    if edit_type == "magnitude_prune":
        expected_sparsity, expected_scope, pruning_error = (
            _expected_literal_pruning_params(spec)
        )
        if pruning_error is not None:
            errors.append(prefix + pruning_error)
        if expected_scope is not None and metadata.get("scope") != expected_scope:
            errors.append(prefix + "magnitude_prune scope does not match scenario")
        parameters = metadata.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        metadata_sparsity = parameters.get("target_sparsity")
        if expected_sparsity is not None and (
            not isinstance(metadata_sparsity, int | float)
            or isinstance(metadata_sparsity, bool)
            or abs(float(metadata_sparsity) - expected_sparsity) > 1e-12
        ):
            errors.append(
                prefix + "magnitude_prune target_sparsity does not match scenario"
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
    if edit_type in _SYNTHETIC_EDIT_TYPES:
        provenance = metadata.get("edit_provenance")
        if not isinstance(provenance, dict):
            errors.append(prefix + "synthetic edit metadata requires edit_provenance")
        else:
            if provenance.get("synthetic") is not True:
                errors.append(prefix + "synthetic edits must declare synthetic=true")
            if edit_type == "synthetic_lowrank_delta":
                if provenance.get("trained_adapter") is not False:
                    errors.append(
                        prefix
                        + "synthetic low-rank delta must declare trained_adapter=false"
                    )
                if provenance.get("adapter_merge_performed") is not False:
                    errors.append(
                        prefix
                        + "synthetic low-rank delta must declare adapter_merge_performed=false"
                    )
            if edit_type == "synthetic_dense_update":
                if provenance.get("optimization_performed") is not False:
                    errors.append(
                        prefix
                        + "synthetic dense update must declare optimization_performed=false"
                    )
                if provenance.get("training_data_used") is not False:
                    errors.append(
                        prefix
                        + "synthetic dense update must declare training_data_used=false"
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
