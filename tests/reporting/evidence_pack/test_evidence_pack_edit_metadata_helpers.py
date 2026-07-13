from __future__ import annotations

import invarlock.evidence_pack_deployable_validation as deployable_mod
import invarlock.evidence_pack_edit_validation as edit_validation_mod
from invarlock.evidence_pack_contracts.deployable_coverage import (
    canonical_names_sha256,
)


def _packed_runtime_facts() -> dict[str, object]:
    module_names = ["model.fc1", "model.fc2"]
    return {
        "quantized_module_count": len(module_names),
        "quantized_module_names": module_names,
        "quantized_module_names_sha256": canonical_names_sha256(module_names),
        "quantized_module_types": ["bitsandbytes.nn.Linear8bitLt"],
        "packed_weight_storage_elements": 6,
    }


def _logical_coverage() -> dict[str, object]:
    weight_names = ["model.fc1.weight", "model.fc2.weight"]
    return {
        "basis": "dense_baseline_unique_parameters",
        "weight_tensor_names": weight_names,
        "weight_tensor_names_sha256": canonical_names_sha256(weight_names),
        "weight_tensor_count": len(weight_names),
        "parameter_elements": 12,
        "total_unique_parameter_elements": 16,
    }


def _positive_edit_coverage() -> dict[str, object]:
    return {
        "edited_tensors": 1,
        "edited_params": 4,
        "total_params": 8,
        "coverage_ratio": 0.5,
    }


def test_deployable_artifact_validation_sidecar_requires_non_authoritative_scope() -> (
    None
):
    assert (
        deployable_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="deployable_artifact_validation.json",
            payload={
                "schema": "invarlock/deployable-artifact-validation-v1",
                "ok": True,
                "validation_scope": "structural_only",
                "runtime_proof_authoritative": False,
                "load_smoke": True,
                "inference_smoke": True,
            },
        )
        == []
    )


def test_runtime_deployability_sidecar_requires_authoritative_reproof() -> None:
    valid_payload = {
        "schema": "invarlock/deployable-artifact-validation-v1",
        "ok": True,
        "validation_scope": "runtime_reproof",
        "runtime_proof_authoritative": True,
        "runtime_proof": {"packed_module_reloaded": True},
        "load_smoke": True,
        "inference_smoke": True,
    }
    assert (
        deployable_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="runtime_deployability_validation.json",
            payload=valid_payload,
        )
        == []
    )

    invalid_payload = dict(valid_payload)
    invalid_payload["runtime_proof_authoritative"] = False
    errors = deployable_mod._deployable_sidecar_consistency_errors(
        scenario_id="deploy",
        sidecar="runtime_deployability_validation.json",
        payload=invalid_payload,
    )

    assert any("runtime_proof_authoritative must be true" in error for error in errors)


def test_backend_inventory_sidecar_accepts_passing_inventory() -> None:
    runtime_facts = _packed_runtime_facts()
    assert (
        deployable_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="backend_inventory.json",
            payload={
                "schema": "invarlock/backend-inventory-v1",
                "ok": True,
                "load_smoke": True,
                "inference_smoke": True,
                **runtime_facts,
                "logical_coverage": _logical_coverage(),
                "memory_footprint": {
                    "reported_bytes": 1024,
                    "method": "get_memory_footprint",
                },
            },
        )
        == []
    )


def test_backend_inventory_sidecar_reports_failed_inventory_contract() -> None:
    errors = deployable_mod._deployable_sidecar_consistency_errors(
        scenario_id="deploy",
        sidecar="backend_inventory.json",
        payload={
            "schema": "wrong",
            "ok": False,
            "load_smoke": False,
            "inference_smoke": False,
            "quantized_module_count": -1,
            "quantized_module_types": "Linear8bitLt",
            "memory_footprint": None,
        },
    )

    assert any("schema mismatch" in error for error in errors)
    assert any("deployable sidecar did not pass" in error for error in errors)
    assert any("load_smoke must be true" in error for error in errors)
    assert any("inference_smoke must be true" in error for error in errors)
    assert any("quantized module names are invalid" in error for error in errors)
    assert any(
        "logical coverage has missing or unsupported fields" in error
        for error in errors
    )
    assert any("memory_footprint must be an object" in error for error in errors)


def test_deployable_sidecar_valid_and_invalid_branch_matrix() -> None:
    runtime_facts = _packed_runtime_facts()
    logical_coverage = _logical_coverage()
    assert (
        deployable_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="future_sidecar.json",
            payload={"ok": True},
        )
        == []
    )
    assert deployable_mod._deployable_sidecar_consistency_errors(
        scenario_id="deploy",
        sidecar="backend_inventory.json",
        payload={
            "schema": "invarlock/backend-inventory-v1",
            "load_smoke": True,
            "inference_smoke": True,
            **runtime_facts,
            "logical_coverage": logical_coverage,
            "memory_footprint": {"reported_bytes": 0},
        },
    ) == [
        "deploy: backend_inventory.json memory_footprint.reported_bytes must be positive"
    ]
    assert (
        deployable_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="publication_commit.json",
            payload={
                "schema": "invarlock/deployable-publication-commit-v1",
                "committed": True,
            },
        )
        == []
    )
    assert (
        deployable_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="memory_report.json",
            payload={
                "schema": "invarlock/deployable-memory-report-v1",
                "ok": True,
                "baseline_reported_bytes": 200,
                "quantized_reported_bytes": 100,
                "reduction_bytes": 100,
                "reduction_ratio": 0.5,
                "runtime_memory_reduction_observed": True,
            },
        )
        == []
    )
    assert (
        deployable_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="load_smoke.json",
            payload={
                "schema": "invarlock/deployable-load-smoke-v1",
                "ok": True,
                "loaded_from_saved_checkpoint": True,
                "load_time_quantization_override": False,
                **runtime_facts,
                "logical_coverage": logical_coverage,
            },
        )
        == []
    )
    assert (
        deployable_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="inference_smoke.json",
            payload={
                "schema": "invarlock/deployable-inference-smoke-v1",
                "ok": True,
                "all_logits_finite": True,
                "logits_sha256": "sha256:" + "a" * 64,
                "logits_shape": [1, 2, 3],
            },
        )
        == []
    )


def test_optional_edit_provenance_and_impact_metadata_accepts_valid_payload() -> None:
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="lora",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "lora_merge:custom:attn"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "lora_merge",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "coverage": _positive_edit_coverage(),
            "edit_provenance": {
                "edit_family": "lora_merge",
                "edit_method": "custom",
                "edit_count": 1,
                "target_set_digest": "sha256:" + "a" * 64,
                "editor_artifact_digest": "sha256:" + "b" * 64,
                "self_edit_data_digest": "sha256:" + "c" * 64,
                "dynamic_runtime_required": False,
            },
            "edit_impact": {
                "scenario_types": [
                    "target_success",
                    "near_neighbor",
                    "unrelated_locality",
                    "general_ability_sentinel",
                ]
            },
        },
    )

    assert errors == []


def test_optional_edit_provenance_and_impact_metadata_reports_malformed_payload() -> (
    None
):
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="knowledge",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:knowledge:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "edit_provenance": {
                "edit_family": "unsupported_edit_family",
                "edit_count": 0,
                "target_set_digest": "not-a-digest",
                "dynamic_runtime_required": "false",
            },
            "edit_impact": {
                "scenario_types": ["target_success", "unsupported_scenario_type"]
            },
        },
    )

    assert any("edit_provenance.edit_family" in error for error in errors)
    assert any("edit_provenance.edit_count" in error for error in errors)
    assert any("edit_provenance.target_set_digest" in error for error in errors)
    assert any("edit_provenance.dynamic_runtime_required" in error for error in errors)
    assert any("edit_impact.scenario_types[1]" in error for error in errors)


def test_optional_edit_metadata_reports_non_string_family_and_scenarios() -> None:
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="knowledge",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:knowledge:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "edit_provenance": {"edit_family": ["lora_merge"]},
            "edit_impact": {"scenario_types": ["target_success", {"kind": "bad"}]},
        },
    )

    assert any("edit_provenance.edit_family" in error for error in errors)
    assert any("edit_impact.scenario_types[1]" in error for error in errors)


def test_optional_edit_metadata_reports_non_object_sections() -> None:
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="knowledge",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:knowledge:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "edit_provenance": ["knowledge_edit"],
            "edit_impact": ["target_success"],
        },
    )

    assert any("edit_provenance must be an object" in error for error in errors)
    assert any("edit_impact must be an object" in error for error in errors)


def test_optional_edit_metadata_reports_invalid_method_and_scenario_list() -> None:
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="knowledge",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:knowledge:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "edit_provenance": {"edit_method": " "},
            "edit_impact": {"scenario_types": "target_success"},
        },
    )

    assert any("edit_provenance.edit_method" in error for error in errors)
    assert any("edit_impact.scenario_types must be a list" in error for error in errors)


def test_optional_edit_impact_accepts_missing_scenario_types() -> None:
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="knowledge",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:knowledge:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "coverage": _positive_edit_coverage(),
            "edit_impact": {},
        },
    )

    assert errors == []


def test_optional_edit_topology_and_delta_privacy_accepts_valid_payload() -> None:
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="dynamic",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:dynamic:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "coverage": _positive_edit_coverage(),
            "edit_topology": {
                "artifact_kind": "dynamic_weight_module",
                "module_hashes": {"generator": "sha256:" + "a" * 64},
                "runtime_activation_policy": "query_conditioned",
                "training_or_edit_data_ref": "hash-only-target-set",
            },
            "delta_privacy": {
                "delta_available": "hash_only",
                "privacy_sensitivity": "customer_controlled",
                "public_raw_delta_approved": False,
            },
        },
    )

    assert errors == []


def test_optional_edit_topology_and_delta_privacy_reports_malformed_payload() -> None:
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="dynamic",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:dynamic:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "edit_topology": {
                "artifact_kind": "unsupported_kind",
                "module_hashes": {"generator": "not-a-digest"},
                "runtime_activation_policy": "",
                "training_or_edit_data_ref": 123,
            },
            "delta_privacy": {
                "delta_available": "raw_everywhere",
                "privacy_sensitivity": "none",
                "public_raw_delta_approved": "false",
            },
        },
    )

    assert any("edit_topology.artifact_kind" in error for error in errors)
    assert any("edit_topology.module_hashes.generator" in error for error in errors)
    assert any("edit_topology.runtime_activation_policy" in error for error in errors)
    assert any("edit_topology.training_or_edit_data_ref" in error for error in errors)
    assert any("delta_privacy.delta_available" in error for error in errors)
    assert any("delta_privacy.privacy_sensitivity" in error for error in errors)
    assert any("delta_privacy.public_raw_delta_approved" in error for error in errors)


def test_optional_edit_topology_accepts_empty_descriptive_payload() -> None:
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="dynamic",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:dynamic:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "coverage": _positive_edit_coverage(),
            "edit_topology": {},
            "delta_privacy": {},
        },
    )

    assert errors == []


def test_optional_edit_topology_reports_non_object_shapes() -> None:
    errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="dynamic",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:dynamic:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "edit_topology": ["checkpoint"],
            "delta_privacy": ["hash_only"],
        },
    )

    assert any("edit_topology must be an object" in error for error in errors)
    assert any("delta_privacy must be an object" in error for error in errors)


def test_optional_edit_topology_reports_bad_module_hash_container_and_key() -> None:
    non_object_errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="dynamic",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:dynamic:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "edit_topology": {"module_hashes": ["sha256:" + "a" * 64]},
        },
    )
    bad_key_errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="dynamic",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "custom:dynamic:all"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "custom",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "edit_topology": {"module_hashes": {"": "sha256:" + "a" * 64}},
        },
    )

    assert any(
        "edit_topology.module_hashes must be an object" in error
        for error in non_object_errors
    )
    assert any(
        "edit_topology.module_hashes key invalid" in error for error in bad_key_errors
    )
