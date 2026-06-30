from __future__ import annotations

import invarlock.evidence_pack_edit_metadata as edit_metadata_mod


def test_deployable_artifact_validation_sidecar_accepts_passing_smokes() -> None:
    assert (
        edit_metadata_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="deployable_artifact_validation.json",
            payload={
                "schema": "invarlock/deployable-artifact-validation-v1",
                "ok": True,
                "load_smoke": True,
                "inference_smoke": True,
            },
        )
        == []
    )


def test_backend_inventory_sidecar_accepts_passing_inventory() -> None:
    assert (
        edit_metadata_mod._deployable_sidecar_consistency_errors(
            scenario_id="deploy",
            sidecar="backend_inventory.json",
            payload={
                "schema": "invarlock/backend-inventory-v1",
                "ok": True,
                "load_smoke": True,
                "inference_smoke": True,
                "quantized_module_count": 2,
                "quantized_module_types": ["Linear8bitLt"],
                "memory_footprint": {
                    "reported_bytes": 1024,
                    "method": "get_memory_footprint",
                },
            },
        )
        == []
    )


def test_backend_inventory_sidecar_reports_failed_inventory_contract() -> None:
    errors = edit_metadata_mod._deployable_sidecar_consistency_errors(
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
    assert any(
        "quantized_module_count must be non-negative int" in error for error in errors
    )
    assert any("quantized_module_types must be a list" in error for error in errors)
    assert any("memory_footprint must be an object" in error for error in errors)


def test_optional_edit_provenance_and_impact_metadata_accepts_valid_payload() -> None:
    errors = edit_metadata_mod._metadata_consistency_errors(
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
    errors = edit_metadata_mod._metadata_consistency_errors(
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
    errors = edit_metadata_mod._metadata_consistency_errors(
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
    errors = edit_metadata_mod._metadata_consistency_errors(
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
    errors = edit_metadata_mod._metadata_consistency_errors(
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
    errors = edit_metadata_mod._metadata_consistency_errors(
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
            "edit_impact": {},
        },
    )

    assert errors == []


def test_optional_edit_topology_and_delta_privacy_accepts_valid_payload() -> None:
    errors = edit_metadata_mod._metadata_consistency_errors(
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
    errors = edit_metadata_mod._metadata_consistency_errors(
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
    errors = edit_metadata_mod._metadata_consistency_errors(
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
            "edit_topology": {},
            "delta_privacy": {},
        },
    )

    assert errors == []


def test_optional_edit_topology_reports_non_object_shapes() -> None:
    errors = edit_metadata_mod._metadata_consistency_errors(
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
    non_object_errors = edit_metadata_mod._metadata_consistency_errors(
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
    bad_key_errors = edit_metadata_mod._metadata_consistency_errors(
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
