from __future__ import annotations

import pytest

from scripts.evidence_packs.python.editing.edit_metadata_contract import (
    EDIT_METADATA_SCHEMA,
    SYNTHETIC_DENSE_UPDATE,
    SYNTHETIC_LOWRANK_DELTA,
    VALIDATION_SUBJECT_CHECKPOINT,
    is_unverifiable_generated_edit_type,
    validate_edit_metadata,
)


def _subject(edit_type: str = "quant_rtn") -> dict[str, object]:
    storage = "float_dequantized"
    return {
        "schema": EDIT_METADATA_SCHEMA,
        "artifact_class": VALIDATION_SUBJECT_CHECKPOINT,
        "edit_type": edit_type,
        "coverage": {
            "edited_tensors": 1,
            "edited_params": 1,
            "total_params": 1,
            "coverage_ratio": 1.0,
        },
        "optimized_deployment_backend": False,
        "packed_quantized_storage": False,
        "runtime_memory_reduction": False,
        "backend": None,
        "storage_format": storage,
        "actual_storage_format": storage,
        "deployable_as_hf_checkpoint": True,
    }


@pytest.mark.parametrize("edit_type", [None, 7, object()])
def test_retired_edit_detection_does_not_coerce_non_text_labels(
    edit_type: object,
) -> None:
    assert is_unverifiable_generated_edit_type(edit_type) is False


def test_validation_subject_reports_missing_coverage_and_false_storage_claims() -> None:
    metadata = _subject()
    metadata["coverage"] = {}
    metadata.update(
        {
            "optimized_deployment_backend": True,
            "packed_quantized_storage": True,
            "runtime_memory_reduction": True,
            "backend": "cuda",
            "storage_format": "packed",
            "actual_storage_format": "packed",
        }
    )

    errors = validate_edit_metadata(metadata)

    for field in ("edited_tensors", "edited_params", "total_params", "coverage_ratio"):
        assert f"coverage.{field} missing" in errors
    assert any("optimized_deployment_backend=false" in error for error in errors)
    assert any("packed_quantized_storage=false" in error for error in errors)
    assert any("runtime_memory_reduction=false" in error for error in errors)
    assert "validation artifacts must set backend=null" in errors
    assert sum("storage_format mismatch" in error for error in errors) == 2


def test_synthetic_edit_contract_rejects_missing_and_fabricated_training_proof() -> (
    None
):
    lowrank = _subject(SYNTHETIC_LOWRANK_DELTA)
    lowrank["storage_format"] = "dense_float_with_synthetic_lowrank_delta"
    lowrank["actual_storage_format"] = "dense_float_with_synthetic_lowrank_delta"
    lowrank["edit_provenance"] = {
        "synthetic": False,
        "trained_adapter": True,
        "adapter_merge_performed": True,
    }

    errors = validate_edit_metadata(lowrank)

    assert "synthetic edits require edit_provenance.synthetic=true" in errors
    assert any("trained_adapter=false" in error for error in errors)
    assert any("adapter_merge_performed=false" in error for error in errors)

    dense = _subject(SYNTHETIC_DENSE_UPDATE)
    dense["storage_format"] = "dense_float_with_synthetic_update"
    dense["actual_storage_format"] = "dense_float_with_synthetic_update"
    dense["edit_provenance"] = {
        "synthetic": True,
        "optimization_performed": True,
        "training_data_used": True,
    }
    errors = validate_edit_metadata(dense)
    assert any("optimization_performed=false" in error for error in errors)
    assert any("training_data_used=false" in error for error in errors)

    dense.pop("edit_provenance")
    assert "synthetic edits require edit_provenance" in validate_edit_metadata(dense)


def test_optional_metadata_blocks_reject_wrong_container_shapes_and_values() -> None:
    metadata = _subject()
    metadata.update(
        {
            "edit_provenance": "claimed",
            "edit_impact": "claimed",
            "edit_topology": "claimed",
            "delta_privacy": "claimed",
        }
    )
    errors = validate_edit_metadata(metadata)
    assert "edit_provenance must be an object when present" in errors
    assert "edit_impact must be an object when present" in errors
    assert "edit_topology must be an object when present" in errors
    assert "delta_privacy must be an object when present" in errors

    metadata.update(
        {
            "edit_provenance": {
                "edit_method": " ",
                "editor_artifact_digest": 1,
                "self_edit_data_digest": "BAD",
            },
            "edit_impact": {"scenario_types": "target_success"},
            "edit_topology": {
                "artifact_kind": 1,
                "module_hashes": {"": "bad", "valid": "bad"},
                "runtime_activation_policy": 0,
                "training_or_edit_data_ref": 1,
            },
            "delta_privacy": {
                "delta_available": 1,
                "privacy_sensitivity": 1,
                "public_raw_delta_approved": "yes",
            },
        }
    )
    errors = validate_edit_metadata(metadata)
    expected = (
        "edit_provenance.edit_method",
        "edit_provenance.editor_artifact_digest",
        "edit_provenance.self_edit_data_digest",
        "edit_impact.scenario_types must be a list",
        "edit_topology.artifact_kind",
        "module_hashes key invalid",
        "module_hashes.valid",
        "runtime_activation_policy",
        "training_or_edit_data_ref",
        "delta_privacy.delta_available",
        "delta_privacy.privacy_sensitivity",
        "public_raw_delta_approved",
    )
    assert all(any(fragment in error for error in errors) for fragment in expected)

    metadata["edit_topology"]["module_hashes"] = []
    assert "edit_topology.module_hashes must be an object" in validate_edit_metadata(
        metadata
    )


def test_absent_optional_metadata_and_empty_impact_are_valid() -> None:
    metadata = _subject()
    metadata.update(
        {
            "edit_provenance": None,
            "edit_impact": {"scenario_types": None},
            "edit_topology": {},
            "delta_privacy": None,
        }
    )

    assert validate_edit_metadata(metadata) == []


def test_all_optional_metadata_value_forms_accept_canonical_values() -> None:
    digest = "sha256:" + "a" * 64
    lowrank = _subject(SYNTHETIC_LOWRANK_DELTA)
    lowrank.update(
        {
            "storage_format": "dense_float_with_synthetic_lowrank_delta",
            "actual_storage_format": "dense_float_with_synthetic_lowrank_delta",
            "edit_provenance": {
                "synthetic": True,
                "trained_adapter": False,
                "adapter_merge_performed": False,
                "edit_family": SYNTHETIC_LOWRANK_DELTA,
                "edit_method": "fixture",
                "edit_count": 1,
                "target_set_digest": digest,
                "editor_artifact_digest": digest,
                "self_edit_data_digest": digest,
                "dynamic_runtime_required": False,
            },
            "edit_impact": {"scenario_types": ["target_success"]},
            "edit_topology": {
                "artifact_kind": "checkpoint",
                "module_hashes": {"projection": digest},
                "runtime_activation_policy": {"mode": "always"},
                "training_or_edit_data_ref": "vendored://fixture",
            },
            "delta_privacy": {
                "delta_available": "hash_only",
                "privacy_sensitivity": "public",
                "public_raw_delta_approved": False,
            },
        }
    )
    assert validate_edit_metadata(lowrank) == []

    dense = _subject(SYNTHETIC_DENSE_UPDATE)
    dense.update(
        {
            "storage_format": "dense_float_with_synthetic_update",
            "actual_storage_format": "dense_float_with_synthetic_update",
            "edit_provenance": {
                "synthetic": True,
                "optimization_performed": False,
                "training_data_used": False,
            },
            "edit_topology": {"runtime_activation_policy": "always"},
        }
    )
    assert validate_edit_metadata(dense) == []
