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
