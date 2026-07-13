from __future__ import annotations

import json
from pathlib import Path

from scripts.evidence_packs.python.editing.implementations import (
    build_validation_edit_metadata,
)
from scripts.evidence_packs.python.task_tools import build_edit_artifact_summary


def test_edit_artifact_summary_counts_scenario_taxonomy(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    report_dir = pack_dir / "reports" / "model" / "quant_4bit_clean" / "run_1"
    report_dir.mkdir(parents=True)
    metadata = build_validation_edit_metadata(
        edit_type="quant_rtn",
        scope="ffn",
        parameters={"bits": 4, "group_size": 32},
        coverage={"edited_tensors": 1, "edited_params": 1, "total_params": 1},
        edit_provenance={
            "edit_family": "quantization_dequantized",
            "edit_method": "deterministic_rtn",
        },
        edit_impact={"scenario_types": ["target_success"]},
        extra={
            "edit_topology": {"artifact_kind": "checkpoint"},
            "delta_privacy": {"delta_available": "none"},
        },
    )
    (report_dir / "edit_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        json.dumps(
            {
                "schema": "evidence_pack_scenarios_v1",
                "schema_version": 1,
                "scenarios": [
                    {
                        "id": "quant_4bit_clean",
                        "category": "clean",
                        "artifact_class": "validation_subject_checkpoint",
                        "failure_class": "common_edit.quant_rtn",
                        "generation": {
                            "kind": "edit",
                            "edit_spec": "quant_rtn:clean:ffn",
                        },
                    },
                    {
                        "id": "nan_injection",
                        "category": "error_injection",
                        "artifact_class": "fault_injection_fixture",
                        "generation": {"kind": "error"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_edit_artifact_summary(pack_dir, scenarios)

    assert summary["counts"]["validation_subject_checkpoint"] == 1
    assert summary["counts"]["fault_injection_fixture"] == 1
    assert summary["by_scenario"]["quant_4bit_clean"]["metadata_present"] is True
    assert (
        summary["by_scenario"]["quant_4bit_clean"]["storage_format"]
        == "float_dequantized"
    )
    assert summary["by_scenario"]["quant_4bit_clean"]["edit_provenance"] == {
        "edit_family": "quantization_dequantized",
        "edit_method": "deterministic_rtn",
    }
    assert summary["by_scenario"]["quant_4bit_clean"]["edit_impact"] == {
        "scenario_types": ["target_success"]
    }
    assert summary["by_scenario"]["quant_4bit_clean"]["edit_topology"] == {
        "artifact_kind": "checkpoint"
    }
    assert summary["by_scenario"]["quant_4bit_clean"]["delta_privacy"] == {
        "delta_available": "none"
    }


def test_edit_artifact_summary_reports_deployable_smokes(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    report_dir = pack_dir / "reports" / "model" / "deploy_bnb_8bit_clean" / "run_1"
    report_dir.mkdir(parents=True)
    (report_dir / "runtime_deployability_validation.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-artifact-validation-v1",
                "ok": True,
                "backend": "bitsandbytes",
                "validation_scope": "runtime_reproof",
                "runtime_proof_authoritative": True,
                "runtime_proof": {"packed_module_reloaded": True},
                "load_smoke": True,
                "inference_smoke": True,
            }
        ),
        encoding="utf-8",
    )
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        json.dumps(
            {
                "schema": "evidence_pack_scenarios_v1",
                "schema_version": 1,
                "scenarios": [
                    {
                        "id": "deploy_bnb_8bit_clean",
                        "category": "deployable_clean",
                        "artifact_class": "deployable_optimized_subject",
                        "generation": {
                            "kind": "deployable_edit",
                            "backend": "bitsandbytes",
                            "edit_spec": "bnb_8bit:clean:ffn",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_edit_artifact_summary(pack_dir, scenarios)

    assert summary["deployable_subjects"]["backends"] == ["bitsandbytes"]
    assert summary["deployable_subjects"]["all_reload_smokes_passed"] is True
    assert summary["deployable_subjects"]["all_inference_smokes_passed"] is True


def test_edit_artifact_summary_ignores_structural_deployable_receipt(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    report_dir = pack_dir / "reports" / "model" / "deploy_bnb_8bit_clean" / "run_1"
    report_dir.mkdir(parents=True)
    (report_dir / "deployable_artifact_validation.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-artifact-validation-v1",
                "ok": True,
                "backend": "bitsandbytes",
                "validation_scope": "structural_only",
                "runtime_proof_authoritative": False,
                "load_smoke": True,
                "inference_smoke": True,
            }
        ),
        encoding="utf-8",
    )
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        json.dumps(
            {
                "schema": "evidence_pack_scenarios_v1",
                "schema_version": 1,
                "scenarios": [
                    {
                        "id": "deploy_bnb_8bit_clean",
                        "category": "deployable_clean",
                        "artifact_class": "deployable_optimized_subject",
                        "generation": {
                            "kind": "deployable_edit",
                            "backend": "bitsandbytes",
                            "edit_spec": "bnb_8bit:clean:ffn",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_edit_artifact_summary(pack_dir, scenarios)

    assert (
        "deployable_validation_ok"
        not in summary["by_scenario"]["deploy_bnb_8bit_clean"]
    )
    assert summary["deployable_subjects"]["all_reload_smokes_passed"] is False
    assert summary["deployable_subjects"]["all_inference_smokes_passed"] is False
