from __future__ import annotations

from pathlib import Path

import invarlock.evidence_pack_edit_common as edit_common_mod
import invarlock.evidence_pack_edit_validation as edit_validation_mod
import invarlock.evidence_pack_edit_verifier as edit_verifier_mod
from tests.reporting._support_evidence_pack_paths import _write_json


def test_evidence_pack_metadata_helper_edges(tmp_path: Path) -> None:
    assert (
        edit_common_mod._infer_scenario_artifact_class(
            {"artifact_class": "custom_class", "generation": {"kind": "error"}}
        )
        == "custom_class"
    )
    assert (
        edit_common_mod._infer_scenario_artifact_class(
            {"generation": {"kind": "error"}}
        )
        == "fault_injection_fixture"
    )
    assert (
        edit_common_mod._infer_scenario_artifact_class(
            {"generation": {"kind": "deployable_edit"}}
        )
        == "deployable_optimized_subject"
    )
    assert (
        edit_common_mod._infer_scenario_artifact_class({"generation": {"kind": "edit"}})
        == "validation_subject_checkpoint"
    )
    assert edit_common_mod._infer_scenario_artifact_class({}) == ""

    mixed_pack = tmp_path / "mixed-pack"
    (mixed_pack / "metadata").mkdir(parents=True)
    _write_json(
        mixed_pack / "metadata" / "scenarios.json",
        {"scenarios": ["bad", {"id": ""}, {"id": "ok"}]},
    )

    assert (
        edit_common_mod._report_scenario_id(
            mixed_pack,
            tmp_path / "outside" / "evaluation.report.json",
        )
        is None
    )
    assert (
        edit_common_mod._report_scenario_id(
            mixed_pack,
            mixed_pack / "reports" / "evaluation.report.json",
        )
        is None
    )
    assert (
        edit_common_mod._report_scenario_id(
            mixed_pack,
            mixed_pack
            / "reports"
            / "model"
            / "errors"
            / "bad"
            / "evaluation.report.json",
        )
        == "bad"
    )

    invalid_sidecar = tmp_path / "invalid-sidecar.json"
    invalid_sidecar.write_text("{", encoding="utf-8")
    assert edit_common_mod._load_json_sidecar(invalid_sidecar)[0] is None
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("[]", encoding="utf-8")
    assert edit_common_mod._load_json_sidecar(sidecar) == (
        None,
        "JSON sidecar must contain an object",
    )

    assert (
        edit_common_mod._expected_edit_type(
            {"failure_class": "deployable_edit.bnb_8bit"}
        )
        == "bnb_8bit"
    )
    assert edit_common_mod._expected_edit_type({}) == ""

    deployable_errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="deploy",
        spec={
            "artifact_class": "deployable_optimized_subject",
            "generation": {"edit_spec": "bnb_8bit:clean:ffn"},
        },
        metadata={
            "schema": "wrong",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "other",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
        },
    )
    assert any("unrecognized schema" in error for error in deployable_errors)
    assert any("artifact_class mismatch" in error for error in deployable_errors)
    assert any("edit_type mismatch" in error for error in deployable_errors)
    assert any(
        "optimized_deployment_backend=true" in error for error in deployable_errors
    )
    assert any("packed_quantized_storage=true" in error for error in deployable_errors)

    validation_errors = edit_validation_mod._metadata_consistency_errors(
        scenario_id="quant",
        spec={
            "artifact_class": "validation_subject_checkpoint",
            "generation": {"edit_spec": "quant_rtn:clean:ffn"},
        },
        metadata={
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "quant_rtn",
            "optimized_deployment_backend": True,
            "packed_quantized_storage": True,
        },
    )
    assert any(
        "optimized_deployment_backend=false" in error for error in validation_errors
    )
    assert any("packed_quantized_storage=false" in error for error in validation_errors)

    assert (
        edit_validation_mod._metadata_consistency_errors(
            scenario_id="quant",
            spec={
                "artifact_class": "validation_subject_checkpoint",
                "generation": {"edit_spec": "quant_rtn:clean:ffn"},
            },
            metadata={
                "schema": "invarlock/evidence-pack-edit-metadata-v1",
                "artifact_class": "validation_subject_checkpoint",
                "edit_type": "quant_rtn",
                "optimized_deployment_backend": False,
                "packed_quantized_storage": False,
                "coverage": {
                    "edited_tensors": 1,
                    "edited_params": 4,
                    "total_params": 8,
                    "coverage_ratio": 0.5,
                },
            },
        )
        == []
    )


def test_current_edit_metadata_dispatch_cannot_use_raw_scenario_index(
    monkeypatch, tmp_path: Path
) -> None:
    """Only the typed scenario contract may select a verifier proof route."""

    assert not hasattr(edit_verifier_mod, "_scenario_index_from_pack")

    def poisoned_raw_index(_pack_dir: Path) -> dict[str, dict[str, object]]:
        raise AssertionError("raw scenario index must not be called")

    monkeypatch.setattr(
        edit_verifier_mod,
        "_scenario_index_from_pack",
        poisoned_raw_index,
        raising=False,
    )
    pack_dir = tmp_path / "pack"
    _write_json(
        pack_dir / "metadata" / "scenarios.json",
        {"scenarios": [{"id": "raw-only"}]},
    )

    errors = edit_verifier_mod._verify_edit_metadata_consistency(pack_dir)

    assert any(
        "raw-only" in error and "fails closed dispatch" in error for error in errors
    )


def test_evidence_pack_metadata_consistency_helper_edges(tmp_path: Path) -> None:
    assert edit_verifier_mod._verify_edit_metadata_consistency(tmp_path / "empty") == []

    pack_dir = tmp_path / "pack"
    _write_json(
        pack_dir / "metadata" / "scenarios.json",
        {
            "scenarios": [
                {
                    "id": "short",
                    "artifact_class": "fault_injection_fixture",
                    "strictness": "informational",
                    "generation": {"kind": "error", "error_type": "nan_injection"},
                },
                {
                    "id": "fault",
                    "artifact_class": "fault_injection_fixture",
                    "strictness": "informational",
                    "generation": {"kind": "error", "error_type": "nan_injection"},
                },
                {
                    "id": "badmeta",
                    "artifact_class": "validation_subject_checkpoint",
                    "strictness": "informational",
                    "generation": {
                        "kind": "edit",
                        "edit_spec": "quant_rtn:clean",
                        "version": "clean",
                    },
                },
                {
                    "id": "valid_validation",
                    "artifact_class": "validation_subject_checkpoint",
                    "strictness": "informational",
                    "generation": {
                        "kind": "edit",
                        "edit_spec": "quant_rtn:clean",
                        "version": "clean",
                    },
                },
                {
                    "id": "deploy_missing_report",
                    "artifact_class": "deployable_optimized_subject",
                    "strictness": "informational",
                    "optimized_deployment_backend": True,
                    "generation": {
                        "kind": "deployable_edit",
                        "backend": "bitsandbytes",
                        "edit_spec": "bnb_8bit:8:all",
                        "version": "deployable",
                    },
                },
                {
                    "id": "deploy",
                    "artifact_class": "deployable_optimized_subject",
                    "strictness": "informational",
                    "optimized_deployment_backend": True,
                    "generation": {
                        "kind": "deployable_edit",
                        "backend": "bitsandbytes",
                        "edit_spec": "bnb_8bit:8:all",
                        "version": "deployable",
                    },
                },
            ]
        },
    )
    (pack_dir / "reports").mkdir()
    (pack_dir / "reports" / "evaluation.report.json").write_text(
        "{}",
        encoding="utf-8",
    )
    fault_report = pack_dir / "reports" / "model" / "fault" / "run_1"
    fault_report.mkdir(parents=True)
    (fault_report / "evaluation.report.json").write_text("{}", encoding="utf-8")

    badmeta_report = pack_dir / "reports" / "model" / "badmeta" / "run_1"
    badmeta_report.mkdir(parents=True)
    (badmeta_report / "evaluation.report.json").write_text("{}", encoding="utf-8")
    (badmeta_report / "edit_metadata.json").write_text("[", encoding="utf-8")

    valid_report = pack_dir / "reports" / "model" / "valid_validation" / "run_1"
    valid_report.mkdir(parents=True)
    (valid_report / "evaluation.report.json").write_text("{}", encoding="utf-8")
    _write_json(
        valid_report / "edit_metadata.json",
        {
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "validation_subject_checkpoint",
            "edit_type": "quant_rtn",
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
        },
    )

    deploy_report = pack_dir / "reports" / "model" / "deploy" / "run_1"
    deploy_report.mkdir(parents=True)
    (deploy_report / "evaluation.report.json").write_text("{}", encoding="utf-8")
    _write_json(
        deploy_report / "edit_metadata.json",
        {
            "schema": "invarlock/evidence-pack-edit-metadata-v1",
            "artifact_class": "deployable_optimized_subject",
            "edit_type": "bnb_8bit",
            "optimized_deployment_backend": True,
            "packed_quantized_storage": True,
        },
    )
    _write_json(deploy_report / "deployable_artifact_validation.json", {"ok": False})
    (deploy_report / "backend_inventory.json").write_text("[]", encoding="utf-8")
    _write_json(deploy_report / "memory_report.json", {"ok": False})
    _write_json(
        deploy_report / "load_smoke.json",
        {"schema": "invarlock/deployable-load-smoke-v1", "ok": False},
    )
    _write_json(deploy_report / "inference_smoke.json", {"ok": True})

    errors = edit_verifier_mod._verify_edit_metadata_consistency(pack_dir)

    assert any("badmeta: edit_metadata.json invalid" in error for error in errors)
    assert any(
        "deployable sidecar invalid (backend_inventory.json)" in error
        for error in errors
    )
    assert any(
        "deployable sidecar did not pass: deployable_artifact_validation.json" in error
        for error in errors
    )
    assert any(
        "deployable sidecar did not pass: memory_report.json" in error
        for error in errors
    )
    assert any(
        "deployable sidecar schema mismatch (memory_report.json)" in error
        for error in errors
    )
    assert any(
        "deployable sidecar did not pass: load_smoke.json" in error for error in errors
    )
    assert any(
        "deploy_missing_report: deployable scenario has no deployability report sidecars"
        in error
        for error in errors
    )


def test_evidence_pack_metadata_consistency_fails_closed_for_non_mapping_scenarios(
    tmp_path: Path,
) -> None:
    pack_dir = tmp_path / "pack"
    report_dir = pack_dir / "reports" / "model" / "bad" / "run_1"
    report_dir.mkdir(parents=True)
    (report_dir / "evaluation.report.json").write_text("{}", encoding="utf-8")
    _write_json(pack_dir / "metadata" / "scenarios.json", {"scenarios": ["bad"]})

    errors = edit_verifier_mod._verify_edit_metadata_consistency(pack_dir)

    assert errors == ["metadata/scenarios.json scenarios[0] must be an object"]
