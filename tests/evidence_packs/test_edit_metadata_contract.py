from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from invarlock.evidence_pack_edit_common import (
    EDIT_PROVENANCE_FAMILIES as PACKAGE_EDIT_PROVENANCE_FAMILIES,
)
from scripts.evidence_packs.python import (
    task_tools_reports,
)
from scripts.evidence_packs.python.editing import (
    validate_artifact as edit_artifact_mod,
)
from scripts.evidence_packs.python.editing.edit_metadata_contract import (
    EDIT_PROVENANCE_FAMILIES as PRODUCER_EDIT_PROVENANCE_FAMILIES,
)
from scripts.evidence_packs.python.editing.implementations import (
    DEPLOYABLE_OPTIMIZED_SUBJECT,
    EDIT_SEMANTICS_DEPLOYABLE,
    EVIDENCE_ONLY_PACK,
    build_edit_metadata,
    build_validation_edit_metadata,
    normalize_coverage,
    parse_edit_specs_json,
    read_edit_metadata,
    validate_edit_metadata,
    write_edit_metadata,
)

save_artifact_mod = edit_artifact_mod
deployable_validator_mod = edit_artifact_mod
_REAL_COVERAGE = {"edited_tensors": 1, "edited_params": 1, "total_params": 1}


def test_edit_provenance_taxonomy_has_one_package_owned_definition() -> None:
    assert PRODUCER_EDIT_PROVENANCE_FAMILIES is PACKAGE_EDIT_PROVENANCE_FAMILIES


def _write_minimal_artifact(path: Path, metadata: dict[str, object] | None) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "pytorch_model.bin").write_text("weights", encoding="utf-8")
    if metadata is not None:
        (path / "edit_metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )


def test_validation_edit_metadata_has_contract_fields() -> None:
    metadata = build_validation_edit_metadata(
        edit_type="magnitude_prune",
        scope="ffn",
        parameters={"target_sparsity": 0.1},
        coverage={"edited_tensors": 2, "edited_params": 10, "total_params": 100},
    )

    assert metadata["schema"] == "invarlock/evidence-pack-edit-metadata-v1"
    assert metadata["artifact_class"] == "validation_subject_checkpoint"
    assert metadata["storage_format"] == "dense_float_with_zeros"
    assert metadata["optimized_deployment_backend"] is False
    assert metadata["packed_quantized_storage"] is False
    assert validate_edit_metadata(metadata) == []


def test_validation_edit_metadata_rejects_impossible_coverage_claims() -> None:
    metadata = build_validation_edit_metadata(
        edit_type="lora_merge",
        scope="attn",
        coverage=_REAL_COVERAGE,
    )
    metadata["coverage"] = {
        "edited_tensors": -1,
        "edited_params": 1.5,
        "total_params": 0,
        "coverage_ratio": 42.0,
    }

    errors = validate_edit_metadata(metadata)

    assert any(
        "edited_tensors must be a non-negative integer" in error for error in errors
    )
    assert any(
        "edited_params must be a non-negative integer" in error for error in errors
    )
    assert any(
        "coverage_ratio must be finite and between 0 and 1" in error for error in errors
    )


def test_validation_edit_metadata_requires_exact_coverage_ratio() -> None:
    metadata = build_validation_edit_metadata(
        edit_type="lora_merge",
        scope="attn",
        coverage={"edited_tensors": 1, "edited_params": 1, "total_params": 3},
    )
    metadata["coverage"]["coverage_ratio"] = (1 / 3) + 1e-15

    errors = validate_edit_metadata(metadata)

    assert "coverage.coverage_ratio must equal edited_params / total_params" in errors


@pytest.mark.parametrize(
    "edit_type", ("fp8_quant", "lowrank_svd", "FP8-QUANT", "LOWRANK-SVD")
)
def test_unverifiable_generated_metadata_cannot_bypass_storage_contract(
    edit_type: str,
) -> None:
    with pytest.raises(ValueError, match="dedicated storage and replay contract"):
        build_edit_metadata(
            edit_type=edit_type,
            scope="ffn",
            storage_format="forged_generic_storage",
            actual_storage_format="forged_generic_storage",
        )

    forged = build_validation_edit_metadata(
        edit_type="quant_rtn", scope="ffn", coverage=_REAL_COVERAGE
    )
    forged.update(
        {
            "edit_type": edit_type,
            "storage_format": "forged_generic_storage",
            "actual_storage_format": "forged_generic_storage",
        }
    )

    errors = validate_edit_metadata(forged)

    assert any("dedicated storage and replay contract" in error for error in errors)


def test_edit_spec_json_and_coverage_normalization_reject_ambiguous_inputs() -> None:
    assert parse_edit_specs_json('[{"spec": "quant_rtn:4:32:ffn"}]') == [
        {"spec": "quant_rtn:4:32:ffn"}
    ]
    with pytest.raises(ValueError, match="Invalid edit_specs JSON"):
        parse_edit_specs_json("{")
    with pytest.raises(ValueError, match="must be a JSON list"):
        parse_edit_specs_json('{"spec": "quant_rtn:4:32:ffn"}')

    assert normalize_coverage(None) == {
        "edited_tensors": 0,
        "edited_params": 0,
        "total_params": 0,
        "coverage_ratio": 0.0,
    }
    with pytest.raises(ValueError, match="edited_count is unsupported"):
        normalize_coverage(
            {
                "edited_count": "2",
                "edited_params": "3",
                "total_params": "4",
            }
        )
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        normalize_coverage(
            {
                "edited_tensors": -2,
                "edited_params": object(),
                "total_params": float("inf"),
                "coverage_ratio": 9.0,
            }
        )


def test_zero_coverage_is_only_producible_for_an_explicit_no_model_route() -> None:
    with pytest.raises(ValueError, match="positive for a proof-routed model edit"):
        build_validation_edit_metadata(edit_type="quant_rtn", scope="ffn")

    metadata = build_edit_metadata(
        edit_type="noop",
        scope="none",
        artifact_class=EVIDENCE_ONLY_PACK,
    )

    assert metadata["coverage"] == {
        "edited_tensors": 0,
        "edited_params": 0,
        "total_params": 0,
        "coverage_ratio": 0.0,
    }
    assert validate_edit_metadata(metadata) == []


def test_edit_metadata_file_round_trip_and_non_object_rejection(tmp_path: Path) -> None:
    metadata_path = tmp_path / "nested" / "edit_metadata.json"
    metadata = build_validation_edit_metadata(
        edit_type="quant_rtn", scope="ffn", coverage=_REAL_COVERAGE
    )

    write_edit_metadata(metadata_path, metadata)

    assert read_edit_metadata(metadata_path) == metadata
    assert metadata_path.read_text(encoding="utf-8").endswith("\n")

    metadata_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        read_edit_metadata(metadata_path)


def test_validation_contract_reports_independent_subject_and_deployable_failures() -> (
    None
):
    subject = build_validation_edit_metadata(
        edit_type="quant_rtn", scope="ffn", coverage=_REAL_COVERAGE
    )
    subject.update(
        {
            "schema": "unknown",
            "artifact_class": "unknown",
            "edit_type": "",
            "coverage": None,
            "deployable_as_hf_checkpoint": False,
        }
    )
    subject_errors = validate_edit_metadata(
        subject,
        expected_edit_type="magnitude_prune",
        expected_artifact_class="validation_subject_checkpoint",
    )
    assert any("unknown edit metadata schema" in error for error in subject_errors)
    assert any("invalid artifact_class" in error for error in subject_errors)
    assert any("artifact_class mismatch" in error for error in subject_errors)
    assert any("edit_type must be" in error for error in subject_errors)
    assert any("edit_type mismatch" in error for error in subject_errors)
    assert "coverage must be an object" in subject_errors

    deployable = _deployable_metadata()
    deployable.update(
        {
            "optimized_deployment_backend": False,
            "packed_quantized_storage": False,
            "backend": "",
            "deployable_as_hf_checkpoint": False,
        }
    )
    deployable_errors = validate_edit_metadata(deployable)
    assert (
        "deployable artifacts must set optimized_deployment_backend=true"
        in deployable_errors
    )
    assert "deployable artifacts must set packed_quantized_storage=true" in (
        deployable_errors
    )
    assert "deployable artifacts must record a backend" in deployable_errors
    assert any("deployable_as_hf_checkpoint=true" in e for e in deployable_errors)


def test_validation_edit_metadata_accepts_optional_provenance_and_impact() -> None:
    metadata = build_validation_edit_metadata(
        edit_type="lora_merge",
        scope="attn",
        parameters={"rank": 4},
        coverage={"edited_tensors": 2, "edited_params": 10, "total_params": 100},
        edit_provenance={
            "edit_family": "lora_merge",
            "edit_method": "custom",
            "edit_count": 1,
            "target_set_digest": "sha256:" + "a" * 64,
            "dynamic_runtime_required": False,
        },
        edit_impact={
            "scenario_types": [
                "target_success",
                "near_neighbor",
                "unrelated_locality",
            ]
        },
    )

    assert metadata["edit_provenance"]["edit_family"] == "lora_merge"
    assert metadata["edit_impact"]["scenario_types"] == [
        "target_success",
        "near_neighbor",
        "unrelated_locality",
    ]
    assert validate_edit_metadata(metadata) == []


def test_validate_edit_metadata_rejects_malformed_topology_and_delta_privacy() -> None:
    metadata = build_validation_edit_metadata(
        edit_type="lora_merge",
        scope="attn",
        coverage=_REAL_COVERAGE,
        extra={
            "edit_topology": {
                "artifact_kind": "raw_delta",
                "module_hashes": {"generator": "bad"},
                "runtime_activation_policy": "",
                "training_or_edit_data_ref": "",
            },
            "delta_privacy": {
                "delta_available": "raw_everywhere",
                "privacy_sensitivity": "none",
                "public_raw_delta_approved": "false",
            },
        },
    )

    errors = validate_edit_metadata(metadata)

    assert any("edit_topology.artifact_kind" in error for error in errors)
    assert any("edit_topology.module_hashes.generator" in error for error in errors)
    assert any("edit_topology.runtime_activation_policy" in error for error in errors)
    assert any("edit_topology.training_or_edit_data_ref" in error for error in errors)
    assert any("delta_privacy.delta_available" in error for error in errors)
    assert any("delta_privacy.privacy_sensitivity" in error for error in errors)
    assert any("delta_privacy.public_raw_delta_approved" in error for error in errors)


def test_validate_edit_metadata_rejects_malformed_optional_provenance() -> None:
    metadata = build_validation_edit_metadata(
        edit_type="custom",
        scope="all",
        coverage=_REAL_COVERAGE,
        edit_provenance={
            "edit_family": "unsupported_edit_family",
            "edit_count": 0,
            "target_set_digest": "bad",
            "dynamic_runtime_required": "false",
        },
        edit_impact={"scenario_types": ["target_success", "unsupported_scenario_type"]},
    )

    errors = validate_edit_metadata(metadata)

    assert any("edit_provenance.edit_family" in error for error in errors)
    assert any("edit_provenance.edit_count" in error for error in errors)
    assert any("edit_provenance.target_set_digest" in error for error in errors)
    assert any("edit_provenance.dynamic_runtime_required" in error for error in errors)
    assert any("edit_impact.scenario_types[1]" in error for error in errors)


def test_validate_edit_metadata_rejects_non_string_optional_taxonomy_values() -> None:
    metadata = build_validation_edit_metadata(
        edit_type="custom",
        scope="all",
        coverage=_REAL_COVERAGE,
        edit_provenance={"edit_family": ["lora_merge"]},
        edit_impact={"scenario_types": ["target_success", {"kind": "bad"}]},
    )

    errors = validate_edit_metadata(metadata)

    assert any("edit_provenance.edit_family" in error for error in errors)
    assert any("edit_impact.scenario_types[1]" in error for error in errors)


def test_task_tools_report_json_and_scenario_helpers(tmp_path: Path) -> None:
    assert task_tools_reports._load_json_optional(None) is None

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert task_tools_reports._load_json_optional(bad_json) is None

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert task_tools_reports._load_json_optional(list_json) is None

    object_json = tmp_path / "object.json"
    object_json.write_text('{"ok": true}', encoding="utf-8")
    assert task_tools_reports._load_json_optional(object_json) == {"ok": True}
    assert task_tools_reports._load_json_object(bad_json) == {}
    assert task_tools_reports._load_json_object(list_json) == {}

    pack_dir = tmp_path / "pack"
    metadata_path = (
        pack_dir / "reports" / "model" / "edit" / "run_1" / "edit_metadata.json"
    )
    error_path = (
        pack_dir
        / "reports"
        / "model"
        / "errors"
        / "nan_injection"
        / "edit_metadata.json"
    )
    outside = tmp_path / "outside" / "edit_metadata.json"
    short = pack_dir / "reports" / "model" / "edit_metadata.json"

    assert (
        task_tools_reports._scenario_from_report_metadata(pack_dir, metadata_path)
        == "edit"
    )
    assert (
        task_tools_reports._scenario_from_report_metadata(pack_dir, error_path)
        == "nan_injection"
    )
    assert task_tools_reports._scenario_from_report_metadata(pack_dir, outside) is None
    assert task_tools_reports._scenario_from_report_metadata(pack_dir, short) is None

    assert (
        task_tools_reports._scenario_artifact_class({"generation": {"kind": "error"}})
        == "fault_injection_fixture"
    )
    assert (
        task_tools_reports._scenario_artifact_class(
            {"generation": {"kind": "deployable_edit"}}
        )
        == "deployable_optimized_subject"
    )
    assert (
        task_tools_reports._scenario_artifact_class({"generation": {"kind": "edit"}})
        == "validation_subject_checkpoint"
    )
    assert task_tools_reports._scenario_artifact_class({}) == "unknown"


def test_task_tools_structural_failure_report_helpers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source report is required"):
        task_tools_reports._build_structural_base_report(None)

    source_report = {
        "run_id": "run-1",
        "meta": {"seed": 42},
        "data": {
            "dataset": "local_jsonl",
            "seq_len": "512",
            "preview_n": "bad",
            "final_n": -1,
        },
        "edit": {"name": "nan_injection"},
        "metrics": {
            "ppl_preview": 11.0,
            "ppl_final": 12.0,
            "primary_metric": {
                "kind": "accuracy",
                "unit": "fraction",
                "direction": "higher",
                "aggregation_scope": "window",
                "paired": False,
                "gating_basis": "lower",
                "supports_bootstrap": False,
                "drift_band": {"min": 0.9, "max": 1.1},
            },
        },
        "artifacts": {"source": "source/report.json"},
        "evaluation_windows": {"preview": [1], "final": [2]},
        "flags": {"primary_metric": True},
    }

    base_report = task_tools_reports._build_structural_base_report(source_report)
    assert base_report["run_id"] == "run-1"
    assert base_report["dataset"]["windows"]["preview"] == 0
    assert base_report["dataset"]["windows"]["final"] == 0
    assert base_report["primary_metric"]["kind"] == "accuracy"
    assert base_report["primary_metric"]["preview"] == 11.0
    assert base_report["primary_metric"]["final"] == 12.0
    assert base_report["primary_metric"]["drift_band"] == {"min": 0.9, "max": 1.1}

    payload = task_tools_reports.build_structural_failure_report(
        error_type="nan_injection",
        message="simulated",
        base_report={**base_report, "validation": "bad", "guard_metric_impact": "bad"},
        source_report=source_report,
        source_report_path="source/report.json",
        edited_report_path="edited/report.json",
        edited_events_path="edited/events.jsonl",
    )
    assert payload["run_id"] == "run-1-structural-failure-nan_injection"
    assert payload["validation"]["invariants_pass"] is False
    assert payload["guard_metric_impact"]["evaluated"] is False
    assert payload["guard_metric_impact"]["passed"] is False
    assert payload["validation"]["guard_metric_impact_acceptable"] is False
    assert payload["primary_metric"]["degraded_reason"] == "structural_failure"
    assert payload["invariants"]["status"] == "fail"
    assert payload["spectral"]["status"] == "structural_failure"
    assert payload["rmt"]["status"] == "structural_failure"

    out_path = tmp_path / "report" / "evaluation.report.json"
    task_tools_reports._write_structural_runtime_manifest(
        out_path=out_path,
        source_runtime_manifest=None,
        error_type="nan_injection",
        message="simulated",
    )
    assert not (out_path.parent / "runtime.manifest.json").exists()

    out_path.parent.mkdir(parents=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    task_tools_reports._write_structural_runtime_manifest(
        out_path=out_path,
        source_runtime_manifest={"schema": "invarlock/runtime-manifest-v1"},
        error_type="nan_injection",
        message="simulated",
    )
    runtime_manifest = json.loads(
        (out_path.parent / "runtime.manifest.json").read_text(encoding="utf-8")
    )
    assert runtime_manifest["report"]["filename"] == "evaluation.report.json"
    assert runtime_manifest["context"]["evidence_pack_structural_failure"] == {
        "error_type": "nan_injection",
        "message": "simulated",
    }


def test_validate_edit_artifact_require_metadata_json(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    metadata = build_validation_edit_metadata(
        edit_type="quant_rtn",
        scope="ffn",
        parameters={"bits": 4, "group_size": 32},
        coverage={"edited_tensors": 1, "edited_params": 1, "total_params": 1},
    )
    _write_minimal_artifact(artifact, metadata)

    script = Path("scripts/evidence_packs/python/editing/validate_artifact.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(artifact),
            "--require-metadata",
            "--expected-edit-type",
            "quant_rtn",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["has_metadata"] is True
    assert payload["artifact_class"] == "validation_subject_checkpoint"


def _deployable_metadata() -> dict[str, object]:
    return build_edit_metadata(
        edit_type="bnb_8bit",
        scope="ffn",
        artifact_class=DEPLOYABLE_OPTIMIZED_SUBJECT,
        edit_semantics=EDIT_SEMANTICS_DEPLOYABLE,
        optimized_deployment_backend=True,
        backend="bitsandbytes",
        storage_format="bitsandbytes_8bit_packed",
        actual_storage_format="bitsandbytes_8bit_packed",
        packed_quantized_storage=True,
        runtime_memory_reduction=True,
        runtime_memory_reduction_expected=True,
        parameters={"bits": 8},
        coverage={"edited_tensors": 1, "edited_params": 1, "total_params": 1},
    )
