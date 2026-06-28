from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evidence_packs.python import (
    create_edits_batch as batch_edit_mod,
)
from scripts.evidence_packs.python import (
    task_tools_reports,
)
from scripts.evidence_packs.python.editing import (
    validate_artifact as edit_artifact_mod,
)
from scripts.evidence_packs.python.editing.implementations import (
    DEPLOYABLE_OPTIMIZED_SUBJECT,
    EDIT_SEMANTICS_DEPLOYABLE,
    build_edit_metadata,
    build_validation_edit_metadata,
    validate_edit_metadata,
)
from scripts.evidence_packs.python.task_tools import (
    build_edit_artifact_summary,
)

save_artifact_mod = edit_artifact_mod
deployable_validator_mod = edit_artifact_mod


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
        base_report={**base_report, "validation": "bad", "guard_overhead": "bad"},
        source_report=source_report,
        source_report_path="source/report.json",
        edited_report_path="edited/report.json",
        edited_events_path="edited/events.jsonl",
    )
    assert payload["run_id"] == "run-1-structural-failure-nan_injection"
    assert payload["validation"]["invariants_pass"] is False
    assert payload["guard_overhead"]["evaluated"] is True
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


def _write_deployable_sidecars(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/backend-inventory-v1",
                "adapter": "hf_bnb",
                "backend": "bitsandbytes",
                "backend_version": "0.1",
                "transformers_version": "1.0",
                "quantization_config": {"bits": 4},
                "quantized_module_count": 1,
                "quantized_module_types": ["bitsandbytes.nn.Linear8bitLt"],
                "device_map": "cuda:0",
                "memory_footprint": {
                    "reported_bytes": 1024,
                    "method": "get_memory_footprint",
                },
                "load_smoke": True,
                "inference_smoke": True,
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "memory_report.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-memory-report-v1",
                "ok": True,
                "runtime_memory_reduction_observed": True,
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "load_smoke.json").write_text(
        json.dumps({"schema": "invarlock/deployable-load-smoke-v1", "ok": True}),
        encoding="utf-8",
    )
    (report_dir / "inference_smoke.json").write_text(
        json.dumps({"schema": "invarlock/deployable-inference-smoke-v1", "ok": True}),
        encoding="utf-8",
    )


def test_validate_deployable_artifact_checks_sidecar_schemas_and_ok(
    monkeypatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "deployable"
    report_dir = tmp_path / "report"
    _write_minimal_artifact(artifact, _deployable_metadata())
    _write_deployable_sidecars(report_dir)
    monkeypatch.setattr(
        deployable_validator_mod,
        "_package_version",
        lambda _package_name: "0.1",
        raising=True,
    )

    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
    )

    assert payload["ok"] is True
    assert payload["load_smoke"] is True
    assert payload["inference_smoke"] is True
    assert payload["runtime_memory_reduction_observed"] is True

    (report_dir / "load_smoke.json").write_text(
        json.dumps({"schema": "invarlock/deployable-load-smoke-v1", "ok": False}),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
    )
    assert payload["ok"] is False
    assert payload["load_smoke"] is False
    assert "load_smoke.json ok must be true" in payload["issues"]

    _write_deployable_sidecars(report_dir)
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(
            {
                "schema": "wrong",
                "backend": "bitsandbytes",
                "load_smoke": True,
                "inference_smoke": True,
                "quantized_module_count": 1,
                "quantized_module_types": [],
                "memory_footprint": {},
            }
        ),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
    )
    assert payload["ok"] is False
    assert any(
        issue.startswith("backend_inventory.json schema mismatch")
        for issue in payload["issues"]
    )

    _write_deployable_sidecars(report_dir)
    backend_inventory = json.loads(
        (report_dir / "backend_inventory.json").read_text(encoding="utf-8")
    )
    backend_inventory["backend"] = "other_backend"
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(backend_inventory),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
    )
    assert payload["ok"] is False
    assert any(
        issue.startswith("backend_inventory.json backend mismatch")
        for issue in payload["issues"]
    )

    _write_deployable_sidecars(report_dir)
    backend_inventory = json.loads(
        (report_dir / "backend_inventory.json").read_text(encoding="utf-8")
    )
    backend_inventory["load_smoke"] = False
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(backend_inventory),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
    )
    assert payload["ok"] is False
    assert "backend_inventory.json load_smoke must be true" in payload["issues"]

    _write_deployable_sidecars(report_dir)
    backend_inventory = json.loads(
        (report_dir / "backend_inventory.json").read_text(encoding="utf-8")
    )
    backend_inventory["inference_smoke"] = False
    (report_dir / "backend_inventory.json").write_text(
        json.dumps(backend_inventory),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
    )
    assert payload["ok"] is False
    assert "backend_inventory.json inference_smoke must be true" in payload["issues"]

    _write_deployable_sidecars(report_dir)
    (report_dir / "inference_smoke.json").write_text(
        json.dumps({"schema": "invarlock/deployable-inference-smoke-v1", "ok": False}),
        encoding="utf-8",
    )
    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=report_dir,
        smoke=True,
    )
    assert payload["ok"] is False
    assert payload["inference_smoke"] is False
    assert "inference_smoke.json ok must be true" in payload["issues"]

    payload = deployable_validator_mod.validate_deployable_artifact(
        artifact,
        backend="bitsandbytes",
        report_dir=None,
        smoke=False,
    )
    assert payload["ok"] is False
    assert payload["load_smoke"] is False
    assert payload["inference_smoke"] is False
    assert "deployable validation requires --report-dir sidecars" in payload["issues"]


def test_save_subject_replace_restores_existing_output_on_swap_failure(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "subject"
    output.mkdir()
    (output / "marker.txt").write_text("original", encoding="utf-8")
    staging = save_artifact_mod.staging_path_for(output)
    staging.mkdir()
    (staging / "marker.txt").write_text("new", encoding="utf-8")
    original_rename = Path.rename

    def _rename_with_staging_failure(self: Path, target: Path) -> Path:
        if self == staging:
            raise OSError("simulated staging swap failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", _rename_with_staging_failure)

    try:
        try:
            save_artifact_mod._replace_output(staging, output)
        except OSError as exc:
            assert "simulated staging swap failure" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("expected staging swap failure")
    finally:
        monkeypatch.setattr(Path, "rename", original_rename)

    assert output.is_dir()
    assert (output / "marker.txt").read_text(encoding="utf-8") == "original"
    assert staging.is_dir()


def test_batch_edit_artifact_can_avoid_model_deepcopy(
    monkeypatch, tmp_path: Path
) -> None:
    class NoDeepcopyModel:
        def __deepcopy__(self, memo: dict[object, object]) -> object:
            raise AssertionError("deepcopy should not be used")

    class Stats:
        edited_tensors = 1

        def coverage_payload(self) -> dict[str, object]:
            return {"edited_tensors": 1, "edited_params": 1, "total_params": 1}

    saved: dict[str, object] = {}

    monkeypatch.setattr(
        batch_edit_mod,
        "apply_rtn_dequantized_simulation",
        lambda model, *, bits, group_size, scope: Stats(),
    )
    monkeypatch.setattr(
        batch_edit_mod,
        "save_edited_subject_artifact",
        lambda **kwargs: saved.update(kwargs),
    )
    monkeypatch.setattr(batch_edit_mod, "_clear_memory", lambda: None)

    model = NoDeepcopyModel()
    batch_edit_mod._create_edit_artifact(
        model=model,
        tokenizer=object(),
        parsed_spec={"type": "quant_rtn", "bits": 4, "group_size": 32, "scope": "ffn"},
        edit_path=tmp_path / "edit",
        clone_model=False,
    )

    assert saved["model"] is model


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
    (report_dir / "deployable_artifact_validation.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-artifact-validation-v1",
                "ok": True,
                "backend": "bitsandbytes",
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
