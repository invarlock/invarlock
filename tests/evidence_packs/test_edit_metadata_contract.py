from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.evidence_packs.python import (
    validate_deployable_artifact as deployable_validator_mod,
)
from scripts.evidence_packs.python.edit_artifact_summary import (
    build_edit_artifact_summary,
)
from scripts.evidence_packs.python.edit_metadata import (
    DEPLOYABLE_OPTIMIZED_SUBJECT,
    EDIT_SEMANTICS_DEPLOYABLE,
    build_edit_metadata,
    build_validation_edit_metadata,
    validate_edit_metadata,
)


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


def test_validate_edit_artifact_require_metadata_json(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    metadata = build_validation_edit_metadata(
        edit_type="quant_rtn",
        scope="ffn",
        parameters={"bits": 4, "group_size": 32},
        coverage={"edited_tensors": 1, "edited_params": 1, "total_params": 1},
    )
    _write_minimal_artifact(artifact, metadata)

    script = Path("scripts/evidence_packs/python/validate_edit_artifact.py")
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


def test_edit_artifact_summary_counts_scenario_taxonomy(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    report_dir = pack_dir / "reports" / "model" / "quant_4bit_clean" / "run_1"
    report_dir.mkdir(parents=True)
    metadata = build_validation_edit_metadata(
        edit_type="quant_rtn",
        scope="ffn",
        parameters={"bits": 4, "group_size": 32},
        coverage={"edited_tensors": 1, "edited_params": 1, "total_params": 1},
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
