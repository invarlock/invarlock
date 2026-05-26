from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.evidence_packs.python.edit_artifact_summary import (
    build_edit_artifact_summary,
)
from scripts.evidence_packs.python.edit_metadata import (
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
    report_dir = pack_dir / "reports" / "model" / "deploy_torchao_int4_clean" / "run_1"
    report_dir.mkdir(parents=True)
    (report_dir / "deployable_artifact_validation.json").write_text(
        json.dumps(
            {
                "schema": "invarlock/deployable-artifact-validation-v1",
                "ok": True,
                "backend": "torchao",
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
                        "id": "deploy_torchao_int4_clean",
                        "category": "deployable_clean",
                        "artifact_class": "deployable_optimized_subject",
                        "generation": {
                            "kind": "deployable_edit",
                            "backend": "torchao",
                            "edit_spec": "torchao_int4:clean:ffn",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_edit_artifact_summary(pack_dir, scenarios)

    assert summary["deployable_subjects"]["backends"] == ["torchao"]
    assert summary["deployable_subjects"]["all_reload_smokes_passed"] is True
    assert summary["deployable_subjects"]["all_inference_smokes_passed"] is True
