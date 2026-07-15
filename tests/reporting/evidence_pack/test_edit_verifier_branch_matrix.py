from __future__ import annotations

import json
from pathlib import Path

import pytest

from invarlock.evidence_pack import EvidencePackStatus, verify_evidence_pack
from invarlock.evidence_pack_edit_common import (
    EDIT_METADATA_SCHEMA,
    VALIDATION_SUBJECT_CHECKPOINT,
    _expected_literal_pruning_params,
    _load_json_sidecar,
    _report_model_name,
    _typed_scenario_index_from_pack,
)
from invarlock.evidence_pack_edit_validation import _metadata_consistency_errors
from invarlock.evidence_pack_edit_verifier import _verify_edit_metadata_consistency
from tests.reporting._support_evidence_pack_paths import _build_pack


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _quant_scenario(scenario_id: str = "quant") -> dict[str, object]:
    return {
        "id": scenario_id,
        "artifact_class": VALIDATION_SUBJECT_CHECKPOINT,
        "strictness": "informational",
        "generation": {
            "kind": "edit",
            "edit_spec": "quant_rtn:4:2:ffn",
            "version": "stress",
        },
    }


def _quant_metadata() -> dict[str, object]:
    return {
        "schema": EDIT_METADATA_SCHEMA,
        "artifact_class": VALIDATION_SUBJECT_CHECKPOINT,
        "edit_type": "quant_rtn",
        "optimized_deployment_backend": False,
        "packed_quantized_storage": False,
        "coverage": {
            "edited_tensors": 1,
            "edited_params": 4,
            "total_params": 8,
            "coverage_ratio": 0.5,
        },
    }


def test_typed_scenario_index_rejects_invalid_file_duplicate_and_wrong_shape(
    tmp_path: Path,
) -> None:
    scenarios_path = tmp_path / "metadata" / "scenarios.json"
    scenarios_path.parent.mkdir(parents=True)
    scenarios_path.write_text("{", encoding="utf-8")
    assert _typed_scenario_index_from_pack(tmp_path)[2][0].startswith(
        "metadata/scenarios.json is invalid:"
    )

    _write_json(scenarios_path, {"scenarios": {"id": "not-a-list"}})
    assert _typed_scenario_index_from_pack(tmp_path)[2] == [
        "metadata/scenarios.json must contain a scenarios list"
    ]

    scenario = _quant_scenario()
    _write_json(scenarios_path, {"scenarios": [scenario, scenario]})
    records, contracts, errors = _typed_scenario_index_from_pack(tmp_path)
    assert set(records) == {"quant"}
    assert set(contracts) == {"quant"}
    assert errors == ["metadata/scenarios.json has duplicate scenario id: quant"]


def test_json_sidecar_and_report_model_helpers_reject_untrusted_shapes(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("[1]", encoding="utf-8")
    assert _load_json_sidecar(sidecar) == (
        None,
        "JSON sidecar must contain an object",
    )

    outside = tmp_path.parent / "outside" / "evaluation.report.json"
    assert _report_model_name(tmp_path, outside) is None
    assert (
        _report_model_name(
            tmp_path,
            tmp_path
            / "reports"
            / "model"
            / "errors"
            / "scenario"
            / "evaluation.report.json",
        )
        is None
    )


@pytest.mark.parametrize(
    ("edit_spec", "expected"),
    [
        (None, (None, None, None)),
        (
            "magnitude_prune:not-a-number:ffn",
            (None, None, "magnitude_prune scenario sparsity is invalid"),
        ),
        (
            "magnitude_prune:1.0:ffn",
            (None, None, "magnitude_prune scenario sparsity must be in (0, 1)"),
        ),
        (
            "magnitude_prune:0.5:",
            (None, None, "magnitude_prune scenario scope is missing"),
        ),
        ("magnitude_prune:0.5:ffn", (0.5, "ffn", None)),
    ],
)
def test_literal_pruning_parameter_parser_fails_closed(
    edit_spec: str | None,
    expected: tuple[float | None, str | None, str | None],
) -> None:
    spec: dict[str, object] = {
        "generation": {"edit_spec": edit_spec}
        if edit_spec is not None
        else {"edit_spec": []}
    }
    assert _expected_literal_pruning_params(spec) == expected


def test_metadata_validation_reports_exact_pruning_contract_failures() -> None:
    metadata = {
        "schema": EDIT_METADATA_SCHEMA,
        "artifact_class": VALIDATION_SUBJECT_CHECKPOINT,
        "edit_type": "magnitude_prune",
        "scope": "all",
        "parameters": {"target_sparsity": 0.25},
        "optimized_deployment_backend": False,
        "packed_quantized_storage": False,
        "coverage": {
            "edited_tensors": 1,
            "edited_params": 4,
            "total_params": 8,
            "coverage_ratio": 0.5,
        },
    }

    errors = _metadata_consistency_errors(
        scenario_id="prune",
        spec={
            "artifact_class": VALIDATION_SUBJECT_CHECKPOINT,
            "generation": {"edit_spec": "magnitude_prune:not-a-number:ffn"},
        },
        metadata=metadata,
    )

    assert errors == [
        "prune: magnitude_prune scenario sparsity is invalid",
    ]


def test_pack_verifier_rejects_symlinked_replay_and_top_level_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = tmp_path / "pack"
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_quant_scenario()]},
    )
    report_dir = pack / "reports" / "model" / "quant" / "run-1"
    _write_json(report_dir / "evaluation.report.json", {})
    _write_json(report_dir / "edit_metadata.json", _quant_metadata())
    external_replay = tmp_path / "external-replay.json"
    _write_json(external_replay, {})
    (report_dir / "transformation_replay.json").symlink_to(external_replay)

    errors = _verify_edit_metadata_consistency(pack)
    assert errors == [
        "quant: transformation replay sidecar missing: transformation_replay.json",
        "quant: active generated transformation scenario has no transformation replay coverage",
    ]

    top_pack = _build_pack(
        tmp_path / "top-pack",
        report_rel_path="reports/model/quant/run-1/evaluation.report.json",
        scenario_strictness="informational",
        scenario_metadata={
            "artifact_class": VALIDATION_SUBJECT_CHECKPOINT,
            "generation": {
                "kind": "edit",
                "edit_spec": "quant_rtn:4:2:ffn",
                "version": "stress",
            },
        },
        report_sidecars={"edit_metadata.json": _quant_metadata()},
    )
    monkeypatch.setenv("INVARLOCK_ALLOW_UNVERIFIED_PROVENANCE", "1")
    result = verify_evidence_pack(
        top_pack,
        skip_verify=True,
        report_assurance="off",
    )
    assert result.status == EvidencePackStatus.INTEGRITY
    assert result.payload["ok"] is False
    assert errors[0] in result.payload["errors"]


def test_pack_verifier_rejects_invalid_report_before_sidecar_trust(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_quant_scenario()]},
    )
    report = pack / "reports" / "model" / "quant" / "run-1" / "evaluation.report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{", encoding="utf-8")
    _write_json(report.parent / "edit_metadata.json", _quant_metadata())

    errors = _verify_edit_metadata_consistency(pack)

    assert len(errors) == 2
    assert errors[0].startswith("quant: evaluation report invalid:")
    assert errors[1] == (
        "quant: active generated transformation scenario has no transformation replay coverage"
    )


def test_pack_verifier_rejects_untyped_report_and_missing_declared_coverage(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_quant_scenario()]},
    )
    _write_json(
        pack / "reports" / "model" / "rogue" / "run-1" / "evaluation.report.json",
        {},
    )

    assert _verify_edit_metadata_consistency(pack) == [
        "rogue: report has no accepted typed scenario",
        "quant: active generated transformation scenario has no evaluation report",
    ]


def test_pack_verifier_rejects_invalid_transformation_sidecar_json(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_quant_scenario()]},
    )
    report_dir = pack / "reports" / "model" / "quant" / "run-1"
    _write_json(report_dir / "evaluation.report.json", {})
    _write_json(report_dir / "edit_metadata.json", _quant_metadata())
    (report_dir / "transformation_replay.json").write_text("{", encoding="utf-8")

    errors = _verify_edit_metadata_consistency(pack)
    assert len(errors) == 2
    assert errors[0].startswith("quant: transformation replay sidecar invalid:")
    assert errors[1] == (
        "quant: active generated transformation scenario has no transformation replay coverage"
    )


def test_pack_verifier_rejects_object_without_transformation_binding_fields(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_quant_scenario()]},
    )
    report_dir = pack / "reports" / "model" / "quant" / "run-1"
    _write_json(report_dir / "evaluation.report.json", {})
    _write_json(report_dir / "edit_metadata.json", _quant_metadata())
    _write_json(report_dir / "transformation_replay.json", {})

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "transformation replay has missing required fields" in error for error in errors
    )
    assert not any("has no transformation replay coverage" in error for error in errors)


def test_deployable_pack_runs_cross_sidecar_binding_after_all_files_load(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    scenario = {
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
    }
    _write_json(pack / "metadata" / "scenarios.json", {"scenarios": [scenario]})
    report_dir = pack / "reports" / "model" / "deploy" / "run-1"
    _write_json(
        report_dir / "evaluation.report.json",
        {
            "meta": {
                "model_identity": {
                    "kind": "local_checkpoint_tree",
                    "sha256": "sha256:" + "0" * 64,
                }
            }
        },
    )
    _write_json(
        report_dir / "edit_metadata.json",
        {
            "schema": EDIT_METADATA_SCHEMA,
            "artifact_class": "deployable_optimized_subject",
            "edit_type": "bnb_8bit",
            "optimized_deployment_backend": True,
            "packed_quantized_storage": True,
        },
    )
    for sidecar in (
        "deployable_artifact_validation.json",
        "runtime_deployability_validation.json",
        "backend_inventory.json",
        "memory_report.json",
        "load_smoke.json",
        "inference_smoke.json",
        "publication_commit.json",
    ):
        _write_json(report_dir / sidecar, {})

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "proof artifact identity does not match evaluation subject identity" in error
        for error in errors
    )
