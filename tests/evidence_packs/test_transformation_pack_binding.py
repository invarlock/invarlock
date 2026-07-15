from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invarlock.evidence_pack_edit_common import (
    _RUNTIME_RELOAD_PROOF_FIELDS,
    RUNTIME_RELOAD_PROOF_SIDECAR,
    TRANSFORMATION_MATERIALIZATION_RECEIPT,
    TRANSFORMATION_REPLAY_SIDECAR,
)
from invarlock.evidence_pack_edit_verifier import _verify_edit_metadata_consistency
from scripts.evidence_packs.python.editing.runtime_reload_proof import _PROOF_KEYS
from scripts.evidence_packs.python.editing.streaming_transform import (
    TRANSFORMATION_MATERIALIZATION_RECEIPT as ARTIFACT_MATERIALIZATION_RECEIPT,
)
from scripts.evidence_packs.python.editing.streaming_transform import (
    materialize_transformation_artifact,
)
from scripts.evidence_packs.python.editing.validate_artifact import (
    validate_transformation_artifact,
)
from tests.evidence_packs._support_transformation_pack import (
    _INDEX_DIGEST,
    _OUT_OF_SCOPE_NAME,
    _OUTPUT_DIGEST,
    _TARGET_NAME,
    _digest,
    _make_pack,
    _rewrite_fully_crosslinked_target_manifest,
    _runtime_reload_proof,
    _sha256_file,
    _write_json,
)


def test_pack_runtime_reload_schema_tracks_runtime_producer() -> None:
    assert _RUNTIME_RELOAD_PROOF_FIELDS == _PROOF_KEYS


@pytest.mark.parametrize(
    "edit_type",
    ("quant_rtn", "synthetic_lowrank_delta", "synthetic_dense_update"),
)
def test_pack_verifier_accepts_each_verifier_grade_transformation(
    tmp_path: Path, edit_type: str
) -> None:
    pack, _, _ = _make_pack(tmp_path, edit_type=edit_type)

    assert _verify_edit_metadata_consistency(pack) == []


def test_pack_verifier_rejects_fully_crosslinked_qwen_visual_target(
    tmp_path: Path,
) -> None:
    pack, report_dir, _ = _make_pack(tmp_path)
    _rewrite_fully_crosslinked_target_manifest(
        report_dir,
        target_name="model.visual.layers.0.mlp.up_proj.weight",
    )

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "target_manifest policy violation" in error
        and "outside the independent transformation scope" in error
        for error in errors
    )
    assert not any(
        "target manifest is not covered by source plan" in error
        or "target_manifest metadata mismatch" in error
        or "materialization receipt target_manifest mismatch" in error
        for error in errors
    )


def test_pack_verifier_accepts_an_actual_materialized_and_replayed_transform(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    artifact = tmp_path / "artifact"
    baseline.mkdir()
    _write_json(
        baseline / "config.json", {"model_type": "qwen2", "num_hidden_layers": 2}
    )
    _write_json(baseline / "tokenizer_config.json", {"model_max_length": 128})
    _write_json(baseline / "tokenizer.json", {"version": "1.0"})
    save_file(
        {
            _TARGET_NAME: torch.tensor(
                [[0.37, 0.72], [-0.31, 1.79]], dtype=torch.float32
            ),
            _OUT_OF_SCOPE_NAME: torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
            ),
        },
        baseline / "model.safetensors",
        metadata={"format": "pt"},
    )
    materialize_transformation_artifact(
        baseline_path=baseline,
        output_path=artifact,
        edit_type="quant_rtn",
        parameters={"bits": 4, "group_size": 2},
        scope="ffn",
    )
    replay = validate_transformation_artifact(
        artifact,
        baseline_dir=baseline,
        edit_type="quant_rtn",
        parameters={"bits": 4, "group_size": 2},
        scope="ffn",
    )
    assert replay["ok"] is True, replay["issues"]

    pack = tmp_path / "pack"
    scenario_id = "quant_stress"
    report_dir = pack / "reports" / "org__model" / scenario_id / "run_1"
    report_dir.mkdir(parents=True)
    _write_json(
        pack / "metadata" / "scenarios.json",
        {
            "scenarios": [
                {
                    "id": scenario_id,
                    "artifact_class": "validation_subject_checkpoint",
                    "strictness": "informational",
                    "generation": {
                        "kind": "edit",
                        "edit_spec": "quant_rtn:4:2:ffn",
                        "version": "stress",
                    },
                }
            ]
        },
    )
    _write_json(
        report_dir / "evaluation.report.json",
        {
            "meta": {"model_identity": replay["artifact_identity"]},
            "baseline_ref": {"model_identity": replay["baseline_identity"]},
        },
    )
    shutil.copyfile(artifact / "edit_metadata.json", report_dir / "edit_metadata.json")
    shutil.copyfile(
        artifact / ARTIFACT_MATERIALIZATION_RECEIPT,
        report_dir / TRANSFORMATION_MATERIALIZATION_RECEIPT,
    )
    _write_json(report_dir / TRANSFORMATION_REPLAY_SIDECAR, replay)
    _write_json(
        report_dir / RUNTIME_RELOAD_PROOF_SIDECAR, _runtime_reload_proof(replay)
    )

    assert _verify_edit_metadata_consistency(pack) == []


@pytest.mark.parametrize(
    "edit_type",
    ("quant_rtn", "synthetic_lowrank_delta", "synthetic_dense_update"),
)
def test_pack_verifier_binds_clean_transformation_selection_source(
    tmp_path: Path, edit_type: str
) -> None:
    pack, report_dir, replay = _make_pack(tmp_path, edit_type=edit_type, clean=True)

    assert _verify_edit_metadata_consistency(pack) == []

    receipt = replay["selection_receipt"]
    assert isinstance(receipt, dict)
    receipt["selection_bundle_sha256"] = "sha256:" + "0" * 64
    replay["selection_receipt_sha256"] = _digest(receipt)
    _write_json(report_dir / TRANSFORMATION_REPLAY_SIDECAR, replay)

    errors = _verify_edit_metadata_consistency(pack)
    assert any(
        "selection receipt selection_bundle_sha256 mismatch" in error
        for error in errors
    )


def test_pack_verifier_requires_replay_and_materialization_sidecars(
    tmp_path: Path,
) -> None:
    pack, report_dir, _ = _make_pack(tmp_path)
    (report_dir / TRANSFORMATION_REPLAY_SIDECAR).unlink()

    errors = _verify_edit_metadata_consistency(pack)
    assert any("transformation replay sidecar missing" in error for error in errors)

    pack, report_dir, _ = _make_pack(tmp_path / "second")
    (report_dir / TRANSFORMATION_MATERIALIZATION_RECEIPT).unlink()
    errors = _verify_edit_metadata_consistency(pack)
    assert any("materialization receipt sidecar missing" in error for error in errors)

    pack, report_dir, _ = _make_pack(tmp_path / "third")
    (report_dir / RUNTIME_RELOAD_PROOF_SIDECAR).unlink()
    errors = _verify_edit_metadata_consistency(pack)
    assert any("runtime reload proof sidecar missing" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        ("replay_schema", "invarlock/wrong-replay-v1", "replay schema mismatch"),
        ("input_device", "meta", "input device is invalid"),
        ("reload_runs", 1, "exactly two reloads"),
        ("all_logits_finite", False, "finite logits evidence missing"),
        ("repeat_deterministic", False, "determinism evidence missing"),
        ("load_diagnostics", {}, "load diagnostics are invalid"),
        ("storage_key_audit", {}, "storage-key audit is invalid"),
    ),
)
def test_pack_verifier_rejects_unbound_runtime_reload_proof(
    tmp_path: Path,
    field: str,
    value: object,
    expected_error: str,
) -> None:
    pack, report_dir, _ = _make_pack(tmp_path)
    proof_path = report_dir / RUNTIME_RELOAD_PROOF_SIDECAR
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof[field] = value
    _write_json(proof_path, proof)

    errors = _verify_edit_metadata_consistency(pack)

    assert any(expected_error in error for error in errors)


def test_pack_verifier_rejects_impossible_runtime_storage_key_counts(
    tmp_path: Path,
) -> None:
    pack, report_dir, _ = _make_pack(tmp_path)
    proof_path = report_dir / RUNTIME_RELOAD_PROOF_SIDECAR
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    storage_key_audit = proof["storage_key_audit"]
    assert isinstance(storage_key_audit, dict)
    reloads = storage_key_audit["reloads"]
    assert isinstance(reloads, list)
    for audit in reloads:
        assert isinstance(audit, dict)
        audit["artifact_storage_key_count"] = 3
        audit["model_state_key_count"] = 2
    _write_json(proof_path, proof)

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "more artifact storage keys than model state keys" in error for error in errors
    )


def test_pack_verifier_rejects_runtime_reload_proof_identity_drift(
    tmp_path: Path,
) -> None:
    pack, report_dir, _ = _make_pack(tmp_path)
    proof_path = report_dir / RUNTIME_RELOAD_PROOF_SIDECAR
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["artifact_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "0" * 64,
    }
    _write_json(proof_path, proof)

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "runtime reload proof artifact_identity mismatch" in error for error in errors
    )


def test_pack_verifier_rejects_digest_only_or_unbound_shard_plans(
    tmp_path: Path,
) -> None:
    pack, report_dir, replay = _make_pack(tmp_path)
    replay.pop("source_shard_plan")
    replay.pop("output_shard_plan")
    _write_json(report_dir / TRANSFORMATION_REPLAY_SIDECAR, replay)

    errors = _verify_edit_metadata_consistency(pack)

    assert any("missing required fields" in error for error in errors)
    assert any("source_shard_plan is invalid" in error for error in errors)
    assert any("output_shard_plan is invalid" in error for error in errors)


def test_pack_verifier_rejects_copied_baseline_with_positive_claims(
    tmp_path: Path,
) -> None:
    pack, report_dir, replay = _make_pack(tmp_path)
    changes = replay["actual_changes"]
    assert isinstance(changes, dict)
    for field in changes:
        changes[field] = 0
    _write_json(report_dir / TRANSFORMATION_REPLAY_SIDECAR, replay)

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "actual_changes.value_changed_params must be positive" in error
        for error in errors
    )
    assert any(
        "actual_changes.byte_changed_params must be positive" in error
        for error in errors
    )


def test_pack_verifier_rejects_report_identity_mismatch(tmp_path: Path) -> None:
    pack, report_dir, _ = _make_pack(tmp_path)
    report_path = report_dir / "evaluation.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["meta"]["model_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "0" * 64,
    }
    _write_json(report_path, report)

    errors = _verify_edit_metadata_consistency(pack)

    assert any("artifact identity mismatch" in error for error in errors)


def test_pack_verifier_binds_output_weight_topology_to_the_shard_plan(
    tmp_path: Path,
) -> None:
    pack, report_dir, replay = _make_pack(tmp_path)
    replacement_weights = {
        "index_sha256": _INDEX_DIGEST,
        "shards": [
            {
                "name": "different-output.safetensors",
                "sha256": _OUTPUT_DIGEST,
            }
        ],
    }
    replay["output_weights"] = {
        "sha256": _digest(replacement_weights),
        **replacement_weights,
    }
    receipt_path = report_dir / TRANSFORMATION_MATERIALIZATION_RECEIPT
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_weights"] = replay["output_weights"]
    _write_json(receipt_path, receipt)
    replay["materialization_receipt_sha256"] = _sha256_file(receipt_path)
    _write_json(report_dir / TRANSFORMATION_REPLAY_SIDECAR, replay)

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "output weights do not match output shard plan" in error for error in errors
    )


def test_pack_verifier_rejects_repeated_transformation_identity_drift(
    tmp_path: Path,
) -> None:
    pack, report_dir, _ = _make_pack(tmp_path)
    duplicate = report_dir.parent / "run_2"
    shutil.copytree(report_dir, duplicate)
    report_path = duplicate / "evaluation.report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["meta"]["model_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "9" * 64,
    }
    _write_json(report_path, report)
    replay_path = duplicate / TRANSFORMATION_REPLAY_SIDECAR
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["artifact_identity"] = report["meta"]["model_identity"]
    replay["baseline_identity"] = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "8" * 64,
    }
    replay["target_manifest_sha256"] = "sha256:" + "7" * 64
    replay["transformation"] = {
        **replay["transformation"],
        "algorithm": "adversarial-algorithm-substitution",
    }
    replay["scope"] = "all"
    _write_json(replay_path, replay)

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "repeated transformation runs disagree on artifact identity" in error
        for error in errors
    )
    assert any(
        "repeated transformation runs disagree on baseline identity" in error
        for error in errors
    )
    assert any(
        "repeated transformation runs disagree on target manifest digest" in error
        for error in errors
    )
    assert any(
        "repeated transformation runs disagree on transformation contract" in error
        for error in errors
    )
    assert any(
        "repeated transformation runs disagree on scope" in error for error in errors
    )


def test_pack_verifier_accepts_identical_repeated_transformation_runs(
    tmp_path: Path,
) -> None:
    pack, report_dir, _ = _make_pack(tmp_path)
    shutil.copytree(report_dir, report_dir.parent / "run_2")

    assert _verify_edit_metadata_consistency(pack) == []


def test_pack_verifier_rejects_transformation_report_without_model_path(
    tmp_path: Path,
) -> None:
    pack, report_dir, _ = _make_pack(tmp_path)
    error_report = pack / "reports" / "org__model" / "errors" / report_dir.parent.name
    shutil.copytree(report_dir, error_report)

    errors = _verify_edit_metadata_consistency(pack)

    assert any("transformation report has no model path" in error for error in errors)


@pytest.mark.parametrize("edit_type", ("fp8_quant", "lowrank_svd"))
def test_pack_verifier_rejects_unsupported_generated_transformation_claims(
    tmp_path: Path, edit_type: str
) -> None:
    pack = tmp_path / "pack"
    _write_json(
        pack / "metadata" / "scenarios.json",
        {
            "scenarios": [
                {
                    "id": "unsupported",
                    "artifact_class": "validation_subject_checkpoint",
                    "strictness": "informational",
                    "generation": {
                        "kind": "edit",
                        "edit_spec": f"{edit_type}:4:ffn",
                        "version": "stress",
                    },
                }
            ]
        },
    )

    errors = _verify_edit_metadata_consistency(pack)

    assert any("unsupported edit type" in error for error in errors)
    assert any(edit_type in error for error in errors)


def test_pack_verifier_requires_every_active_transformation_to_have_coverage(
    tmp_path: Path,
) -> None:
    pack, _, _ = _make_pack(tmp_path, scenario_id="first")
    scenarios_path = pack / "metadata" / "scenarios.json"
    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenarios["scenarios"].append(
        {
            "id": "second",
            "artifact_class": "validation_subject_checkpoint",
            "strictness": "informational",
            "generation": {
                "kind": "edit",
                "edit_spec": "quant_rtn:4:2:ffn",
                "version": "stress",
            },
        }
    )
    _write_json(scenarios_path, scenarios)

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "second: active generated transformation scenario has no evaluation report"
        in error
        for error in errors
    )


def test_pack_verifier_rejects_materialization_plan_mismatch(tmp_path: Path) -> None:
    pack, report_dir, replay = _make_pack(tmp_path)
    receipt_path = report_dir / TRANSFORMATION_MATERIALIZATION_RECEIPT
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["max_output_shard_bytes"] = 2 * 1024 * 1024
    _write_json(receipt_path, receipt)
    replay["materialization_receipt_sha256"] = _sha256_file(receipt_path)
    _write_json(report_dir / TRANSFORMATION_REPLAY_SIDECAR, replay)

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "transformation materialization receipt max_output_shard_bytes mismatch"
        in error
        for error in errors
    )
