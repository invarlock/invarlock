from __future__ import annotations

from pathlib import Path

from invarlock.evidence_pack_edit_common import (
    RUNTIME_RELOAD_PROOF_SCHEMA,
    RUNTIME_RELOAD_PROOF_SIDECAR,
)
from invarlock.evidence_pack_edit_verifier import _verify_edit_metadata_consistency
from invarlock.pruning_contract import (
    PRUNING_REPLAY_SCHEMA,
)
from tests.evidence_packs._support_pruning_replay_validation import (
    _metadata,
    _pruning_scenario,
    _replay_payload,
    _write_json,
)


def test_pack_verifier_requires_bound_pruning_replay_and_selection_snapshot(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    report_dir = pack / "reports" / "org__model" / "prune_clean" / "run_1"
    report_dir.mkdir(parents=True)
    (pack / "metadata").mkdir(parents=True)
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_pruning_scenario("prune_clean", "magnitude_prune:clean")]},
    )
    artifact_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "a" * 64,
    }
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "b" * 64,
    }
    _write_json(
        report_dir / "evaluation.report.json",
        {
            "meta": {"model_identity": artifact_identity},
            "baseline_ref": {"model_identity": baseline_identity},
        },
    )
    _write_json(report_dir / "edit_metadata.json", _metadata())

    assert any(
        "pruning replay sidecar missing" in error
        for error in _verify_edit_metadata_consistency(pack)
    )

    (report_dir / "pruning_replay.json").write_text("{", encoding="utf-8")
    assert any(
        "pruning replay sidecar invalid" in error
        for error in _verify_edit_metadata_consistency(pack)
    )
    _write_json(report_dir / "pruning_replay.json", {})
    assert any(
        "pruning_replay.json has unrecognized schema" in error
        for error in _verify_edit_metadata_consistency(pack)
    )

    replay = _replay_payload(
        artifact_identity=artifact_identity,
        baseline_identity=baseline_identity,
    )
    _write_json(report_dir / "pruning_replay.json", replay)

    assert any(
        "runtime reload proof sidecar missing" in error
        for error in _verify_edit_metadata_consistency(pack)
    )
    _write_json(
        report_dir / RUNTIME_RELOAD_PROOF_SIDECAR,
        {
            "schema": RUNTIME_RELOAD_PROOF_SCHEMA,
            "ok": True,
            "replay_schema": PRUNING_REPLAY_SCHEMA,
            "edit_type": "magnitude_prune",
            "artifact_identity": artifact_identity,
            "replay_artifact_identity": artifact_identity,
            "prompt_sha256": "sha256:" + "1" * 64,
            "device": "cpu",
            "input_device": "cpu",
            "reload_runs": 2,
            "token_ids_sha256": "sha256:" + "2" * 64,
            "token_ids_shape": [1, 4],
            "logits_sha256": "sha256:" + "3" * 64,
            "logits_shape": [1, 4, 8],
            "all_logits_finite": True,
            "repeat_deterministic": True,
            "load_diagnostics": {
                "schema": "invarlock/pretrained-load-diagnostics-v1",
                "reloads": [
                    {
                        "unexpected_keys": [],
                        "missing_keys": [],
                        "mismatched_keys": [],
                        "error_msgs": [],
                    },
                    {
                        "unexpected_keys": [],
                        "missing_keys": [],
                        "mismatched_keys": [],
                        "error_msgs": [],
                    },
                ],
            },
            "storage_key_audit": {
                "schema": "invarlock/safetensors-storage-key-audit-v1",
                "reloads": [
                    {
                        "artifact_storage_key_count": 1,
                        "artifact_storage_keys_sha256": "sha256:" + "4" * 64,
                        "model_state_key_count": 2,
                        "model_state_keys_sha256": "sha256:" + "5" * 64,
                        "unexpected_storage_keys": [],
                    },
                    {
                        "artifact_storage_key_count": 1,
                        "artifact_storage_keys_sha256": "sha256:" + "4" * 64,
                        "model_state_key_count": 2,
                        "model_state_keys_sha256": "sha256:" + "5" * 64,
                        "unexpected_storage_keys": [],
                    },
                ],
            },
        },
    )

    errors = _verify_edit_metadata_consistency(pack)
    assert any(
        "clean magnitude-prune v1 selection snapshot is invalid" in error
        for error in errors
    )


def test_pack_verifier_rejects_pruning_replay_identity_drift(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    report_dir = pack / "reports" / "org__model" / "prune_clean" / "run_1"
    report_dir.mkdir(parents=True)
    (pack / "metadata").mkdir(parents=True)
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_pruning_scenario("prune_clean", "magnitude_prune:clean")]},
    )
    artifact_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "a" * 64,
    }
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "b" * 64,
    }
    _write_json(
        report_dir / "evaluation.report.json",
        {
            "meta": {"model_identity": artifact_identity},
            "baseline_ref": {"model_identity": baseline_identity},
        },
    )
    _write_json(report_dir / "edit_metadata.json", _metadata())
    replay = _replay_payload(
        artifact_identity={
            "kind": "local_checkpoint_tree",
            "sha256": "sha256:" + "d" * 64,
        },
        baseline_identity=baseline_identity,
    )
    _write_json(report_dir / "pruning_replay.json", replay)

    assert any(
        "artifact identity mismatch" in error
        for error in _verify_edit_metadata_consistency(pack)
    )


def test_pack_verifier_binds_literal_pruning_scenario_parameters(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    report_dir = pack / "reports" / "model" / "prune_stress" / "run_1"
    report_dir.mkdir(parents=True)
    (pack / "metadata").mkdir(parents=True)
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_pruning_scenario("prune_stress", "magnitude_prune:0.4:all")]},
    )
    artifact_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "a" * 64,
    }
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "b" * 64,
    }
    _write_json(
        report_dir / "evaluation.report.json",
        {
            "meta": {"model_identity": artifact_identity},
            "baseline_ref": {"model_identity": baseline_identity},
        },
    )
    _write_json(
        report_dir / "edit_metadata.json",
        _metadata(scope="ffn", target_sparsity=0.5),
    )
    _write_json(
        report_dir / "pruning_replay.json",
        _replay_payload(
            artifact_identity=artifact_identity,
            baseline_identity=baseline_identity,
            scope="ffn",
            target_sparsity=0.5,
        ),
    )

    errors = _verify_edit_metadata_consistency(pack)

    assert any("scope does not match scenario" in error for error in errors)
    assert any("target_sparsity does not match scenario" in error for error in errors)


def test_pack_verifier_retires_previous_pruning_contract_and_target_manifest(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    report_dir = pack / "reports" / "model" / "prune_stress" / "run_1"
    report_dir.mkdir(parents=True)
    (pack / "metadata").mkdir(parents=True)
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_pruning_scenario("prune_stress", "magnitude_prune:0.5:ffn")]},
    )
    artifact_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "a" * 64,
    }
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "b" * 64,
    }
    _write_json(
        report_dir / "evaluation.report.json",
        {
            "meta": {"model_identity": artifact_identity},
            "baseline_ref": {"model_identity": baseline_identity},
        },
    )
    _write_json(report_dir / "edit_metadata.json", _metadata())
    replay = _replay_payload(
        artifact_identity=artifact_identity,
        baseline_identity=baseline_identity,
    )
    replay["schema"] = "invarlock/magnitude-prune-replay-v3"
    replay["scope_policy"] = "unbound-policy"
    replay["target_manifest_sha256"] = "sha256:" + "d" * 64
    _write_json(report_dir / "pruning_replay.json", replay)

    errors = _verify_edit_metadata_consistency(pack)

    assert any("unrecognized schema" in error for error in errors)
    assert any("scope_policy mismatch" in error for error in errors)
    assert any("target_manifest digest mismatch" in error for error in errors)
