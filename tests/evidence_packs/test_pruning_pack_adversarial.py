from __future__ import annotations

import copy
from pathlib import Path

import pytest

from invarlock.evidence_pack_edit_verifier import _verify_edit_metadata_consistency
from invarlock.evidence_pack_pruning_validation import _pruning_replay_errors
from tests.evidence_packs._support_pruning_replay_validation import (
    _make_targets_noncanonical,
    _metadata,
    _pruning_scenario,
    _replay_payload,
    _self_consistent_pruning_sidecars,
    _target_manifest,
    _write_json,
)


@pytest.mark.parametrize(
    ("scope", "mutate", "expected_error"),
    [
        (
            "all",
            lambda manifest: manifest["targets"][0].update(
                {"name": "model.visual.blocks.0.mlp.up_proj.weight"}
            ),
            "outside the canonical pruning scope",
        ),
        (
            "all",
            lambda manifest: manifest["targets"][0].update(
                {"name": "model.mtp.layers.0.mlp.up_proj.weight"}
            ),
            "outside the canonical pruning scope",
        ),
        (
            "ffn",
            lambda manifest: manifest["targets"][0].update(
                {"name": "model.layers.0.self_attn.q_proj.weight"}
            ),
            "outside the canonical pruning scope",
        ),
        (
            "ffn",
            lambda manifest: manifest.update(
                {"model_type": "gpt_oss", "architecture": "decoder"}
            ),
            "model_type and architecture mismatch",
        ),
        (
            "ffn",
            lambda manifest: manifest.update(
                {
                    "model_type": "unreviewed_architecture",
                    "architecture": "decoder",
                }
            ),
            "model_type and architecture mismatch",
        ),
        (
            "ffn",
            lambda manifest: manifest["targets"].append(
                copy.deepcopy(manifest["targets"][0])
            ),
            "sorted and unique",
        ),
        ("ffn", _make_targets_noncanonical, "sorted and unique"),
    ],
)
def test_generic_pack_rejects_self_consistent_forged_pruning_target_manifests(
    scope: str,
    mutate: object,
    expected_error: str,
) -> None:
    manifest = _target_manifest(scope=scope)
    assert callable(mutate)
    mutate(manifest)
    report, metadata, payload = _self_consistent_pruning_sidecars(
        manifest=manifest,
        scope=scope,
    )

    errors = _pruning_replay_errors(
        scenario_id="prune_stress",
        report=report,
        metadata=metadata,
        payload=payload,
        spec=_pruning_scenario(
            "prune_stress",
            f"magnitude_prune:0.5:{scope}",
        ),
    )

    assert any(expected_error in error for error in errors)


def test_generic_pack_dispatch_rejects_a_self_consistent_vision_pruning_claim(
    tmp_path: Path,
) -> None:
    """Exercise the real generic-pack dispatcher, not only its helper."""

    pack = tmp_path / "pack"
    report_dir = pack / "reports" / "model" / "prune_stress" / "run_1"
    report_dir.mkdir(parents=True)
    (pack / "metadata").mkdir(parents=True)
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_pruning_scenario("prune_stress", "magnitude_prune:0.5:all")]},
    )
    manifest = _target_manifest(scope="all")
    manifest["targets"][0].update({"name": "model.visual.blocks.0.mlp.up_proj.weight"})
    report, metadata, payload = _self_consistent_pruning_sidecars(
        manifest=manifest,
        scope="all",
    )
    _write_json(report_dir / "evaluation.report.json", report)
    _write_json(report_dir / "edit_metadata.json", metadata)
    _write_json(report_dir / "pruning_replay.json", payload)

    errors = _verify_edit_metadata_consistency(pack)

    assert any("outside the canonical pruning scope" in error for error in errors)


def test_pack_verifier_rejects_retired_v1_selection_fields_in_clean_pruning_replay(
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
    replay = _replay_payload(
        artifact_identity=artifact_identity,
        baseline_identity=baseline_identity,
    )
    _write_json(report_dir / "pruning_replay.json", replay)

    replay["selection_receipt"] = {"legacy": "untrusted"}
    replay["selection_receipt_sha256"] = "sha256:" + "d" * 64
    _write_json(report_dir / "pruning_replay.json", replay)

    errors = _verify_edit_metadata_consistency(pack)
    assert any(
        "clean pruning replay must not carry retired v1 selection fields" in error
        for error in errors
    )


def test_pack_verifier_requires_replay_coverage_for_every_active_pruning_scenario(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    (pack / "metadata").mkdir(parents=True)
    _write_json(
        pack / "metadata" / "scenarios.json",
        {
            "scenarios": [
                _pruning_scenario("prune_a", "magnitude_prune:0.5:ffn"),
                _pruning_scenario("prune_b", "magnitude_prune:0.5:ffn"),
            ]
        },
    )
    artifact_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "a" * 64,
    }
    baseline_identity = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "b" * 64,
    }
    report_a = pack / "reports" / "model" / "prune_a" / "run_1"
    report_a.mkdir(parents=True)
    _write_json(
        report_a / "evaluation.report.json",
        {
            "meta": {"model_identity": artifact_identity},
            "baseline_ref": {"model_identity": baseline_identity},
        },
    )
    _write_json(report_a / "edit_metadata.json", _metadata())
    _write_json(
        report_a / "pruning_replay.json",
        _replay_payload(
            artifact_identity=artifact_identity,
            baseline_identity=baseline_identity,
        ),
    )

    errors = _verify_edit_metadata_consistency(pack)
    assert any(
        "prune_b: active magnitude-prune scenario has no evaluation report" in error
        for error in errors
    )

    report_b = pack / "reports" / "model" / "prune_b" / "run_1"
    report_b.mkdir(parents=True)
    _write_json(
        report_b / "evaluation.report.json",
        {
            "meta": {"model_identity": artifact_identity},
            "baseline_ref": {"model_identity": baseline_identity},
        },
    )
    _write_json(report_b / "edit_metadata.json", _metadata())

    errors = _verify_edit_metadata_consistency(pack)
    assert any(
        "prune_b: active magnitude-prune scenario has no pruning replay coverage"
        in error
        for error in errors
    )


def test_pack_verifier_rejects_repeated_pruning_run_identity_and_manifest_drift(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "pack"
    (pack / "metadata").mkdir(parents=True)
    _write_json(
        pack / "metadata" / "scenarios.json",
        {"scenarios": [_pruning_scenario("prune_stress", "magnitude_prune:0.5:ffn")]},
    )
    first_artifact = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "a" * 64,
    }
    first_baseline = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "b" * 64,
    }
    second_artifact = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "d" * 64,
    }
    second_baseline = {
        "kind": "local_checkpoint_tree",
        "sha256": "sha256:" + "e" * 64,
    }
    first_dir = pack / "reports" / "model" / "prune_stress" / "run_1"
    second_dir = pack / "reports" / "model" / "prune_stress" / "run_2"
    third_dir = pack / "reports" / "model" / "prune_stress" / "run_3"
    no_model_dir = pack / "reports" / "model" / "errors" / "prune_stress"
    for report_dir, artifact_identity, baseline_identity in (
        (first_dir, first_artifact, first_baseline),
        (second_dir, second_artifact, second_baseline),
        (third_dir, first_artifact, first_baseline),
        (no_model_dir, first_artifact, first_baseline),
    ):
        report_dir.mkdir(parents=True)
        _write_json(
            report_dir / "evaluation.report.json",
            {
                "meta": {"model_identity": artifact_identity},
                "baseline_ref": {"model_identity": baseline_identity},
            },
        )

    first_manifest = _target_manifest()
    second_manifest = _target_manifest()
    targets = second_manifest["targets"]
    assert isinstance(targets, list)
    assert isinstance(targets[0], dict)
    targets[0]["dtype"] = "torch.bfloat16"
    _write_json(
        first_dir / "edit_metadata.json", _metadata(target_manifest=first_manifest)
    )
    _write_json(
        second_dir / "edit_metadata.json", _metadata(target_manifest=second_manifest)
    )
    for report_dir in (third_dir, no_model_dir):
        _write_json(
            report_dir / "edit_metadata.json", _metadata(target_manifest=first_manifest)
        )
    _write_json(
        first_dir / "pruning_replay.json",
        _replay_payload(
            artifact_identity=first_artifact,
            baseline_identity=first_baseline,
            target_manifest=first_manifest,
        ),
    )
    _write_json(
        second_dir / "pruning_replay.json",
        _replay_payload(
            artifact_identity=second_artifact,
            baseline_identity=second_baseline,
            target_manifest=second_manifest,
        ),
    )
    for report_dir in (third_dir, no_model_dir):
        _write_json(
            report_dir / "pruning_replay.json",
            _replay_payload(
                artifact_identity=first_artifact,
                baseline_identity=first_baseline,
                target_manifest=first_manifest,
            ),
        )

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "repeated pruning runs disagree on artifact identity" in error
        for error in errors
    )
    assert any(
        "repeated pruning runs disagree on baseline identity" in error
        for error in errors
    )
    assert any(
        "repeated pruning runs disagree on target manifest digest" in error
        for error in errors
    )
    assert any("pruning report has no model path" in error for error in errors)


def test_pack_verifier_rejects_boolean_pruning_coverage_counter(tmp_path: Path) -> None:
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
    replay["selected_params"] = True
    _write_json(report_dir / "pruning_replay.json", replay)

    errors = _verify_edit_metadata_consistency(pack)

    assert any(
        "selected_params must be a non-negative int" in error for error in errors
    )
    assert any(
        "selected_params does not match target manifest" in error for error in errors
    )


def test_pruning_replay_rejects_non_numeric_sparsity_without_crashing() -> None:
    errors = _pruning_replay_errors(
        scenario_id="prune_clean",
        report={"meta": {}, "baseline_ref": {}},
        metadata={
            "scope": "ffn",
            "parameters": {"target_sparsity": 0.5},
            "coverage": {},
        },
        payload={
            "schema": "invarlock/magnitude-prune-replay-v1",
            "ok": True,
            "edit_type": "magnitude_prune",
            "scope": "ffn",
            "target_sparsity": "half",
            "checked_tensors": 1,
            "selected_tensors": 1,
            "selected_params": 4,
            "expected_pruned_params": 2,
            "expected_changed_params": 2,
            "observed_changed_params": 2,
            "original_zero_params": 0,
            "out_of_scope_tensors_checked": 0,
            "out_of_scope_bytes_checked": 0,
            "support_files_checked": 2,
            "issues": [],
        },
    )

    assert any("target_sparsity missing" in error for error in errors)
