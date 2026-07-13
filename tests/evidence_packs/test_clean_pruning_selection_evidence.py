from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import pytest

from invarlock.clean_pruning_selection_artifacts import (
    validate_clean_pruning_candidate_replay_runtime,
)
from invarlock.clean_pruning_selection_common import (
    CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME,
    CleanPruningSelectionEvidenceError,
)
from invarlock.clean_pruning_selection_contract import (
    select_clean_pruning,
)
from invarlock.clean_pruning_selection_contracts.snapshot import (
    referenced_clean_pruning_candidate_paths,
    selected_clean_pruning_artifact_identity_for,
    selected_clean_pruning_entry_for,
    snapshot_clean_pruning_selection_bundle_file,
    verify_clean_pruning_selection_bundle_file,
    verify_clean_pruning_selection_snapshot_tree,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _bind_clean_replay_to_manifest,
    _bundle,
    _candidate_mapping,
    _identity,
    _make_targets_noncanonical,
    _pruning,
    _record,
    _replay,
    _runtime,
    _stage_snapshot,
    _write,
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
def test_clean_pack_rejects_self_consistent_forged_pruning_target_manifests(
    scope: str,
    mutate: object,
    expected_error: str,
) -> None:
    baseline = _identity("a")
    artifact = _identity("b")
    pruning = _pruning(scope, 0.5)
    replay = _replay(pruning=pruning, baseline=baseline, artifact=artifact)
    manifest = replay["target_manifest"]
    assert isinstance(manifest, dict)
    assert callable(mutate)
    mutate(manifest)
    _bind_clean_replay_to_manifest(replay, manifest)

    with pytest.raises(CleanPruningSelectionEvidenceError, match=expected_error):
        validate_clean_pruning_candidate_replay_runtime(
            replay=replay,
            runtime=_runtime(artifact),
            pruning=pruning,
            baseline_identity=baseline,
        )


def test_verifies_all_candidates_recomputes_mean_winner_and_snapshots_bytes(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    bundle_path, bundle = _bundle(tmp_path, record)

    assert verify_clean_pruning_selection_bundle_file(bundle_path) == bundle
    entry = selected_clean_pruning_entry_for(bundle, model_key="org/model")
    selected = cast(dict[str, object], entry["selected_entry"])
    assert selected["scope"] == "ffn"
    assert selected["target_sparsity"] == 0.5
    assert referenced_clean_pruning_candidate_paths(bundle) == sorted(
        referenced_clean_pruning_candidate_paths(bundle)
    )
    assert len(referenced_clean_pruning_candidate_paths(bundle)) == 14
    snapshot = snapshot_clean_pruning_selection_bundle_file(bundle_path)
    assert snapshot.bundle == bundle
    assert len(snapshot.sidecar_bytes) == 14
    assert selected_clean_pruning_artifact_identity_for(
        bundle, model_key="org/model"
    ) == _identity("e")
    with pytest.raises(CleanPruningSelectionEvidenceError, match="requested scope"):
        selected_clean_pruning_entry_for(
            bundle, model_key="org/model", requested_scope="attn"
        )


@pytest.mark.parametrize(
    ("location", "retired_value"),
    (
        ("record_schema", "invarlock/clean-pruning-candidate-record-v2"),
        ("contract", "clean-pruning-selection-v2"),
        (
            "evaluation_schema",
            "invarlock/clean-pruning-candidate-evaluation-v2",
        ),
    ),
)
def test_candidate_record_rejects_retired_pruning_selection_protocols(
    tmp_path: Path, location: str, retired_value: str
) -> None:
    record = _record(tmp_path)
    if location == "record_schema":
        record["schema"] = retired_value
    elif location == "contract":
        record["contract_version"] = retired_value
    else:
        candidate = _candidate_mapping(record, 0)
        evaluation = cast(dict[str, object], candidate["evaluation"])
        evaluation["schema"] = retired_value

    with pytest.raises(CleanPruningSelectionEvidenceError):
        select_clean_pruning(record)


@pytest.mark.parametrize(
    ("location", "retired_value"),
    (
        ("bundle", "invarlock/clean-pruning-selection-bundle-v2"),
        ("selected", "invarlock/clean-pruning-selected-entry-v2"),
        ("receipt", "invarlock/clean-pruning-selection-receipt-v2"),
    ),
)
def test_bundle_rejects_retired_pruning_selection_protocols(
    tmp_path: Path, location: str, retired_value: str
) -> None:
    record = _record(tmp_path)
    bundle_path, bundle = _bundle(tmp_path, record)
    if location == "bundle":
        bundle["schema"] = retired_value
    else:
        entries = cast(list[dict[str, object]], bundle["entries"])
        selected = cast(dict[str, object], entries[0]["selected_entry"])
        if location == "selected":
            selected["schema"] = retired_value
        else:
            receipt = cast(dict[str, object], selected["selection_receipt"])
            receipt["schema"] = retired_value
    _write(bundle_path, bundle)

    with pytest.raises(CleanPruningSelectionEvidenceError):
        verify_clean_pruning_selection_bundle_file(bundle_path)


def test_bundle_rejects_one_candidate_as_not_a_selection(tmp_path: Path) -> None:
    record = _record(tmp_path)
    bundle_path, bundle = _bundle(tmp_path, record)
    entries = bundle["entries"]
    assert isinstance(entries, list)
    entry = entries[0]
    assert isinstance(entry, dict)
    selected = entry["selected_entry"]
    assert isinstance(selected, dict)
    receipt = selected["selection_receipt"]
    assert isinstance(receipt, dict)
    candidates = receipt["candidates"]
    assert isinstance(candidates, list)
    receipt["candidates"] = [copy.deepcopy(candidates[0])]
    receipt["candidate_set_sha256"] = "sha256:" + "0" * 64
    selected["selection_receipt_sha256"] = "sha256:" + "0" * 64
    _write(bundle_path, bundle)

    with pytest.raises(
        CleanPruningSelectionEvidenceError, match="at least two candidates"
    ):
        verify_clean_pruning_selection_bundle_file(bundle_path)


def test_snapshot_bridge_requires_exact_staged_sidecars_and_no_extras(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    bundle_path, bundle = _bundle(tmp_path, record)
    stage = _stage_snapshot(tmp_path, bundle_path, bundle)

    staged = verify_clean_pruning_selection_snapshot_tree(stage)
    assert staged.bundle == bundle
    assert len(staged.sidecar_bytes) == 14

    (stage / "stale-preset.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CleanPruningSelectionEvidenceError, match="file inventory"):
        verify_clean_pruning_selection_snapshot_tree(stage)
    (stage / "stale-preset.json").unlink()
    (stage / "linked.json").symlink_to(CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="symlink"):
        verify_clean_pruning_selection_snapshot_tree(stage)
