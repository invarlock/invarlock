from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from invarlock.clean_pruning_selection_common import (
    CleanPruningSelectionEvidenceError,
    canonical_json_sha256,
    raw_file_sha256,
)
from invarlock.clean_pruning_selection_contract import (
    canonical_clean_pruning_candidate_set_sha256,
    select_clean_pruning,
)
from invarlock.clean_pruning_selection_contracts.snapshot import (
    snapshot_clean_pruning_selection_bundle_file,
    verify_clean_pruning_selection_bundle_file,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _bundle,
    _candidate_mapping,
    _record,
    _refresh_record_and_bundle,
    _refresh_report_and_manifest,
    _write,
)


def test_clean_pruning_bundle_rejects_nonfinite_json(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text('{"bundle_sha256": NaN}\n', encoding="utf-8")

    with pytest.raises(CleanPruningSelectionEvidenceError, match="non-standard"):
        snapshot_clean_pruning_selection_bundle_file(bundle)


def test_rejects_candidate_and_bundle_digest_drift(tmp_path: Path) -> None:
    record = _record(tmp_path)
    record["candidate_set_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(
        CleanPruningSelectionEvidenceError, match="candidate_set_sha256"
    ):
        select_clean_pruning(record)

    root = tmp_path / "bundle"
    record = _record(root)
    bundle_path, bundle = _bundle(root, record)
    bundle["bundle_sha256"] = "sha256:" + "0" * 64
    _write(bundle_path, bundle)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="bundle_sha256"):
        verify_clean_pruning_selection_bundle_file(bundle_path)


def test_rejects_forged_topology_tie_algorithm_and_stale_preset_claims(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    replay_ref = cast(dict[str, object], evaluation["replay"])
    replay_path = tmp_path / cast(str, replay_ref["path"])
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["pruning_algorithm"] = "threshold_without_ties_v1"
    _write(replay_path, replay)
    replay_ref["sha256"] = raw_file_sha256(replay_path)
    bundle_path, _ = _refresh_record_and_bundle(tmp_path, record)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="exact candidate"):
        verify_clean_pruning_selection_bundle_file(bundle_path)

    root = tmp_path / "topology"
    record = _record(root)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    replay_ref = cast(dict[str, object], evaluation["replay"])
    replay_path = root / cast(str, replay_ref["path"])
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["target_manifest"]["targets"][0]["shape"] = [4, 2]
    replay["target_manifest_sha256"] = canonical_json_sha256(replay["target_manifest"])
    _write(replay_path, replay)
    replay_ref["sha256"] = raw_file_sha256(replay_path)
    bundle_path, _ = _refresh_record_and_bundle(root, record)
    with pytest.raises(
        CleanPruningSelectionEvidenceError, match="numel does not match"
    ):
        verify_clean_pruning_selection_bundle_file(bundle_path)

    root = tmp_path / "family"
    record = _record(root)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    replay_ref = cast(dict[str, object], evaluation["replay"])
    replay_path = root / cast(str, replay_ref["path"])
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["architecture"] = "gpt2"
    replay["target_manifest"]["architecture"] = "gpt2"
    replay["target_manifest_sha256"] = canonical_json_sha256(replay["target_manifest"])
    _write(replay_path, replay)
    replay_ref["sha256"] = raw_file_sha256(replay_path)
    bundle_path, _ = _refresh_record_and_bundle(root, record)
    with pytest.raises(
        CleanPruningSelectionEvidenceError, match="model_type and architecture"
    ):
        verify_clean_pruning_selection_bundle_file(bundle_path)

    root = tmp_path / "preset"
    record = _record(root)
    record["selected_by_tuned_preset"] = "selected_by_replayable_contract"
    with pytest.raises(CleanPruningSelectionEvidenceError, match="bare selected_by"):
        canonical_clean_pruning_candidate_set_sha256(record)


def test_rejects_posthoc_execution_binding_and_non_strict_report(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    reports = cast(list[dict[str, object]], evaluation["reports"])
    report_ref = cast(dict[str, object], reports[0]["report"])
    report_path = tmp_path / cast(str, report_ref["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["provenance"]["clean_pruning_selection_execution"]
    _write(report_path, report)
    _refresh_report_and_manifest(tmp_path, candidate, 0)
    bundle_path, _ = _refresh_record_and_bundle(tmp_path, record)
    with pytest.raises(
        CleanPruningSelectionEvidenceError, match="evaluator provenance"
    ):
        verify_clean_pruning_selection_bundle_file(bundle_path)

    root = tmp_path / "strict"
    record = _record(root)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    reports = cast(list[dict[str, object]], evaluation["reports"])
    report_ref = cast(dict[str, object], reports[0]["report"])
    report_path = root / cast(str, report_ref["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["assurance"]["report_local_verdict"] = "warn"
    _write(report_path, report)
    _refresh_report_and_manifest(root, candidate, 0)
    bundle_path, _ = _refresh_record_and_bundle(root, record)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="eligible strict"):
        verify_clean_pruning_selection_bundle_file(bundle_path)


def test_rejects_runtime_proof_copy_sidecar_symlink_and_duplicate_json(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    runtime_ref = cast(dict[str, object], evaluation["runtime"])
    runtime_path = tmp_path / cast(str, runtime_ref["path"])
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["replay_schema"] = "invarlock/generated-transformation-replay-v1"
    _write(runtime_path, runtime)
    runtime_ref["sha256"] = raw_file_sha256(runtime_path)
    bundle_path, _ = _refresh_record_and_bundle(tmp_path, record)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="two-reload"):
        verify_clean_pruning_selection_bundle_file(bundle_path)

    root = tmp_path / "storage-audit"
    record = _record(root)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    runtime_ref = cast(dict[str, object], evaluation["runtime"])
    runtime_path = root / cast(str, runtime_ref["path"])
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    del runtime["storage_key_audit"]
    _write(runtime_path, runtime)
    runtime_ref["sha256"] = raw_file_sha256(runtime_path)
    bundle_path, _ = _refresh_record_and_bundle(root, record)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="unbound or missing"):
        verify_clean_pruning_selection_bundle_file(bundle_path)

    root = tmp_path / "symlink"
    record = _record(root)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    replay_ref = cast(dict[str, object], evaluation["replay"])
    replay_path = root / cast(str, replay_ref["path"])
    copied = replay_path.with_name("copied-replay.json")
    copied.write_bytes(replay_path.read_bytes())
    replay_path.unlink()
    replay_path.symlink_to(copied.name)
    bundle_path, _ = _bundle(root, record)
    with pytest.raises(
        CleanPruningSelectionEvidenceError, match="must not traverse a symlink"
    ):
        verify_clean_pruning_selection_bundle_file(bundle_path)

    root = tmp_path / "duplicate-json"
    record = _record(root)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    replay_ref = cast(dict[str, object], evaluation["replay"])
    replay_path = root / cast(str, replay_ref["path"])
    replay_path.write_text('{"schema":"x","schema":"x"}\n', encoding="utf-8")
    replay_ref["sha256"] = raw_file_sha256(replay_path)
    bundle_path, _ = _refresh_record_and_bundle(root, record)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="duplicate key"):
        verify_clean_pruning_selection_bundle_file(bundle_path)


def test_rejects_impossible_runtime_storage_key_counts(tmp_path: Path) -> None:
    record = _record(tmp_path)
    candidate = _candidate_mapping(record, 0)
    evaluation = cast(dict[str, object], candidate["evaluation"])
    runtime_ref = cast(dict[str, object], evaluation["runtime"])
    runtime_path = tmp_path / cast(str, runtime_ref["path"])
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    storage_key_audit = runtime["storage_key_audit"]
    assert isinstance(storage_key_audit, dict)
    reloads = storage_key_audit["reloads"]
    assert isinstance(reloads, list)
    for audit in reloads:
        assert isinstance(audit, dict)
        audit["artifact_storage_key_count"] = 3
        audit["model_state_key_count"] = 2
    _write(runtime_path, runtime)
    runtime_ref["sha256"] = raw_file_sha256(runtime_path)
    bundle_path, _ = _refresh_record_and_bundle(tmp_path, record)

    with pytest.raises(
        CleanPruningSelectionEvidenceError,
        match="more artifact storage keys than model state keys",
    ):
        verify_clean_pruning_selection_bundle_file(bundle_path)
