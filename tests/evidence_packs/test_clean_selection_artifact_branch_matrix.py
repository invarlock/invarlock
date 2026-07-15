from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from invarlock import clean_pruning_selection_artifacts as pruning
from invarlock.clean_pruning_selection_common import (
    CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME,
    CleanPruningSelectionBundleSnapshot,
    CleanPruningSelectionEvidenceError,
)
from invarlock.clean_pruning_selection_contracts import snapshot
from invarlock.clean_selection import artifacts as selection
from invarlock.clean_selection.common import CleanSelectionEvidenceError
from tests.evidence_packs._support_clean_pruning_selection import (
    _bundle,
    _pruning,
    _replay,
    _runtime,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _identity as pruning_identity,
)
from tests.evidence_packs._support_clean_pruning_selection import (
    _record as pruning_record,
)
from tests.evidence_packs._support_clean_selection import _record


def test_pruning_replay_rejects_each_forged_topology_counter_or_binding() -> None:
    baseline = pruning_identity("a")
    artifact = pruning_identity("b")
    spec = _pruning("ffn", 0.5)
    valid = _replay(pruning=spec, baseline=baseline, artifact=artifact)
    cases = [
        lambda replay: replay.pop("schema"),
        lambda replay: replay.__setitem__("ok", False),
        lambda replay: replay.__setitem__("target_sparsity", 0.4),
        lambda replay: replay.__setitem__("model_type", "forged"),
        lambda replay: replay.__setitem__(
            "target_manifest_sha256", "sha256:" + "0" * 64
        ),
        lambda replay: replay.__setitem__("selected_tensors", 2),
        lambda replay: replay.__setitem__("expected_pruned_params", 1),
        lambda replay: replay.__setitem__("observed_changed_params", 1),
        lambda replay: replay.__setitem__("observed_zero_params", 0),
        lambda replay: replay.__setitem__("checked_tensors", 99),
        lambda replay: replay.__setitem__("out_of_scope_bytes_checked", 0),
        lambda replay: replay.__setitem__("support_files_checked", 0),
    ]
    for mutate in cases:
        forged = copy.deepcopy(valid)
        mutate(forged)
        with pytest.raises(CleanPruningSelectionEvidenceError):
            pruning._assert_pruning_replay(
                forged,
                pruning=spec,
                baseline_identity=baseline,
                artifact_identity=artifact,
            )


def test_pruning_runtime_rejects_each_forged_reload_and_storage_claim() -> None:
    artifact = pruning_identity("b")
    valid = _runtime(artifact)

    def bad_diagnostic(runtime: dict[str, object]) -> None:
        diagnostics = runtime["load_diagnostics"]
        assert isinstance(diagnostics, dict)
        reloads = diagnostics["reloads"]
        assert isinstance(reloads, list)
        reloads[0] = {}

    def reported_load_error(runtime: dict[str, object]) -> None:
        diagnostics = runtime["load_diagnostics"]
        assert isinstance(diagnostics, dict)
        reloads = diagnostics["reloads"]
        assert isinstance(reloads, list)
        first = reloads[0]
        assert isinstance(first, dict)
        first["missing_keys"] = ["weight"]

    def bad_audit(runtime: dict[str, object]) -> None:
        envelope = runtime["storage_key_audit"]
        assert isinstance(envelope, dict)
        reloads = envelope["reloads"]
        assert isinstance(reloads, list)
        reloads[0] = {}

    def too_many_storage_keys(runtime: dict[str, object]) -> None:
        envelope = runtime["storage_key_audit"]
        assert isinstance(envelope, dict)
        reloads = envelope["reloads"]
        assert isinstance(reloads, list)
        first = reloads[0]
        assert isinstance(first, dict)
        first["artifact_storage_key_count"] = 99

    def unexpected_storage(runtime: dict[str, object]) -> None:
        envelope = runtime["storage_key_audit"]
        assert isinstance(envelope, dict)
        reloads = envelope["reloads"]
        assert isinstance(reloads, list)
        first = reloads[0]
        assert isinstance(first, dict)
        first["unexpected_storage_keys"] = ["extra"]

    def disagreeing_audits(runtime: dict[str, object]) -> None:
        envelope = runtime["storage_key_audit"]
        assert isinstance(envelope, dict)
        reloads = envelope["reloads"]
        assert isinstance(reloads, list)
        second = reloads[1]
        assert isinstance(second, dict)
        second["model_state_key_count"] = 3

    cases = [
        lambda runtime: runtime.pop("schema"),
        lambda runtime: runtime.__setitem__("repeat_deterministic", False),
        lambda runtime: runtime.__setitem__("token_ids_shape", []),
        lambda runtime: runtime.__setitem__("load_diagnostics", {}),
        lambda runtime: runtime["load_diagnostics"].__setitem__("reloads", []),  # type: ignore[union-attr]
        bad_diagnostic,
        reported_load_error,
        lambda runtime: runtime.__setitem__("storage_key_audit", {}),
        lambda runtime: runtime["storage_key_audit"].__setitem__("reloads", []),  # type: ignore[union-attr]
        bad_audit,
        too_many_storage_keys,
        unexpected_storage,
        disagreeing_audits,
    ]
    for mutate in cases:
        forged = copy.deepcopy(valid)
        mutate(forged)
        with pytest.raises(CleanPruningSelectionEvidenceError):
            pruning._assert_runtime_reload_proof(forged, artifact_identity=artifact)


def _generic_pair(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, str],
    dict[str, str],
]:
    record = _record(tmp_path)
    candidates = record["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)
    transformation = candidate["transformation"]
    evaluation = candidate["evaluation"]
    baseline = record["baseline_identity"]
    assert isinstance(transformation, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(baseline, dict)
    replay_ref = evaluation["replay"]
    runtime_ref = evaluation["runtime"]
    assert isinstance(replay_ref, dict) and isinstance(runtime_ref, dict)
    replay = json.loads(
        (tmp_path / str(replay_ref["path"])).read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (tmp_path / str(runtime_ref["path"])).read_text(encoding="utf-8")
    )
    artifact = replay["artifact_identity"]
    assert isinstance(artifact, dict)
    return replay, runtime, transformation, baseline, artifact


def test_generic_runtime_rejects_each_forged_reload_and_storage_claim(
    tmp_path: Path,
) -> None:
    replay, valid, transformation, baseline, artifact = _generic_pair(tmp_path)

    def bad_load(runtime: dict[str, object]) -> None:
        diagnostics = runtime["load_diagnostics"]
        assert isinstance(diagnostics, dict)
        diagnostics["reloads"] = []

    def reported_load_error(runtime: dict[str, object]) -> None:
        diagnostics = runtime["load_diagnostics"]
        assert isinstance(diagnostics, dict)
        reloads = diagnostics["reloads"]
        assert isinstance(reloads, list)
        first = reloads[0]
        assert isinstance(first, dict)
        first["missing_keys"] = ["weight"]

    def bad_audit(runtime: dict[str, object]) -> None:
        audit = runtime["storage_key_audit"]
        assert isinstance(audit, dict)
        audit["reloads"] = []

    def too_many_storage(runtime: dict[str, object]) -> None:
        audit = runtime["storage_key_audit"]
        assert isinstance(audit, dict)
        reloads = audit["reloads"]
        assert isinstance(reloads, list)
        first = reloads[0]
        assert isinstance(first, dict)
        first["artifact_storage_key_count"] = 99

    def unexpected_storage(runtime: dict[str, object]) -> None:
        audit = runtime["storage_key_audit"]
        assert isinstance(audit, dict)
        reloads = audit["reloads"]
        assert isinstance(reloads, list)
        first = reloads[0]
        assert isinstance(first, dict)
        first["unexpected_storage_keys"] = ["extra"]

    def disagreeing_audits(runtime: dict[str, object]) -> None:
        audit = runtime["storage_key_audit"]
        assert isinstance(audit, dict)
        reloads = audit["reloads"]
        assert isinstance(reloads, list)
        second = reloads[1]
        assert isinstance(second, dict)
        second["model_state_key_count"] = 3

    cases = [
        lambda runtime: runtime.pop("schema"),
        lambda runtime: runtime.__setitem__("ok", False),
        lambda runtime: runtime.__setitem__("token_ids_shape", []),
        lambda runtime: runtime.__setitem__("load_diagnostics", {}),
        bad_load,
        reported_load_error,
        lambda runtime: runtime.__setitem__("storage_key_audit", {}),
        bad_audit,
        too_many_storage,
        unexpected_storage,
        disagreeing_audits,
    ]
    for mutate in cases:
        forged = copy.deepcopy(valid)
        mutate(forged)
        with pytest.raises(CleanSelectionEvidenceError):
            selection._assert_candidate_replay_runtime(
                replay,
                forged,
                transformation=transformation,
                baseline_identity=baseline,
                artifact_identity=artifact,
            )


def test_generic_replay_rejects_missing_or_rebound_target_manifest(
    tmp_path: Path,
) -> None:
    replay, runtime, transformation, baseline, artifact = _generic_pair(tmp_path)
    for index, mutate in enumerate(
        (
            lambda value: value.pop("target_manifest"),
            lambda value: value.__setitem__(
                "target_manifest_sha256", "sha256:" + "0" * 64
            ),
            lambda value: value.__setitem__("scope", "forged"),
            lambda value: value.__setitem__("model_type", "forged"),
        )
    ):
        forged = copy.deepcopy(replay)
        mutate(forged)
        try:
            selection._assert_candidate_replay_runtime(
                forged,
                runtime,
                transformation=transformation,
                baseline_identity=baseline,
                artifact_identity=artifact,
            )
        except CleanSelectionEvidenceError:
            pass
        else:
            raise AssertionError(f"generic replay forgery {index} was accepted")


def _generic_report_context(tmp_path: Path) -> dict[str, object]:
    record = _record(tmp_path)
    candidates = record["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    reports = evaluation["reports"]
    assert isinstance(reports, list)
    report_ref = reports[0]["report"]
    execution_ref = evaluation["execution"]
    assert isinstance(report_ref, dict) and isinstance(execution_ref, dict)
    report = json.loads(
        (tmp_path / str(report_ref["path"])).read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (tmp_path / str(execution_ref["path"])).read_text(encoding="utf-8")
    )
    return {
        "report": report,
        "receipt": receipt,
        "candidate": candidate,
        "evaluation": evaluation,
        "baseline": record["baseline_identity"],
        "config": record["selection_config"],
        "artifact": report_ref["artifact_identity"],
        "execution_sha256": execution_ref["sha256"],
    }


def test_generic_report_eligibility_rejects_each_forged_strict_claim(
    tmp_path: Path,
) -> None:
    context = _generic_report_context(tmp_path)
    valid = context["report"]
    assert isinstance(valid, dict)
    baseline = context["baseline"]
    artifact = context["artifact"]
    assert isinstance(baseline, dict) and isinstance(artifact, dict)

    def set_nested(
        report: dict[str, object], section: str, field: str, value: object
    ) -> None:
        payload = report[section]
        assert isinstance(payload, dict)
        payload[field] = value

    cases = [
        lambda report: set_nested(report, "meta", "model_identity", {}),
        lambda report: set_nested(report, "meta", "model_id", "other/model"),
        lambda report: set_nested(report, "baseline_ref", "model_identity", {}),
        lambda report: set_nested(report, "assurance", "mode", "advisory"),
        lambda report: set_nested(report, "validation", "invariants_pass", False),
        lambda report: set_nested(report, "invariants", "passed", False),
        lambda report: set_nested(report, "primary_metric", "ratio_vs_baseline", 0),
    ]
    for mutate in cases:
        forged = copy.deepcopy(valid)
        mutate(forged)
        with pytest.raises(CleanSelectionEvidenceError):
            selection._eligible_report_quality_loss(
                forged,
                model_key="org/model",
                baseline_identity=baseline,
                artifact_identity=artifact,
            )


def test_generic_execution_receipt_rejects_schema_digest_and_expected_bindings(
    tmp_path: Path,
) -> None:
    context = _generic_report_context(tmp_path)
    receipt = context["receipt"]
    candidate = context["candidate"]
    baseline = context["baseline"]
    config = context["config"]
    assert isinstance(receipt, dict)
    assert isinstance(candidate, dict)
    assert isinstance(baseline, dict)
    assert isinstance(config, dict)
    transformation = candidate["transformation"]
    assert isinstance(transformation, dict)
    assert selection.validate_selection_execution_receipt(receipt) == receipt
    for field, value in (
        ("schema", "retired"),
        ("contract_version", "retired"),
        ("candidate_id", "bad id"),
        ("selection_config_sha256", "sha256:" + "0" * 64),
    ):
        forged = copy.deepcopy(receipt)
        forged[field] = value
        with pytest.raises(CleanSelectionEvidenceError):
            selection.validate_selection_execution_receipt(forged)
    for kwargs in (
        {"expected_model_key": "other/model"},
        {"expected_candidate_id": "other"},
        {"expected_transformation": {**transformation, "scope": "all"}},
        {"expected_baseline_identity": {**baseline, "sha256": "sha256:" + "0" * 64}},
        {"expected_selection_config": {**config, "seed": 999}},
    ):
        with pytest.raises(CleanSelectionEvidenceError):
            selection.validate_selection_execution_receipt(receipt, **kwargs)


def test_generic_evaluator_provenance_rejects_schedule_and_identity_substitution(
    tmp_path: Path,
) -> None:
    context = _generic_report_context(tmp_path)
    report = context["report"]
    receipt = context["receipt"]
    candidate = context["candidate"]
    baseline = context["baseline"]
    config = context["config"]
    assert isinstance(report, dict) and isinstance(receipt, dict)
    assert isinstance(candidate, dict) and isinstance(baseline, dict)
    assert isinstance(config, dict)
    transformation = candidate["transformation"]
    assert isinstance(transformation, dict)
    execution_sha256 = str(context["execution_sha256"])
    with pytest.raises(CleanSelectionEvidenceError, match="repeat_index"):
        selection.build_evaluator_execution_provenance(
            report=report,
            execution_receipt=receipt,
            execution_receipt_sha256=execution_sha256,
            repeat_index=True,
        )
    with pytest.raises(CleanSelectionEvidenceError, match="outside"):
        selection.build_evaluator_execution_provenance(
            report=report,
            execution_receipt=receipt,
            execution_receipt_sha256=execution_sha256,
            repeat_index=99,
        )
    bad_windows = copy.deepcopy(report)
    windows = bad_windows["evaluation_windows"]
    assert isinstance(windows, dict)
    preview = windows["preview"]
    assert isinstance(preview, dict)
    preview["window_ids"] = []
    with pytest.raises(CleanSelectionEvidenceError, match="window_ids"):
        selection.build_evaluator_execution_provenance(
            report=bad_windows,
            execution_receipt=receipt,
            execution_receipt_sha256=execution_sha256,
            repeat_index=0,
        )

    valid_native = report["provenance"]
    assert isinstance(valid_native, dict)
    for mutate in (
        lambda value: value.pop("clean_selection_execution"),
        lambda value: value["clean_selection_execution"].__setitem__(
            "schema", "retired"
        ),  # type: ignore[union-attr]
        lambda value: value["clean_selection_execution"].__setitem__(
            "candidate_id", "wrong"
        ),  # type: ignore[union-attr]
        lambda value: value["clean_selection_execution"].__setitem__("repeat_index", 1),  # type: ignore[union-attr]
        lambda value: value["clean_selection_execution"].__setitem__(
            "report_run_id", "wrong"
        ),  # type: ignore[union-attr]
        lambda value: value["clean_selection_execution"].__setitem__("seed", 999),  # type: ignore[union-attr]
        lambda value: value["clean_selection_execution"].__setitem__(
            "effective_schedule", {}
        ),  # type: ignore[union-attr]
        lambda value: value["clean_selection_execution"].__setitem__(
            "ordered_two_arm_schedule_sha256", "sha256:" + "0" * 64
        ),  # type: ignore[union-attr]
    ):
        forged = copy.deepcopy(report)
        provenance = forged["provenance"]
        assert isinstance(provenance, dict)
        mutate(provenance)
        with pytest.raises(CleanSelectionEvidenceError):
            selection._assert_report_native_execution_provenance(
                forged,
                execution_receipt_sha256=execution_sha256,
                selection_config=config,
                original_model_key="org/model",
                candidate_id=str(candidate["candidate_id"]),
                transformation=transformation,
                baseline_identity=baseline,
                repeat_index=0,
            )


def test_snapshot_inventory_rejects_missing_and_non_directory_roots(
    tmp_path: Path,
) -> None:
    with pytest.raises(CleanPruningSelectionEvidenceError, match="root is missing"):
        snapshot._snapshot_tree_inventory(tmp_path / "missing")

    file_root = tmp_path / "file"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(CleanPruningSelectionEvidenceError, match="regular directory"):
        snapshot._snapshot_tree_inventory(file_root)

    symlink_root = tmp_path / "link"
    symlink_root.symlink_to(file_root)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="regular directory"):
        snapshot._snapshot_tree_inventory(symlink_root)


def test_selected_pruning_entry_requires_a_unique_model_match(tmp_path: Path) -> None:
    record = pruning_record(tmp_path)
    _, bundle = _bundle(tmp_path, record)
    with pytest.raises(CleanPruningSelectionEvidenceError, match="no unique matching"):
        snapshot.selected_clean_pruning_entry_for(bundle, model_key="missing/model")


def test_snapshot_tree_rejects_unexpected_directory_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retained = {
        "candidates/report.json": b"{}",
    }
    staged = CleanPruningSelectionBundleSnapshot(
        bundle={"entries": []},
        bundle_bytes=b"{}",
        sidecar_bytes=retained,
    )
    monkeypatch.setattr(
        snapshot,
        "snapshot_clean_pruning_selection_bundle_file",
        lambda *_args, **_kwargs: staged,
    )
    monkeypatch.setattr(
        snapshot,
        "_snapshot_tree_inventory",
        lambda _root: (
            {
                CLEAN_PRUNING_SELECTION_SNAPSHOT_BUNDLE_FILENAME,
                "candidates/report.json",
            },
            {"candidates", "unexpected"},
        ),
    )
    with pytest.raises(CleanPruningSelectionEvidenceError, match="directory inventory"):
        snapshot.verify_clean_pruning_selection_snapshot_tree(tmp_path)


def test_candidate_snapshot_rejects_reused_sidecar_path(tmp_path: Path) -> None:
    record = pruning_record(tmp_path)
    _, bundle = _bundle(tmp_path, record)
    entries = bundle["entries"]
    assert isinstance(entries, list)
    entry = copy.deepcopy(entries[0])
    selected = entry["selected_entry"]
    assert isinstance(selected, dict)
    receipt = selected["selection_receipt"]
    assert isinstance(receipt, dict)
    candidates = receipt["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    execution = evaluation["execution"]
    replay = evaluation["replay"]
    assert isinstance(execution, dict) and isinstance(replay, dict)
    original_replay_path = replay["path"]
    replay["path"] = execution["path"]

    with pytest.raises(CleanPruningSelectionEvidenceError, match="reuse one sidecar"):
        snapshot._verify_candidate_artifacts(
            entry,
            evidence_root=tmp_path,
            globally_referenced_paths=set(),
        )

    replay["path"] = original_replay_path
    execution_path = str(execution["path"])
    # A path unique within one candidate still cannot be shared by another.
    with pytest.raises(CleanPruningSelectionEvidenceError, match="must not reuse"):
        snapshot._verify_candidate_artifacts(
            entry,
            evidence_root=tmp_path,
            globally_referenced_paths={execution_path},
        )


def _pruning_report_context(tmp_path: Path) -> dict[str, object]:
    record = pruning_record(tmp_path)
    candidates = record["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    reports = evaluation["reports"]
    assert isinstance(reports, list)
    report_ref = reports[0]["report"]
    execution_ref = evaluation["execution"]
    assert isinstance(report_ref, dict) and isinstance(execution_ref, dict)
    report = json.loads(
        (tmp_path / str(report_ref["path"])).read_text(encoding="utf-8")
    )
    return {
        "report": report,
        "candidate": candidate,
        "baseline": record["baseline_identity"],
        "config": record["selection_config"],
        "artifact": report_ref["artifact_identity"],
        "execution_sha256": execution_ref["sha256"],
    }


def test_pruning_report_eligibility_and_provenance_reject_forged_claims(
    tmp_path: Path,
) -> None:
    context = _pruning_report_context(tmp_path)
    valid = context["report"]
    candidate = context["candidate"]
    baseline = context["baseline"]
    config = context["config"]
    artifact = context["artifact"]
    assert isinstance(valid, dict) and isinstance(candidate, dict)
    assert isinstance(baseline, dict) and isinstance(config, dict)
    assert isinstance(artifact, dict)
    pruning_spec = candidate["pruning"]
    assert isinstance(pruning_spec, dict)

    def set_nested(
        report: dict[str, object], section: str, field: str, value: object
    ) -> None:
        payload = report[section]
        assert isinstance(payload, dict)
        payload[field] = value

    for mutate in (
        lambda report: set_nested(report, "meta", "model_id", "other/model"),
        lambda report: set_nested(report, "baseline_ref", "model_identity", {}),
        lambda report: set_nested(report, "assurance", "mode", "advisory"),
        lambda report: set_nested(report, "validation", "invariants_pass", False),
        lambda report: set_nested(report, "invariants", "passed", False),
        lambda report: set_nested(report, "primary_metric", "ratio_vs_baseline", 0),
    ):
        forged = copy.deepcopy(valid)
        mutate(forged)
        with pytest.raises(CleanPruningSelectionEvidenceError):
            pruning._eligible_report_quality_loss(
                forged,
                original_model_key="org/model",
                baseline_identity=baseline,
                artifact_identity=artifact,
            )

    for mutate in (
        lambda report: report.pop("provenance"),
        lambda report: report["provenance"][
            "clean_pruning_selection_execution"
        ].__setitem__("schema", "retired"),  # type: ignore[index,union-attr]
        lambda report: report["provenance"][
            "clean_pruning_selection_execution"
        ].__setitem__("candidate_id", "wrong"),  # type: ignore[index,union-attr]
        lambda report: report["provenance"][
            "clean_pruning_selection_execution"
        ].__setitem__("repeat_index", 1),  # type: ignore[index,union-attr]
        lambda report: report["provenance"][
            "clean_pruning_selection_execution"
        ].__setitem__("report_run_id", "wrong"),  # type: ignore[index,union-attr]
        lambda report: report["provenance"][
            "clean_pruning_selection_execution"
        ].__setitem__("dataset", {}),  # type: ignore[index,union-attr]
        lambda report: report["provenance"][
            "clean_pruning_selection_execution"
        ].__setitem__("effective_schedule", {}),  # type: ignore[index,union-attr]
        lambda report: report["provenance"][
            "clean_pruning_selection_execution"
        ].__setitem__("ordered_two_arm_schedule_sha256", "sha256:" + "0" * 64),  # type: ignore[index,union-attr]
    ):
        forged = copy.deepcopy(valid)
        mutate(forged)
        with pytest.raises(CleanPruningSelectionEvidenceError):
            pruning._assert_report_native_execution_provenance(
                forged,
                execution_receipt_sha256=str(context["execution_sha256"]),
                selection_config=config,
                original_model_key="org/model",
                candidate_id=str(candidate["candidate_id"]),
                pruning=pruning_spec,
                baseline_identity=baseline,
                repeat_index=0,
            )
