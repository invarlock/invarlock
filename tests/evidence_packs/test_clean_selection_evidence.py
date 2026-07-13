from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from invarlock.clean_selection.artifacts import build_selection_execution_receipt
from invarlock.clean_selection.bundle import select_clean_transformation
from invarlock.clean_selection.common import (
    TRANSFORMATION_REPLAY_SCHEMA,
    CleanSelectionEvidenceError,
    canonical_json_sha256,
    raw_file_sha256,
)
from invarlock.clean_selection.snapshot import (
    selected_entry_for,
    verify_selection_bundle_file,
)
from scripts.evidence_packs.python.editing.bind_clean_selection_report import (
    bind_candidate_report,
)
from tests.evidence_packs._support_clean_selection import (
    _bundle,
    _candidate_mapping,
    _identity,
    _manifest_reference,
    _record,
    _refresh_bundle,
    _refresh_report_manifest,
    _report_reference,
    _selection_config,
    _write,
)


def test_v1_bundle_retains_pre_eval_receipts_all_repeats_and_recomputes_winner(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    bundle_path, bundle = _bundle(tmp_path, record)

    assert verify_selection_bundle_file(bundle_path) == bundle
    selected = selected_entry_for(bundle, model_key="org/model", edit_type="quant_rtn")
    entry = selected["selected_entry"]
    assert isinstance(entry, dict)
    assert entry["scope"] == "attn"
    assert entry["parameters"] == {"bits": 8, "group_size": 32}
    with pytest.raises(CleanSelectionEvidenceError, match="requested scope"):
        selected_entry_for(
            bundle,
            model_key="org/model",
            edit_type="quant_rtn",
            requested_scope="ffn",
        )
    with pytest.raises(CleanSelectionEvidenceError, match="no unique matching"):
        selected_entry_for(
            bundle,
            model_key="other/model",
            edit_type="quant_rtn",
        )


@pytest.mark.parametrize(
    ("edit_type", "parameters", "error"),
    (
        (
            "synthetic_lowrank_delta",
            {"rank": 33, "scale": 8.0},
            "rank must be at most 32",
        ),
        (
            "synthetic_dense_update",
            {"step_size": 0.001, "iterations": 17},
            "iterations must be at most 16",
        ),
    ),
)
def test_v1_selection_execution_rejects_uncapped_expensive_transformations(
    edit_type: str,
    parameters: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(CleanSelectionEvidenceError, match=error):
        build_selection_execution_receipt(
            original_model_key="org/model",
            candidate_id="candidate",
            transformation={
                "edit_type": edit_type,
                "parameters": parameters,
                "scope": "ffn",
            },
            baseline_identity=_identity("c"),
            selection_config=_selection_config(),
        )


@pytest.mark.parametrize(
    ("edit_type", "parameters"),
    (
        ("synthetic_lowrank_delta", {"rank": 32, "scale": 8.0}),
        ("synthetic_dense_update", {"step_size": 0.001, "iterations": 16}),
    ),
)
def test_v1_selection_execution_accepts_the_canonical_transform_caps(
    edit_type: str, parameters: dict[str, object]
) -> None:
    receipt = build_selection_execution_receipt(
        original_model_key="org/model",
        candidate_id="candidate",
        transformation={
            "edit_type": edit_type,
            "parameters": parameters,
            "scope": "ffn",
        },
        baseline_identity=_identity("c"),
        selection_config=_selection_config(),
    )

    transformation = receipt["transformation"]
    assert isinstance(transformation, dict)
    assert transformation["parameters"] == parameters


@pytest.mark.parametrize(
    ("field", "retired_value", "error"),
    (
        (
            "schema",
            "invarlock/clean-transformation-candidate-record-v2",
            "candidate record has an unrecognized schema",
        ),
        (
            "contract_version",
            "clean-transformation-selection-v2",
            "candidate record has an unrecognized contract version",
        ),
    ),
)
def test_v1_selection_rejects_retired_v2_candidate_identifiers(
    tmp_path: Path, field: str, retired_value: str, error: str
) -> None:
    record = _record(tmp_path)
    record[field] = retired_value

    with pytest.raises(CleanSelectionEvidenceError, match=error):
        select_clean_transformation(record)


@pytest.mark.parametrize(
    ("retired_schema", "error"),
    (
        (
            "invarlock/clean-transformation-selected-entry-v2",
            "selected entry has an unrecognized schema",
        ),
        (
            "invarlock/clean-transformation-selection-receipt-v2",
            "selection receipt has an unrecognized schema",
        ),
        (
            "invarlock/clean-transformation-candidate-evaluation-v2",
            "candidate.evaluation has an unrecognized schema",
        ),
    ),
)
def test_v1_bundle_rejects_retired_v2_nested_schemas(
    tmp_path: Path, retired_schema: str, error: str
) -> None:
    record = _record(tmp_path)
    bundle_path, bundle = _bundle(tmp_path, record)
    entries = bundle["entries"]
    assert isinstance(entries, list) and len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, dict)
    selected = entry["selected_entry"]
    assert isinstance(selected, dict)
    receipt = selected["selection_receipt"]
    assert isinstance(receipt, dict)

    if retired_schema.endswith("selected-entry-v2"):
        entry["schema"] = retired_schema
    elif retired_schema.endswith("selection-receipt-v2"):
        receipt["schema"] = retired_schema
    else:
        candidates = receipt["candidates"]
        assert isinstance(candidates, list) and len(candidates) == 2
        candidate = candidates[0]
        assert isinstance(candidate, dict)
        evaluation = candidate["evaluation"]
        assert isinstance(evaluation, dict)
        evaluation["schema"] = retired_schema
    _write(bundle_path, bundle)

    with pytest.raises(CleanSelectionEvidenceError, match=error):
        verify_selection_bundle_file(bundle_path)


def test_v1_bundle_rejects_sparse_pre_repair_replay_even_with_v1_identifiers(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    replay_reference = evaluation["replay"]
    assert isinstance(replay_reference, dict)
    replay_path = tmp_path / str(replay_reference["path"])
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["schema"] == TRANSFORMATION_REPLAY_SCHEMA
    replay.pop("target_manifest")
    replay.pop("target_manifest_sha256")
    _write(replay_path, replay)
    replay_reference["sha256"] = raw_file_sha256(replay_path)
    bundle_path, _ = _refresh_bundle(tmp_path, record)

    with pytest.raises(
        CleanSelectionEvidenceError, match="candidate replay target_manifest is missing"
    ):
        verify_selection_bundle_file(bundle_path)


def test_bundle_rejects_fully_crosslinked_visual_target_replay(tmp_path: Path) -> None:
    record = _record(tmp_path)
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    replay_reference = evaluation["replay"]
    assert isinstance(replay_reference, dict)
    replay_path = tmp_path / str(replay_reference["path"])
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    target_manifest = replay["target_manifest"]
    assert isinstance(target_manifest, dict)
    targets = target_manifest["targets"]
    assert isinstance(targets, list) and len(targets) == 1
    target = targets[0]
    assert isinstance(target, dict)
    target["name"] = "model.visual.layers.0.self_attn.q_proj.weight"
    replay["target_manifest_sha256"] = canonical_json_sha256(target_manifest)
    _write(replay_path, replay)
    replay_reference["sha256"] = raw_file_sha256(replay_path)
    bundle_path, _ = _refresh_bundle(tmp_path, record)

    with pytest.raises(
        CleanSelectionEvidenceError,
        match="target_manifest is invalid: .*outside the independent transformation scope",
    ):
        verify_selection_bundle_file(bundle_path)


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

    with pytest.raises(CleanSelectionEvidenceError, match="at least two candidates"):
        verify_selection_bundle_file(bundle_path)


def test_bundle_rejects_missing_sidecar_digest_drift_and_identity_drift(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    bundle_path, _ = _bundle(tmp_path, record)
    candidate = _candidate_mapping(record)
    manifest = tmp_path / str(_manifest_reference(candidate, 0)["path"])
    manifest.unlink()
    with pytest.raises(
        CleanSelectionEvidenceError,
        match="runtime manifest repeat 0 sidecar is missing",
    ):
        verify_selection_bundle_file(bundle_path)

    root = tmp_path / "digest"
    record = _record(root)
    bundle_path, _ = _bundle(root, record)
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    replay = root / str(evaluation["replay"]["path"])
    replay.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        CleanSelectionEvidenceError, match="replay sidecar digest mismatch"
    ):
        verify_selection_bundle_file(bundle_path)

    root = tmp_path / "identity"
    record = _record(root)
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    runtime = root / str(evaluation["runtime"]["path"])
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    payload["artifact_identity"] = _identity("f")
    _write(runtime, payload)
    evaluation["runtime"]["sha256"] = raw_file_sha256(runtime)
    bundle_path, _ = _refresh_bundle(root, record)
    with pytest.raises(CleanSelectionEvidenceError, match="runtime sidecar"):
        verify_selection_bundle_file(bundle_path)

    root = tmp_path / "storage-audit"
    record = _record(root)
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    runtime = root / str(evaluation["runtime"]["path"])
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    del payload["storage_key_audit"]
    _write(runtime, payload)
    evaluation["runtime"]["sha256"] = raw_file_sha256(runtime)
    bundle_path, _ = _refresh_bundle(root, record)
    with pytest.raises(CleanSelectionEvidenceError, match="unbound or missing"):
        verify_selection_bundle_file(bundle_path)


def test_bundle_rejects_impossible_runtime_storage_key_counts(tmp_path: Path) -> None:
    record = _record(tmp_path)
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    runtime = tmp_path / str(evaluation["runtime"]["path"])
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    storage_key_audit = payload["storage_key_audit"]
    assert isinstance(storage_key_audit, dict)
    reloads = storage_key_audit["reloads"]
    assert isinstance(reloads, list)
    for audit in reloads:
        assert isinstance(audit, dict)
        audit["artifact_storage_key_count"] = 3
        audit["model_state_key_count"] = 2
    _write(runtime, payload)
    evaluation["runtime"]["sha256"] = raw_file_sha256(runtime)
    bundle_path, _ = _refresh_bundle(tmp_path, record)

    with pytest.raises(
        CleanSelectionEvidenceError,
        match="more artifact storage keys than model state keys",
    ):
        verify_selection_bundle_file(bundle_path)


def test_bundle_rejects_static_claims_forged_strict_report_and_ineligible_guard(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    bundle_path, bundle = _bundle(tmp_path, record)
    bundle["selected_by_operator"] = "selected_by_operator"
    _write(bundle_path, bundle)
    with pytest.raises(CleanSelectionEvidenceError, match="bare selected_by claim"):
        verify_selection_bundle_file(bundle_path)

    root = tmp_path / "forged"
    record = _record(root)
    candidate = _candidate_mapping(record)
    report_ref = _report_reference(candidate, 0)
    report_path = root / str(report_ref["path"])
    forged = {
        "meta": {"model_id": "org/model", "model_identity": _identity("d")},
        "baseline_ref": {"model_identity": _identity("c")},
        "assurance": {
            "mode": "strict",
            "report_local_verdict": "pass",
            "canonical_guard_chain_enforced": True,
            "fallback_fields_used": False,
            "blocking_reasons": [],
        },
        "validation": dict.fromkeys(
            (
                "invariants_pass",
                "spectral_stable",
                "rmt_stable",
                "preview_final_drift_acceptable",
                "primary_metric_acceptable",
                "primary_metric_tail_acceptable",
                "guard_metric_impact_acceptable",
                "guard_warning_policy_acceptable",
            ),
            True,
        ),
        "invariants": {"passed": True, "supported": True},
        "primary_metric": {"ratio_vs_baseline": 1.01},
        "clean_selection": json.loads(report_path.read_text())["clean_selection"],
    }
    _write(report_path, forged)
    _refresh_report_manifest(root, candidate, 0)
    bundle_path, _ = _refresh_bundle(root, record)
    with pytest.raises(
        CleanSelectionEvidenceError, match="candidate report.provenance"
    ):
        verify_selection_bundle_file(bundle_path)

    root = tmp_path / "ineligible"
    record = _record(root)
    candidate = _candidate_mapping(record)
    report_ref = _report_reference(candidate, 0)
    report_path = root / str(report_ref["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["validation"].pop("guard_metric_impact_acceptable")
    _write(report_path, report)
    _refresh_report_manifest(root, candidate, 0)
    bundle_path, _ = _refresh_bundle(root, record)
    with pytest.raises(CleanSelectionEvidenceError, match="ineligible guard result"):
        verify_selection_bundle_file(bundle_path)


def test_binder_requires_pre_eval_provenance_and_rejects_relabelled_config(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path, bind_reports=False)
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    report_path = tmp_path / str(_report_reference(candidate, 0)["path"])
    replay_path = tmp_path / str(evaluation["replay"]["path"])
    runtime_path = tmp_path / str(evaluation["runtime"]["path"])
    execution_path = tmp_path / str(evaluation["execution"]["path"])
    config_path = tmp_path / "selection_config.json"
    _write(config_path, record["selection_config"])

    binding = bind_candidate_report(
        report_path=report_path,
        replay_path=replay_path,
        runtime_path=runtime_path,
        selection_config_path=config_path,
        execution_receipt_path=execution_path,
        model_key="org/model",
        candidate_id="attn8",
        repeat_index=0,
    )
    assert binding["quality_loss"] == pytest.approx(0.01)
    assert (
        bind_candidate_report(
            report_path=report_path,
            replay_path=replay_path,
            runtime_path=runtime_path,
            selection_config_path=config_path,
            execution_receipt_path=execution_path,
            model_key="org/model",
            candidate_id="attn8",
            repeat_index=0,
        )
        == binding
    )
    relabelled = copy.deepcopy(record["selection_config"])
    assert isinstance(relabelled, dict)
    relabelled["seed"] = 18
    relabelled_path = tmp_path / "relabelled-selection-config.json"
    _write(relabelled_path, relabelled)
    with pytest.raises(CleanSelectionEvidenceError, match="selection config mismatch"):
        bind_candidate_report(
            report_path=report_path,
            replay_path=replay_path,
            runtime_path=runtime_path,
            selection_config_path=relabelled_path,
            execution_receipt_path=execution_path,
            model_key="org/model",
            candidate_id="attn8",
            repeat_index=0,
        )


@pytest.mark.parametrize(
    "field",
    (
        "dataset.content_sha256",
        "seed",
        "schedule.evaluation_repeats",
        "schedule.max_examples",
        "schedule.batch_size",
        "schedule.shuffle",
    ),
)
def test_binder_rejects_every_selection_config_delta(
    tmp_path: Path, field: str
) -> None:
    record = _record(tmp_path)
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    changed = copy.deepcopy(record["selection_config"])
    assert isinstance(changed, dict)
    dataset = changed["dataset"]
    schedule = changed["schedule"]
    assert isinstance(dataset, dict)
    assert isinstance(schedule, dict)
    if field == "dataset.content_sha256":
        dataset["content_sha256"] = "sha256:" + "f" * 64
    elif field == "seed":
        changed["seed"] = 18
    elif field == "schedule.evaluation_repeats":
        schedule["evaluation_repeats"] = 3
    elif field == "schedule.max_examples":
        schedule["max_examples"] = 3
    elif field == "schedule.batch_size":
        schedule["batch_size"] = 2
    else:
        schedule["shuffle"] = True
    changed_path = tmp_path / "changed-selection-config.json"
    _write(changed_path, changed)

    with pytest.raises(CleanSelectionEvidenceError, match="selection config mismatch"):
        bind_candidate_report(
            report_path=tmp_path / str(_report_reference(candidate, 0)["path"]),
            replay_path=tmp_path / str(evaluation["replay"]["path"]),
            runtime_path=tmp_path / str(evaluation["runtime"]["path"]),
            selection_config_path=changed_path,
            execution_receipt_path=tmp_path / str(evaluation["execution"]["path"]),
            model_key="org/model",
            candidate_id="attn8",
            repeat_index=0,
        )


def test_bundle_rejects_underretained_repeats_and_post_manifest_mutation(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path / "underretained")
    config = record["selection_config"]
    assert isinstance(config, dict)
    config["schedule"]["evaluation_repeats"] = 3  # type: ignore[index]
    candidate = _candidate_mapping(record)
    evaluation = candidate["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["selection_config_sha256"] = canonical_json_sha256(config)
    with pytest.raises(
        CleanSelectionEvidenceError, match="retain exactly evaluation_repeats"
    ):
        select_clean_transformation(record)

    root = tmp_path / "manifest-mutation"
    record = _record(root)
    candidate = _candidate_mapping(record)
    report_ref = _report_reference(candidate, 0)
    report_path = root / str(report_ref["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["primary_metric"]["final"] = 2.4
    report["primary_metric"]["ratio_vs_baseline"] = 1.2
    report["clean_selection"]["quality_loss"] = 0.2
    _write(report_path, report)
    report_ref["sha256"] = raw_file_sha256(report_path)
    bundle_path, _ = _refresh_bundle(root, record)
    with pytest.raises(
        CleanSelectionEvidenceError,
        match="runtime manifest is not an eligible strict binding",
    ):
        verify_selection_bundle_file(bundle_path)
