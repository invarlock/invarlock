from __future__ import annotations

import json
from pathlib import Path

from scripts.evidence_workflows.workflow_state import (
    WorkflowLaneResult,
    WorkflowLaneRunState,
    WorkflowPhaseResult,
    WorkflowRunMetadata,
    WorkflowVerificationSummary,
    capture_artifacts,
    collect_artifact_paths,
    write_artifact_manifest,
    write_summary_files,
    write_verification_summary,
)


def test_workflow_lane_run_state_rolls_up_phase_results() -> None:
    state = WorkflowLaneRunState(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        preset="configs/preset.yaml",
        report_path="eval/lane/report/evaluation.report.json",
        verify_path="eval/lane/verify.json",
        phases=(
            WorkflowPhaseResult("materialize_dataset", 0, "ok"),
            WorkflowPhaseResult("evaluate", 0, "ok"),
            WorkflowPhaseResult("verify", 1, "failed", "verify_failed"),
        ),
    )

    result = state.to_lane_result()

    assert result.evaluate_exit == 0
    assert result.verify_exit == 1
    assert result.status == "failed"
    assert result.detail == "verify_failed"
    assert state.to_summary_entry()["phases"][-1]["ok"] is False


def test_workflow_lane_run_state_preserves_skip_as_success_result() -> None:
    state = WorkflowLaneRunState(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        preset="configs/preset.yaml",
        report_path="eval/lane/report/evaluation.report.json",
        verify_path=None,
        phases=(WorkflowPhaseResult("prefetch", 1, "skipped", "gated_repo"),),
    )

    result = state.to_lane_result()

    assert result.ok is True
    assert result.evaluate_exit == 1
    assert result.verify_exit is None
    assert result.status == "skipped"
    assert result.detail == "gated_repo"


def test_workflow_lane_result_marks_skipped_as_ok() -> None:
    result = WorkflowLaneResult(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        preset="configs/preset.yaml",
        evaluate_exit=1,
        verify_exit=None,
        report_path="report/evaluation.report.json",
        verify_path=None,
        status="skipped",
        detail="gated_repo",
    )

    assert result.ok is True
    assert result.to_summary_entry()["ok"] is True


def test_workflow_lane_run_state_handles_empty_and_nonzero_fallback() -> None:
    empty = WorkflowLaneRunState(
        slug="empty",
        lane_id="lane-id",
        model_id="org/model",
        preset="configs/preset.yaml",
        report_path="report.json",
        verify_path=None,
        phases=(),
    )
    assert empty.status == "failed"
    assert empty.to_lane_result().evaluate_exit == 0

    state = WorkflowLaneRunState(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        preset="configs/preset.yaml",
        report_path="report.json",
        verify_path=None,
        phases=(
            WorkflowPhaseResult("prefetch", None, "ok"),
            WorkflowPhaseResult("materialize_dataset", 0, "ok"),
            WorkflowPhaseResult("evaluate", 5, "failed"),
        ),
    )
    assert state.to_lane_result().evaluate_exit == 5

    zero_fallback = WorkflowLaneRunState(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        preset="configs/preset.yaml",
        report_path="report.json",
        verify_path=None,
        phases=(
            WorkflowPhaseResult("prefetch", None, "ok"),
            WorkflowPhaseResult("materialize_dataset", 0, "ok"),
        ),
    )
    assert zero_fallback.to_lane_result().evaluate_exit == 0


def test_workflow_summary_and_artifact_manifest_are_deterministic(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "workflow"
    output_root.mkdir()
    (output_root / "manifest.json").write_text('{"ok": true}\n', encoding="utf-8")
    log_dir = output_root / "logs"
    log_dir.mkdir()
    (log_dir / "lane.log").write_text("log\n", encoding="utf-8")
    result = WorkflowLaneResult(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        preset="configs/preset.yaml",
        evaluate_exit=0,
        verify_exit=0,
        report_path="eval/lane/report/evaluation.report.json",
        verify_path="eval/lane/verify.json",
        status="ok",
    )
    metadata = WorkflowRunMetadata(
        suite="demo",
        execution_mode="host",
        shard_index=1,
        shard_count=2,
    )

    write_summary_files(output_root, metadata=metadata, results=[result])
    write_artifact_manifest(
        output_root,
        schema="demo/schema-v1",
        metadata=metadata,
        results=[result],
        artifact_patterns=[
            "manifest.json",
            "summary.json",
            "summary.tsv",
            "logs/*.log",
        ],
    )

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["suite"] == "demo"
    assert summary["execution_mode"] == "host"
    assert summary["ok"] is True
    assert summary["results"][0]["model_id"] == "org/model"
    assert "lane\tlane-id\tok\t\t0\t0\t" in (output_root / "summary.tsv").read_text(
        encoding="utf-8"
    )

    manifest = json.loads(
        (output_root / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "demo/schema-v1"
    assert manifest["lane_results"][0]["ok"] is True
    files = {entry["path"]: entry for entry in manifest["files"]}
    assert set(files) == {
        "logs/lane.log",
        "manifest.json",
        "summary.json",
        "summary.tsv",
    }
    assert files["manifest.json"]["sha256"]


def test_capture_artifacts_deduplicates_overlapping_patterns(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "summary.json").write_text("{}\n", encoding="utf-8")

    files = capture_artifacts(
        root,
        patterns=["summary.json", "*.json"],
    )

    assert [entry["path"] for entry in files] == ["summary.json"]


def test_collect_artifact_paths_recurses_and_excludes_control_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pack"
    (root / "results").mkdir(parents=True)
    (root / "results" / "final_verdict.json").write_text("{}\n", encoding="utf-8")
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "checksums.sha256").write_text("checksum\n", encoding="utf-8")

    relpaths = collect_artifact_paths(
        root,
        patterns=["**/*"],
        exclude_names={"manifest.json", "checksums.sha256"},
    )

    assert relpaths == ["results/final_verdict.json"]


def test_write_verification_summary_preserves_evidence_pack_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results" / "verification_summary.json"

    write_verification_summary(
        path,
        summary=WorkflowVerificationSummary(
            clean_reports=2,
            error_injection_reports=1,
            expected_failure_reports=3,
            failed_reports=0,
            policy_profile="release",
        ),
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "clean_reports": 2,
        "error_injection_reports": 1,
        "expected_failure_reports": 3,
        "failed_reports": 0,
        "policy_profile": "release",
    }


def test_write_verification_summary_includes_optional_release_review_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results" / "verification_summary.json"

    write_verification_summary(
        path,
        summary=WorkflowVerificationSummary(
            clean_reports=2,
            error_injection_reports=1,
            expected_failure_reports=3,
            failed_reports=0,
            policy_profile="ci",
            report_assurance="strict",
            evaluate_assurance="strict",
            release_review=True,
        ),
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "clean_reports": 2,
        "error_injection_reports": 1,
        "expected_failure_reports": 3,
        "failed_reports": 0,
        "policy_profile": "ci",
        "report_assurance": "strict",
        "evaluate_assurance": "strict",
        "release_review": True,
    }
