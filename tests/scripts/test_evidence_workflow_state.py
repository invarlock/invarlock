from __future__ import annotations

import json
from pathlib import Path

from scripts.evidence_workflows.workflow_state import (
    WorkflowLaneResult,
    WorkflowRunMetadata,
    capture_artifacts,
    write_artifact_manifest,
    write_summary_files,
)


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
        artifact_patterns=["manifest.json", "summary.json", "summary.tsv", "logs/*.log"],
    )

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["suite"] == "demo"
    assert summary["execution_mode"] == "host"
    assert summary["ok"] is True
    assert summary["results"][0]["model_id"] == "org/model"
    assert "lane\tlane-id\tok\t\t0\t0\t" in (
        output_root / "summary.tsv"
    ).read_text(encoding="utf-8")

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
