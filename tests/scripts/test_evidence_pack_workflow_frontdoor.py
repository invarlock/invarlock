from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_workflow_frontdoor():
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "scripts" / "evidence_packs" / "python" / "workflow_frontdoor.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evidence_pack_workflow_frontdoor", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evidence_pack_workflow_frontdoor"] = module
    spec.loader.exec_module(module)
    return module


def test_evidence_pack_workflow_frontdoor_dry_run_uses_typed_plan(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "scripts" / "evidence_packs" / "python" / "workflow_frontdoor.py"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "run-pack",
            "--dry-run",
            "--output-root",
            str(tmp_path / "workflow"),
            "--",
            "--suite",
            "subset",
            "--out",
            str(tmp_path / "out"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["output_root"] == str(tmp_path / "workflow")
    lane = payload["workflow"][0]
    assert lane["slug"] == "run-pack"
    assert lane["preset"] == "scripts/evidence_packs/scenarios.json"
    assert lane["steps"][0]["name"] == "run_pack"
    assert lane["steps"][0]["command"][-4:] == [
        "--suite",
        "subset",
        "--out",
        str(tmp_path / "out"),
    ]


def test_evidence_pack_workflow_frontdoor_executes_via_shared_runner(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_workflow_frontdoor()
    from evidence_workflows import workflow_runner

    calls: list[tuple[list[str], str | None, str | None]] = []

    def fake_run(cmd, **kwargs):
        env = kwargs.get("env") or {}
        calls.append(
            (
                list(cmd),
                env.get("PACK_WORKFLOW_SUBPROCESS"),
                env.get("PACK_USE_WORKFLOW_FRONTDOOR"),
            )
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(workflow_runner.subprocess, "run", fake_run)

    rc = mod.run_evidence_pack_workflow(
        mod.EvidencePackWorkflowRequest(
            frontdoor="run-suite",
            args=("--suite", "subset"),
            output_root=tmp_path / "workflow",
        )
    )

    assert rc == 0
    assert calls
    assert calls[0][0][:2] == ["bash", str(mod.FRONTDOOR_SCRIPTS["run-suite"])]
    assert calls[0][1] == "1"
    assert calls[0][2] == "0"
    summary = json.loads((tmp_path / "workflow" / "summary.json").read_text())
    assert summary["ok"] is True
    assert summary["results"][0]["slug"] == "run-suite"
    assert (tmp_path / "workflow" / "artifact_manifest.json").is_file()


def test_evidence_pack_workflow_frontdoor_emits_failure_diagnostics(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    mod = _load_workflow_frontdoor()
    from evidence_workflows import WorkflowLaneResult

    output_root = tmp_path / "workflow"
    (output_root / "logs").mkdir(parents=True)
    (output_root / "status.log").write_text("start\nfailed\n", encoding="utf-8")
    (output_root / "logs" / "lane-a.log").write_text(
        "\n".join(f"line {idx}" for idx in range(90)),
        encoding="utf-8",
    )

    class FakeWorkflow:
        def __init__(self) -> None:
            self.output_root = output_root
            self.execution_root = tmp_path
            self.metadata = mod.WorkflowRunMetadata(
                suite="subset",
                execution_mode="test",
            )

        def to_plan(self):
            return object()

    failed_result = WorkflowLaneResult(
        slug="lane-a",
        lane_id="lane-a",
        model_id="org/model",
        preset="preset.yaml",
        evaluate_exit=2,
        verify_exit=None,
        report_path="report.json",
        verify_path=None,
        status="failed",
        detail="verify failed",
    )

    monkeypatch.setattr(
        mod,
        "build_evidence_pack_workflow",
        lambda request: FakeWorkflow(),
    )
    monkeypatch.setattr(
        mod,
        "execute_workflow_sweep",
        lambda request: [failed_result],
    )
    monkeypatch.setattr(mod, "write_summary_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "write_artifact_manifest", lambda *args, **kwargs: None)

    rc = mod.run_evidence_pack_workflow(
        mod.EvidencePackWorkflowRequest(
            frontdoor="run-pack",
            args=("--suite", "subset"),
            output_root=output_root,
        )
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert f"Evidence-pack workflow failed; output_root={output_root}" in captured.err
    assert "Status log:" in captured.err
    assert "failed" in captured.err
    assert "Failed lane lane-a: status=failed detail=verify failed" in captured.err
    assert "evaluate_exit=2 verify_exit=NA" in captured.err
    assert "line 89" in captured.err
    assert "line 0" not in captured.err


def test_evidence_pack_workflow_frontdoor_diagnostic_edge_cases(
    tmp_path: Path, capsys
) -> None:
    mod = _load_workflow_frontdoor()
    from evidence_workflows import WorkflowLaneResult

    assert mod._tail_text(tmp_path / "missing.log") == ""

    ok_result = WorkflowLaneResult(
        slug="lane-ok",
        lane_id="lane-ok",
        model_id="org/model",
        preset="preset.yaml",
        evaluate_exit=0,
        verify_exit=0,
        report_path="report.json",
        verify_path="verify.json",
        status="ok",
    )
    mod._emit_failure_diagnostics(output_root=tmp_path, results=[ok_result])
    assert capsys.readouterr().err == ""

    failed_result = WorkflowLaneResult(
        slug="lane-missing-log",
        lane_id="lane-missing-log",
        model_id="org/model",
        preset="preset.yaml",
        evaluate_exit=4,
        verify_exit=5,
        report_path="report.json",
        verify_path="verify.json",
        status="failed",
    )

    no_status_root = tmp_path / "no-status"
    no_status_root.mkdir()
    mod._emit_failure_diagnostics(
        output_root=no_status_root,
        results=[failed_result],
    )
    no_status_err = capsys.readouterr().err
    assert "Status log:" not in no_status_err
    assert "Log tail:" not in no_status_err
    assert "detail=-" in no_status_err
    assert "verify_exit=5" in no_status_err

    empty_status_root = tmp_path / "empty-status"
    empty_status_root.mkdir()
    (empty_status_root / "status.log").write_text("", encoding="utf-8")
    mod._emit_failure_diagnostics(
        output_root=empty_status_root,
        results=[failed_result],
    )
    empty_status_err = capsys.readouterr().err
    assert "Status log:" in empty_status_err
    assert "Log tail:" not in empty_status_err

    empty_log_root = tmp_path / "empty-log"
    (empty_log_root / "logs").mkdir(parents=True)
    (empty_log_root / "logs" / "lane-missing-log.log").write_text(
        "",
        encoding="utf-8",
    )
    mod._emit_failure_diagnostics(
        output_root=empty_log_root,
        results=[failed_result],
    )
    empty_log_err = capsys.readouterr().err
    assert "Log tail:" in empty_log_err
    assert "verify_exit=5" in empty_log_err


def test_evidence_pack_workflow_frontdoor_parse_errors_and_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_workflow_frontdoor()

    with pytest.raises(SystemExit, match="Usage: workflow_frontdoor.py"):
        mod._parse_args([])
    with pytest.raises(SystemExit, match="Usage: workflow_frontdoor.py"):
        mod._parse_args(["--help"])
    with pytest.raises(SystemExit, match="Unknown evidence-pack workflow frontdoor"):
        mod._parse_args(["unknown"])
    with pytest.raises(SystemExit, match="--output-root requires a value"):
        mod._parse_args(["run-pack", "--output-root"])

    monkeypatch.setenv("PACK_WORKFLOW_OUTPUT_ROOT", str(tmp_path / "env-root"))
    env_request = mod._parse_args(["mini-pack", "--dry-run", "--", "--quick"])
    assert env_request.frontdoor == "mini-pack"
    assert env_request.args == ("--quick",)
    assert env_request.output_root == tmp_path / "env-root"
    assert env_request.dry_run is True

    monkeypatch.delenv("PACK_WORKFLOW_OUTPUT_ROOT", raising=False)
    default_request = mod._parse_args(["run-suite", "--suite", "subset"])
    assert default_request.frontdoor == "run-suite"
    assert default_request.args == ("--suite", "subset")
    assert default_request.output_root.parent == (
        mod.REPO_ROOT / "tmp" / "evidence_pack_workflows"
    )


def test_evidence_pack_workflow_frontdoor_in_process_dry_run(
    tmp_path: Path, capsys
) -> None:
    mod = _load_workflow_frontdoor()

    rc = mod.run_evidence_pack_workflow(
        mod.EvidencePackWorkflowRequest(
            frontdoor="run-pack",
            args=("--suite", "subset"),
            output_root=tmp_path / "workflow",
            dry_run=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_root"] == str(tmp_path / "workflow")
    assert payload["workflow"][0]["steps"][0]["name"] == "run_pack"


def test_evidence_pack_workflow_frontdoor_main_paths(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_workflow_frontdoor()
    seen: list[tuple[str, tuple[str, ...], Path, bool]] = []

    def fake_run(request):
        seen.append(
            (
                request.frontdoor,
                request.args,
                request.output_root,
                request.dry_run,
            )
        )
        return 7

    monkeypatch.setattr(mod, "run_evidence_pack_workflow", fake_run)

    rc_explicit = mod.main(
        [
            "run-pack",
            "--dry-run",
            "--output-root",
            str(tmp_path / "explicit"),
            "--",
            "--suite",
            "subset",
        ]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflow_frontdoor.py",
            "run-suite",
            "--output-root",
            str(tmp_path / "argv"),
        ],
    )
    rc_argv = mod.main(None)

    assert rc_explicit == 7
    assert rc_argv == 7
    assert seen[0] == (
        "run-pack",
        ("--suite", "subset"),
        tmp_path / "explicit",
        True,
    )
    assert seen[1] == ("run-suite", (), tmp_path / "argv", False)
