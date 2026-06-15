from __future__ import annotations

import os
import sys
from pathlib import Path

from scripts.evidence_workflows.workflow_plan import (
    WorkflowCommandStep,
    WorkflowLanePlan,
    WorkflowSweepPlan,
)
from scripts.evidence_workflows.workflow_runner import (
    run_logged_command_with_retry,
    workflow_return_code,
)
from scripts.evidence_workflows.workflow_state import (
    WorkflowLaneResult,
    WorkflowRunMetadata,
)


def test_workflow_lane_plan_serializes_compat_dry_run_fields(tmp_path: Path) -> None:
    lane_root = tmp_path / "exec" / "eval" / "lane"
    plan = WorkflowLanePlan(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        execution_mode="host",
        lane_root=lane_root,
        published_lane_root=tmp_path / "out" / "eval" / "lane",
        report_path=lane_root / "report" / "evaluation.report.json",
        verify_path=lane_root / "verify.json",
        profile="dev",
        resource_preflight={"ok": True},
        prepared_preset="exec/eval/lane/prepared_preset.yaml",
        steps=(
            WorkflowCommandStep("materialize_dataset", ("python", "materialize.py")),
            WorkflowCommandStep("prefetch", ("python", "-c", "prefetch")),
            WorkflowCommandStep("evaluate", ("python", "-m", "invarlock", "evaluate")),
            WorkflowCommandStep(
                "verify",
                ("python", "-m", "invarlock", "verify"),
                output_path=lane_root / "verify.json",
            ),
        ),
    )

    payload = plan.to_dry_run_entry()

    assert payload["slug"] == "lane"
    assert payload["profile"] == "dev"
    assert payload["evaluate"] == ["python", "-m", "invarlock", "evaluate"]
    assert payload["verify"] == ["python", "-m", "invarlock", "verify"]
    assert payload["prefetch"] == ["python", "-c", "prefetch"]
    assert payload["materialize_dataset"] == ["python", "materialize.py"]
    assert payload["prepared_preset"] == "exec/eval/lane/prepared_preset.yaml"
    assert [step["name"] for step in payload["steps"]] == [
        "materialize_dataset",
        "prefetch",
        "evaluate",
        "verify",
    ]


def test_workflow_sweep_plan_reuses_lane_dry_run_payload(tmp_path: Path) -> None:
    lane_root = tmp_path / "exec" / "eval" / "lane"
    lane = WorkflowLanePlan(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        execution_mode="container",
        lane_root=lane_root,
        published_lane_root=tmp_path / "out" / "eval" / "lane",
        report_path=lane_root / "report" / "evaluation.report.json",
        verify_path=lane_root / "verify.json",
        profile="ci",
        steps=(
            WorkflowCommandStep("evaluate", ("python", "-m", "invarlock", "evaluate")),
            WorkflowCommandStep("verify", ("python", "-m", "invarlock", "verify")),
        ),
    )
    plan = WorkflowSweepPlan(
        metadata=WorkflowRunMetadata("suite", "container"),
        output_root=tmp_path / "out",
        execution_root=tmp_path / "exec",
        lanes=(lane,),
    )

    assert plan.to_dry_run_payload()[0]["slug"] == "lane"
    assert plan.to_dry_run_payload()[0]["execution_mode"] == "container"


def test_workflow_runner_retries_configured_returncode(tmp_path: Path) -> None:
    script = tmp_path / "flaky.py"
    state = tmp_path / "state"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"state = Path({str(state)!r})\n"
        "if not state.exists():\n"
        "    state.write_text('seen')\n"
        "    sys.exit(3)\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "lane.log"

    run = run_logged_command_with_retry(
        name="evaluate",
        command=(sys.executable, str(script)),
        cwd=tmp_path,
        env=os.environ,
        log_path=log_path,
        log_mode="w",
        retry_returncodes=(3,),
        retry_message="{name} exited with {returncode}; retrying once.",
    )

    assert run.ok is True
    assert run.attempts == 2
    assert "evaluate exited with 3; retrying once." in log_path.read_text(
        encoding="utf-8"
    )


def test_workflow_return_code_uses_result_ok_semantics() -> None:
    ok = WorkflowLaneResult(
        slug="ok",
        lane_id="lane",
        model_id="org/model",
        preset="preset.yaml",
        evaluate_exit=0,
        verify_exit=0,
        report_path="report.json",
        verify_path="verify.json",
        status="ok",
    )
    skipped = WorkflowLaneResult(
        slug="skipped",
        lane_id="lane",
        model_id="org/model",
        preset="preset.yaml",
        evaluate_exit=1,
        verify_exit=None,
        report_path="report.json",
        verify_path=None,
        status="skipped",
    )

    assert workflow_return_code([ok, skipped]) == 0
    assert workflow_return_code([]) == 1
