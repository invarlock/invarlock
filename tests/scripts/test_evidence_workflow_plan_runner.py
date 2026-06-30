from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from scripts.evidence_workflows.workflow_plan import (
    WorkflowCommandStep,
    WorkflowLanePlan,
    WorkflowSweepPlan,
)
from scripts.evidence_workflows.workflow_runner import (
    WorkflowLaneExecutionRequest,
    WorkflowSweepExecutionRequest,
    execute_workflow_lane,
    execute_workflow_sweep,
    run_logged_command_with_retry,
    workflow_return_code,
)
from scripts.evidence_workflows.workflow_state import (
    WorkflowLaneResult,
    WorkflowLaneRunState,
    WorkflowRunMetadata,
)


def test_workflow_lane_plan_serializes_compat_dry_run_fields(tmp_path: Path) -> None:
    lane_root = tmp_path / "exec" / "eval" / "lane"
    plan = WorkflowLanePlan(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        execution_mode="host",
        preset="configs/preset.yaml",
        lane_root=lane_root,
        published_lane_root=tmp_path / "out" / "eval" / "lane",
        report_path=lane_root / "report" / "evaluation.report.json",
        verify_path=lane_root / "verify.json",
        profile="dev",
        resource_preflight={"ok": True},
        prepared_preset="exec/eval/lane/prepared_preset.yaml",
        prepared_preset_source="configs/override.yaml",
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
    assert payload["prepared_preset_source"] == "configs/override.yaml"
    assert [step["name"] for step in payload["steps"]] == [
        "materialize_dataset",
        "prefetch",
        "evaluate",
        "verify",
    ]


def test_workflow_plan_optional_fields_and_manifest_payload(tmp_path: Path) -> None:
    lane_root = tmp_path / "exec" / "lane"
    step = WorkflowCommandStep(
        "prefetch",
        ("python", "-c", "print('prefetch')"),
        output_path=lane_root / "prefetch.json",
        retry_returncodes=(3,),
        retry_message="retry {name} after {returncode}",
        requires_report=True,
    )
    plan = WorkflowLanePlan(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        execution_mode="host",
        preset="configs/preset.yaml",
        lane_root=lane_root,
        published_lane_root=tmp_path / "out" / "lane",
        report_path=lane_root / "report.json",
        verify_path=lane_root / "verify.json",
        profile="dev",
        steps=(step,),
    )

    payload = plan.to_dry_run_entry()

    assert payload["steps"][0]["output_path"].endswith("prefetch.json")
    assert payload["steps"][0]["retry_returncodes"] == [3]
    assert payload["steps"][0]["retry_message"] == "retry {name} after {returncode}"
    assert payload["steps"][0]["requires_report"] is True
    assert "evaluate" not in payload
    assert "verify" not in payload
    assert plan.optional_step("missing") is None
    with pytest.raises(KeyError, match="has no step"):
        plan.step("missing")

    eval_step = WorkflowCommandStep("evaluate", ("python", "eval.py"))
    verify_step = WorkflowCommandStep("verify", ("python", "verify.py"))
    lane_with_standard_steps = WorkflowLanePlan(
        slug="standard",
        lane_id="standard-id",
        model_id="org/model",
        execution_mode="host",
        preset="configs/preset.yaml",
        lane_root=tmp_path / "exec" / "standard",
        published_lane_root=tmp_path / "out" / "standard",
        report_path=tmp_path / "exec" / "standard" / "report.json",
        verify_path=tmp_path / "exec" / "standard" / "verify.json",
        profile="dev",
        steps=(eval_step, verify_step),
    )
    assert lane_with_standard_steps.evaluate_step is eval_step
    assert lane_with_standard_steps.verify_step is verify_step

    sweep = WorkflowSweepPlan(
        metadata=WorkflowRunMetadata("suite", "host", shard_index=1, shard_count=3),
        output_root=tmp_path / "out",
        execution_root=tmp_path / "exec",
        lanes=(plan,),
    )
    manifest = sweep.to_manifest_payload(
        lane_entries=[{"slug": "lane", "ok": True}],
        generated_at="2026-06-15T00:00:00Z",
    )
    assert manifest == {
        "suite": "suite",
        "execution_mode": "host",
        "shard_index": 1,
        "shard_count": 3,
        "generated_at": "2026-06-15T00:00:00Z",
        "lanes": [{"slug": "lane", "ok": True}],
    }


def test_workflow_sweep_plan_reuses_lane_dry_run_payload(tmp_path: Path) -> None:
    lane_root = tmp_path / "exec" / "eval" / "lane"
    lane = WorkflowLanePlan(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        execution_mode="container",
        preset="configs/preset.yaml",
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


def test_workflow_runner_returns_without_retry_for_unconfigured_code(
    tmp_path: Path,
) -> None:
    script = tmp_path / "exit_two.py"
    script.write_text("raise SystemExit(2)\n", encoding="utf-8")

    run = run_logged_command_with_retry(
        name="evaluate",
        command=(sys.executable, str(script)),
        cwd=tmp_path,
        env=os.environ,
        log_path=tmp_path / "lane.log",
        retry_returncodes=(3,),
    )

    assert run.returncode == 2
    assert run.attempts == 1
    assert run.to_payload()["returncode"] == 2


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


def test_execute_workflow_lane_owns_step_sequence_and_report_gating(
    tmp_path: Path,
) -> None:
    script = tmp_path / "write_report.py"
    report_path = (
        tmp_path / "exec" / "eval" / "lane" / "report" / "evaluation.report.json"
    )
    script.write_text(
        "from pathlib import Path\n"
        f"path = Path({str(report_path)!r})\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_text('{}')\n",
        encoding="utf-8",
    )
    lane_root = tmp_path / "exec" / "eval" / "lane"
    plan = WorkflowLanePlan(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        execution_mode="host",
        preset="configs/preset.yaml",
        lane_root=lane_root,
        published_lane_root=tmp_path / "out" / "eval" / "lane",
        report_path=report_path,
        verify_path=lane_root / "verify.json",
        profile="dev",
        steps=(
            WorkflowCommandStep(
                "evaluate", (sys.executable, str(script)), log_mode="w"
            ),
            WorkflowCommandStep(
                "verify",
                (sys.executable, "-c", "print('verify')"),
                output_path=lane_root / "verify.json",
                requires_report=True,
            ),
        ),
    )

    observed: list[tuple[str, tuple[str, ...]]] = []

    result = execute_workflow_lane(
        WorkflowLaneExecutionRequest(
            plan=plan,
            cwd=tmp_path,
            env=os.environ,
            log_path=tmp_path / "lane.log",
        ),
        after_successful_step=lambda _plan, step, run: observed.append(
            (step.name, run.command)
        ),
    )

    assert result.status == "ok"
    assert result.evaluate_exit == 0
    assert result.verify_exit == 0
    assert [name for name, _cmd in observed] == ["evaluate", "verify"]


def test_execute_workflow_lane_fails_verify_when_report_missing(tmp_path: Path) -> None:
    lane_root = tmp_path / "exec" / "eval" / "lane"
    plan = WorkflowLanePlan(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        execution_mode="host",
        preset="configs/preset.yaml",
        lane_root=lane_root,
        published_lane_root=tmp_path / "out" / "eval" / "lane",
        report_path=lane_root / "report" / "evaluation.report.json",
        verify_path=lane_root / "verify.json",
        profile="dev",
        steps=(
            WorkflowCommandStep(
                "evaluate",
                (sys.executable, "-c", "print('no report')"),
                log_mode="w",
            ),
            WorkflowCommandStep(
                "verify",
                (sys.executable, "-c", "raise SystemExit(99)"),
                output_path=lane_root / "verify.json",
                requires_report=True,
            ),
        ),
    )

    result = execute_workflow_lane(
        WorkflowLaneExecutionRequest(
            plan=plan,
            cwd=tmp_path,
            env=os.environ,
            log_path=tmp_path / "lane.log",
        )
    )

    assert result.status == "failed"
    assert result.detail == "report_missing"
    assert result.verify_exit is None


def test_execute_workflow_lane_skips_remaining_steps_after_failure(
    tmp_path: Path,
) -> None:
    lane_root = tmp_path / "exec" / "eval" / "lane"
    plan = WorkflowLanePlan(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        execution_mode="host",
        preset="configs/preset.yaml",
        lane_root=lane_root,
        published_lane_root=tmp_path / "out" / "eval" / "lane",
        report_path=lane_root / "report.json",
        verify_path=lane_root / "verify.json",
        profile="dev",
        steps=(
            WorkflowCommandStep(
                "evaluate",
                (sys.executable, "-c", "raise SystemExit(7)"),
                log_mode="w",
            ),
            WorkflowCommandStep(
                "verify",
                (sys.executable, "-c", "raise SystemExit(99)"),
            ),
        ),
    )
    seen: list[WorkflowLaneRunState] = []

    result = execute_workflow_lane(
        WorkflowLaneExecutionRequest(
            plan=plan,
            cwd=tmp_path,
            env=os.environ,
            log_path=tmp_path / "lane.log",
        ),
        after_lane=lambda _plan, state: seen.append(state),
    )

    assert result.status == "failed"
    assert result.evaluate_exit == 7
    assert [phase.name for phase in seen[0].phases] == ["evaluate"]


def test_execute_workflow_lane_uses_retry_steps_and_custom_result(
    tmp_path: Path,
) -> None:
    lane_root = tmp_path / "exec" / "eval" / "lane"
    script = tmp_path / "flaky.py"
    state_file = tmp_path / "state"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"state = Path({str(state_file)!r})\n"
        "if not state.exists():\n"
        "    state.write_text('seen')\n"
        "    raise SystemExit(3)\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    plan = WorkflowLanePlan(
        slug="lane",
        lane_id="lane-id",
        model_id="org/model",
        execution_mode="host",
        preset="configs/preset.yaml",
        lane_root=lane_root,
        published_lane_root=tmp_path / "out" / "eval" / "lane",
        report_path=lane_root / "report.json",
        verify_path=lane_root / "verify.json",
        profile="dev",
        steps=(
            WorkflowCommandStep(
                "evaluate",
                (sys.executable, str(script)),
                log_mode="w",
                retry_returncodes=(3,),
            ),
        ),
    )

    result = execute_workflow_lane(
        WorkflowLaneExecutionRequest(
            plan=plan,
            cwd=tmp_path,
            env=os.environ,
            log_path=tmp_path / "lane.log",
        ),
        lane_result=lambda plan, state, log_path: WorkflowLaneResult(
            slug=plan.slug,
            lane_id=plan.lane_id,
            model_id=plan.model_id,
            preset=plan.preset,
            evaluate_exit=state.phase_returncode("evaluate") or 0,
            verify_exit=None,
            report_path=state.report_path,
            verify_path=str(log_path),
            status="ok",
            detail="custom",
        ),
    )

    assert result.status == "ok"
    assert result.detail == "custom"
    assert result.verify_path.endswith("lane.log")


def test_execute_workflow_sweep_owns_status_log_and_fail_fast(tmp_path: Path) -> None:
    lanes: list[WorkflowLanePlan] = []
    for slug, exit_code in (("first", 1), ("second", 0)):
        lane_root = tmp_path / "exec" / "eval" / slug
        lanes.append(
            WorkflowLanePlan(
                slug=slug,
                lane_id=f"{slug}-id",
                model_id=f"org/{slug}",
                execution_mode="host",
                preset="configs/preset.yaml",
                lane_root=lane_root,
                published_lane_root=tmp_path / "out" / "eval" / slug,
                report_path=lane_root / "report" / "evaluation.report.json",
                verify_path=lane_root / "verify.json",
                profile="dev",
                steps=(
                    WorkflowCommandStep(
                        "evaluate",
                        (sys.executable, "-c", f"raise SystemExit({exit_code})"),
                        log_mode="w",
                    ),
                ),
            )
        )
    plan = WorkflowSweepPlan(
        metadata=WorkflowRunMetadata("suite", "host"),
        output_root=tmp_path / "out",
        execution_root=tmp_path / "exec",
        lanes=tuple(lanes),
    )

    seen_states: list[WorkflowLaneRunState] = []
    results = execute_workflow_sweep(
        WorkflowSweepExecutionRequest(
            plan=plan,
            cwd=tmp_path,
            env=os.environ,
            fail_fast=True,
        ),
        after_lane=lambda _plan, state: seen_states.append(state),
    )

    assert [result.slug for result in results] == ["first"]
    assert seen_states[0].phases[0].returncode == 1
    status_log = (tmp_path / "out" / "status.log").read_text(encoding="utf-8")
    assert "START first" in status_log
    assert "DONE first status=failed" in status_log
    assert "START second" not in status_log


def test_execute_workflow_sweep_warns_and_uses_lane_env_without_fail_fast(
    tmp_path: Path,
) -> None:
    lanes: list[WorkflowLanePlan] = []
    for slug, exit_code in (("first", 1), ("second", 0)):
        lane_root = tmp_path / "exec" / "eval" / slug
        lanes.append(
            WorkflowLanePlan(
                slug=slug,
                lane_id=f"{slug}-id",
                model_id=f"org/{slug}",
                execution_mode="host",
                preset="configs/preset.yaml",
                lane_root=lane_root,
                published_lane_root=tmp_path / "out" / "eval" / slug,
                report_path=lane_root / "report.json",
                verify_path=lane_root / "verify.json",
                profile="dev",
                resource_preflight={"warning": "low memory"} if slug == "first" else {},
                steps=(
                    WorkflowCommandStep(
                        "evaluate",
                        (
                            sys.executable,
                            "-c",
                            "import os, sys; assert os.environ['LANE_SLUG']; "
                            f"raise SystemExit({exit_code})",
                        ),
                        log_mode="w",
                    ),
                ),
            )
        )
    plan = WorkflowSweepPlan(
        metadata=WorkflowRunMetadata("suite", "host"),
        output_root=tmp_path / "out",
        execution_root=tmp_path / "exec",
        lanes=tuple(lanes),
    )

    results = execute_workflow_sweep(
        WorkflowSweepExecutionRequest(
            plan=plan,
            cwd=tmp_path,
            env={},
            fail_fast=False,
        ),
        lane_env=lambda lane, env: {**env, "LANE_SLUG": lane.slug},
    )

    assert [result.slug for result in results] == ["first", "second"]
    assert [result.status for result in results] == ["failed", "ok"]
    status_log = (tmp_path / "out" / "status.log").read_text(encoding="utf-8")
    assert "WARN first resource_preflight=low memory" in status_log
    assert "START second" in status_log
