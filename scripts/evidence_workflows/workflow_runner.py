"""Reusable execution helpers for evidence workflow scripts."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .workflow_plan import WorkflowCommandStep, WorkflowLanePlan, WorkflowSweepPlan
from .workflow_state import (
    WorkflowLaneResult,
    WorkflowLaneRunState,
    WorkflowPhaseResult,
)


@dataclass(frozen=True)
class WorkflowCommandRun:
    name: str
    command: tuple[str, ...]
    returncode: int
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowLaneExecutionRequest:
    """Runtime inputs for executing one planned evidence lane."""

    plan: WorkflowLanePlan
    cwd: Path
    env: Mapping[str, str]
    log_path: Path


@dataclass(frozen=True)
class WorkflowSweepExecutionRequest:
    """Runtime inputs for executing a planned evidence sweep."""

    plan: WorkflowSweepPlan
    cwd: Path
    env: Mapping[str, str]
    fail_fast: bool = False
    status_log_path: Path | None = None
    log_dir: Path | None = None


AfterStepHook = Callable[
    [WorkflowLanePlan, WorkflowCommandStep, WorkflowCommandRun],
    None,
]
AfterLaneHook = Callable[[WorkflowLanePlan, WorkflowLaneRunState], None]
LaneResultHook = Callable[
    [WorkflowLanePlan, WorkflowLaneRunState, Path],
    WorkflowLaneResult,
]


def workflow_return_code(results: Sequence[object]) -> int:
    """Return process exit code for a sequence with result.ok semantics."""
    return (
        0
        if results and all(bool(getattr(result, "ok", False)) for result in results)
        else 1
    )


def write_status_event(
    handle,
    event: str,
    *,
    slug: str | None = None,
    fields: Mapping[str, object] | None = None,
) -> None:
    """Write a stable status.log event line."""
    parts = [f"[{datetime.now(UTC).isoformat()}]", event]
    if slug:
        parts.append(slug)
    for key, value in (fields or {}).items():
        rendered = "-" if value is None else str(value)
        parts.append(f"{key}={rendered}")
    handle.write(" ".join(parts) + "\n")
    handle.flush()


def _default_lane_result(
    plan: WorkflowLanePlan,
    state: WorkflowLaneRunState,
    _log_path: Path,
) -> WorkflowLaneResult:
    return state.to_lane_result()


def run_logged_command(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    log_mode: str = "a",
    output_path: Path | None = None,
) -> WorkflowCommandRun:
    """Run a command while recording the command line and output sidecars."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            output_path.open("w", encoding="utf-8") as output_file,
            log_path.open(log_mode, encoding="utf-8") as log_file,
        ):
            log_file.write("$ " + " ".join(command) + "\n")
            proc = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdout=output_file,
                stderr=log_file,
                text=True,
                check=False,
            )
    else:
        with log_path.open(log_mode, encoding="utf-8") as log_file:
            log_file.write("$ " + " ".join(command) + "\n")
            proc = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    return WorkflowCommandRun(
        name=name,
        command=tuple(command),
        returncode=proc.returncode,
    )


def run_logged_command_with_retry(
    *,
    name: str,
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    log_mode: str = "a",
    output_path: Path | None = None,
    retry_returncodes: Sequence[int] = (),
    retry_message: str | None = None,
) -> WorkflowCommandRun:
    """Run a logged command and retry once for configured transient exits."""
    first = run_logged_command(
        name=name,
        command=command,
        cwd=cwd,
        env=env,
        log_path=log_path,
        log_mode=log_mode,
        output_path=output_path,
    )
    if first.returncode not in set(retry_returncodes):
        return first
    with log_path.open("a", encoding="utf-8") as log_file:
        message = retry_message or (
            f"{name} exited with {first.returncode}; retrying once."
        )
        message = message.format(returncode=first.returncode, name=name)
        log_file.write(f"\n[WARN] {message}\n")
    second = run_logged_command(
        name=name,
        command=command,
        cwd=cwd,
        env=env,
        log_path=log_path,
        log_mode="a",
        output_path=output_path,
    )
    return WorkflowCommandRun(
        name=name,
        command=tuple(command),
        returncode=second.returncode,
        attempts=2,
    )


def execute_workflow_lane(
    request: WorkflowLaneExecutionRequest,
    *,
    after_successful_step: AfterStepHook | None = None,
    after_lane: AfterLaneHook | None = None,
    lane_result: LaneResultHook | None = None,
) -> WorkflowLaneResult:
    """Execute a planned evidence lane and return a typed lane result.

    Domain scripts own command construction and domain-specific failure
    classification. This function owns command sequencing, retry handling,
    report-dependent step gating, and phase rollup.
    """

    plan = request.plan
    plan.lane_root.mkdir(parents=True, exist_ok=True)
    phases: list[WorkflowPhaseResult] = []

    for step in plan.steps:
        if step.requires_report and not plan.report_path.is_file():
            phases.append(
                WorkflowPhaseResult(
                    step.name,
                    None,
                    "failed",
                    "report_missing",
                )
            )
            break

        if step.retry_returncodes:
            command_run = run_logged_command_with_retry(
                name=step.name,
                command=step.command,
                cwd=request.cwd,
                env=request.env,
                log_path=request.log_path,
                log_mode=step.log_mode,
                output_path=step.output_path,
                retry_returncodes=step.retry_returncodes,
                retry_message=step.retry_message,
            )
        else:
            command_run = run_logged_command(
                name=step.name,
                command=step.command,
                cwd=request.cwd,
                env=request.env,
                log_path=request.log_path,
                log_mode=step.log_mode,
                output_path=step.output_path,
            )
        phase_status = "ok" if command_run.ok else "failed"
        phases.append(
            WorkflowPhaseResult(
                step.name,
                command_run.returncode,
                phase_status,
            )
        )
        if command_run.ok and after_successful_step is not None:
            after_successful_step(plan, step, command_run)
        if not command_run.ok:
            break

    state = WorkflowLaneRunState(
        slug=plan.slug,
        lane_id=plan.lane_id,
        model_id=plan.model_id,
        preset=plan.preset,
        report_path=str(plan.report_path),
        verify_path=str(plan.verify_path) if plan.verify_path else None,
        phases=tuple(phases),
    )
    if after_lane is not None:
        after_lane(plan, state)
    result_factory = lane_result or _default_lane_result
    return result_factory(plan, state, request.log_path)


def execute_workflow_sweep(
    request: WorkflowSweepExecutionRequest,
    *,
    lane_env: Callable[[WorkflowLanePlan, Mapping[str, str]], Mapping[str, str]]
    | None = None,
    after_successful_step: AfterStepHook | None = None,
    after_lane: AfterLaneHook | None = None,
    lane_result: LaneResultHook | None = None,
) -> list[WorkflowLaneResult]:
    """Execute all lanes in a planned evidence sweep."""

    plan = request.plan
    status_log_path = request.status_log_path or (plan.output_root / "status.log")
    log_dir = request.log_dir or (plan.output_root / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[WorkflowLaneResult] = []

    with status_log_path.open("w", encoding="utf-8") as handle:
        write_status_event(handle, "START")
        for lane_plan in plan.lanes:
            preflight = lane_plan.resource_preflight
            if preflight and preflight.get("warning"):
                write_status_event(
                    handle,
                    "WARN",
                    slug=lane_plan.slug,
                    fields={"resource_preflight": preflight["warning"]},
                )
            write_status_event(handle, "START", slug=lane_plan.slug)
            effective_env = (
                lane_env(lane_plan, request.env) if lane_env else dict(request.env)
            )
            result = execute_workflow_lane(
                WorkflowLaneExecutionRequest(
                    plan=lane_plan,
                    cwd=request.cwd,
                    env=effective_env,
                    log_path=log_dir / f"{lane_plan.slug}.log",
                ),
                after_successful_step=after_successful_step,
                after_lane=after_lane,
                lane_result=lane_result,
            )
            results.append(result)
            verify_repr = (
                "NA" if result.verify_exit is None else str(result.verify_exit)
            )
            write_status_event(
                handle,
                "DONE",
                slug=result.slug,
                fields={
                    "status": result.status,
                    "detail": result.detail or "-",
                    "eval": result.evaluate_exit,
                    "verify": verify_repr,
                },
            )
            if request.fail_fast and not result.ok:
                break
        write_status_event(handle, "ALL_TASKS_COMPLETE")
    return results


__all__ = [
    "AfterLaneHook",
    "AfterStepHook",
    "LaneResultHook",
    "WorkflowCommandRun",
    "WorkflowLaneExecutionRequest",
    "WorkflowSweepExecutionRequest",
    "execute_workflow_lane",
    "execute_workflow_sweep",
    "run_logged_command",
    "run_logged_command_with_retry",
    "workflow_return_code",
    "write_status_event",
]
