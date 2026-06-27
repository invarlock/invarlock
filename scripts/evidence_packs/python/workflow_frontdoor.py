#!/usr/bin/env python3
"""Typed workflow front door for evidence-pack shell entrypoints."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evidence_workflows import (  # noqa: E402
    WorkflowCommandStep,
    WorkflowLanePlan,
    WorkflowLaneResult,
    WorkflowRunMetadata,
    WorkflowSweepExecutionRequest,
    WorkflowSweepPlan,
    execute_workflow_sweep,
    workflow_return_code,
)
from evidence_workflows.workflow_state import (  # noqa: E402
    write_artifact_manifest,
    write_summary_files,
)

EVIDENCE_PACK_ROOT = REPO_ROOT / "scripts" / "evidence_packs"
FRONTDOOR_SCRIPTS = {
    "run-suite": EVIDENCE_PACK_ROOT / "run_suite.sh",
    "run-pack": EVIDENCE_PACK_ROOT / "run_pack.sh",
    "mini-pack": EVIDENCE_PACK_ROOT / "run_mini_pack_gate.sh",
}


@dataclass(frozen=True)
class EvidencePackWorkflowRequest:
    frontdoor: str
    args: tuple[str, ...]
    output_root: Path
    dry_run: bool = False


def _default_output_root(frontdoor: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "tmp" / "evidence_pack_workflows" / f"{frontdoor}-{stamp}"


def build_evidence_pack_workflow(
    request: EvidencePackWorkflowRequest,
) -> WorkflowSweepPlan:
    script = FRONTDOOR_SCRIPTS[request.frontdoor]
    lane_root = request.output_root / request.frontdoor
    step_name = request.frontdoor.replace("-", "_")
    command = ("bash", str(script), *request.args)
    step = WorkflowCommandStep(
        name=step_name,
        command=command,
        output_path=lane_root / f"{step_name}.log",
    )
    lane = WorkflowLanePlan(
        slug=request.frontdoor,
        lane_id=request.frontdoor,
        model_id="evidence-pack",
        execution_mode="local",
        preset="scripts/evidence_packs/scenarios.json",
        lane_root=lane_root,
        published_lane_root=lane_root,
        report_path=lane_root / "summary.json",
        verify_path=lane_root / "verify.json",
        profile="evidence-pack",
        steps=(step,),
        resource_preflight={"frontdoor": request.frontdoor},
    )
    return WorkflowSweepPlan(
        metadata=WorkflowRunMetadata(
            suite=f"evidence-pack/{request.frontdoor}",
            execution_mode="local",
        ),
        output_root=request.output_root,
        execution_root=REPO_ROOT,
        lanes=(lane,),
    )


def _parse_args(argv: list[str]) -> EvidencePackWorkflowRequest:
    if not argv or argv[0] in {"--help", "-h"}:
        commands = ", ".join(sorted(FRONTDOOR_SCRIPTS))
        raise SystemExit(f"Usage: workflow_frontdoor.py <{commands}> [--] [args...]")
    frontdoor = argv[0]
    if frontdoor not in FRONTDOOR_SCRIPTS:
        raise SystemExit(f"Unknown evidence-pack workflow frontdoor: {frontdoor}")

    output_root: Path | None = None
    dry_run = False
    script_args: list[str] = []
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            script_args.extend(argv[index + 1 :])
            break
        if arg == "--dry-run":
            dry_run = True
            index += 1
            continue
        if arg == "--output-root":
            if index + 1 >= len(argv):
                raise SystemExit("--output-root requires a value")
            output_root = Path(argv[index + 1])
            index += 2
            continue
        script_args.append(arg)
        index += 1

    if output_root is None:
        env_root = os.environ.get("PACK_WORKFLOW_OUTPUT_ROOT")
        output_root = Path(env_root) if env_root else _default_output_root(frontdoor)

    return EvidencePackWorkflowRequest(
        frontdoor=frontdoor,
        args=tuple(script_args),
        output_root=output_root,
        dry_run=dry_run,
    )


def _tail_text(path: Path, *, max_lines: int = 80) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _emit_failure_diagnostics(
    *,
    output_root: Path,
    results: list[WorkflowLaneResult],
) -> None:
    failed = [result for result in results if not result.ok]
    if not failed:
        return

    print(
        f"Evidence-pack workflow failed; output_root={output_root}",
        file=sys.stderr,
    )
    status_log = output_root / "status.log"
    if status_log.is_file():
        print(f"Status log: {status_log}", file=sys.stderr)
        status_tail = _tail_text(status_log, max_lines=40)
        if status_tail:
            print(status_tail, file=sys.stderr)

    logs_dir = output_root / "logs"
    for result in failed:
        verify_exit = "NA" if result.verify_exit is None else str(result.verify_exit)
        print(
            "Failed lane "
            f"{result.slug}: status={result.status} detail={result.detail or '-'} "
            f"evaluate_exit={result.evaluate_exit} verify_exit={verify_exit}",
            file=sys.stderr,
        )
        lane_log = logs_dir / f"{result.slug}.log"
        if lane_log.is_file():
            print(f"Log tail: {lane_log}", file=sys.stderr)
            log_tail = _tail_text(lane_log)
            if log_tail:
                print(log_tail, file=sys.stderr)


def run_evidence_pack_workflow(request: EvidencePackWorkflowRequest) -> int:
    workflow = build_evidence_pack_workflow(request)
    if request.dry_run:
        print(
            json.dumps(
                {
                    "workflow": workflow.to_dry_run_payload(),
                    "output_root": str(workflow.output_root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    env = dict(os.environ)
    env["PACK_WORKFLOW_SUBPROCESS"] = "1"
    results = execute_workflow_sweep(
        WorkflowSweepExecutionRequest(
            plan=workflow,
            cwd=workflow.execution_root,
            env=env,
            fail_fast=True,
            status_log_path=workflow.output_root / "status.log",
        )
    )
    write_summary_files(
        workflow.output_root,
        metadata=workflow.metadata,
        results=results,
    )
    write_artifact_manifest(
        workflow.output_root,
        schema="invarlock/evidence-pack-workflow-artifacts/v1",
        metadata=workflow.metadata,
        results=results,
        artifact_patterns=("**/*.log", "summary.json", "status.log"),
    )
    rc = workflow_return_code(results)
    if rc != 0:
        _emit_failure_diagnostics(output_root=workflow.output_root, results=results)
    return rc


def main(argv: list[str] | None = None) -> int:
    request = _parse_args(list(sys.argv[1:] if argv is None else argv))
    return run_evidence_pack_workflow(request)


if __name__ == "__main__":
    raise SystemExit(main())
