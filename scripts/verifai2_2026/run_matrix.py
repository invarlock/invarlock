#!/usr/bin/env python3
"""
run_matrix.py
=============

Execute (or dry-run) a small verifier-trace matrix for F4/S1 experiments.

This is intentionally lightweight glue:
  - Runs `run_verifier_trace_pipeline.py` for each job in a JSON plan.
  - Optionally assembles a verifier-carrying artifact via pilot_assemble_artifact.py.
  - Optionally validates the artifact via schema_verify.py.

It does not run model generation or InvarLock evaluation; those live in the
surrounding GPU pipeline. The plan should point at existing input files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.verifai2_2026 import (
    pilot_assemble_artifact,
    run_verifier_trace_pipeline,
    schema_verify,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_flag_value(argv: list[str], flag: str) -> str | None:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True, help="Path to matrix plan JSON.")
    p.add_argument(
        "--execute",
        action="store_true",
        help="If set, run jobs. Otherwise, just print what would run.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If set, keep going after failures and return non-zero at end.",
    )
    args = p.parse_args(argv)

    plan = _read_json(args.plan)
    if not isinstance(plan, dict):
        print("plan must be a JSON object", file=sys.stderr)
        return 2
    if plan.get("schema_version") != "verifai2_matrix_plan.v1":
        print("plan.schema_version must be verifai2_matrix_plan.v1", file=sys.stderr)
        return 2

    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        print("plan.jobs must be a non-empty array", file=sys.stderr)
        return 2

    any_failed = False
    for j in jobs:
        if not isinstance(j, dict):
            print("job must be a JSON object", file=sys.stderr)
            return 2
        job_id = str(j.get("job_id", ""))
        if not job_id:
            print("job.job_id is required", file=sys.stderr)
            return 2

        trace_argv = j.get("trace_argv")
        if not isinstance(trace_argv, list) or not all(
            isinstance(x, str) for x in trace_argv
        ):
            print(
                f"job_id={job_id}: trace_argv must be an array of strings",
                file=sys.stderr,
            )
            return 2

        trace_out = _parse_flag_value(trace_argv, "--trace-out")
        if trace_out is None:
            print(
                f"job_id={job_id}: trace_argv must include --trace-out", file=sys.stderr
            )
            return 2

        print(f"[job {job_id}] run_verifier_trace_pipeline.py {' '.join(trace_argv)}")
        if args.execute:
            rc = int(run_verifier_trace_pipeline.main(trace_argv))
            if rc != 0:
                any_failed = True
                print(f"[job {job_id}] trace failed rc={rc}", file=sys.stderr)
                if not args.continue_on_error:
                    return rc
                continue

        artifact = j.get("artifact")
        if artifact is not None:
            if not isinstance(artifact, dict):
                print(
                    f"job_id={job_id}: artifact must be an object when present",
                    file=sys.stderr,
                )
                return 2

            eval_report = artifact.get("evaluation_report")
            artifact_out = artifact.get("out")
            if not (isinstance(eval_report, str) and isinstance(artifact_out, str)):
                print(
                    f"job_id={job_id}: artifact.evaluation_report and artifact.out are required",
                    file=sys.stderr,
                )
                return 2

            trace_paths = artifact.get("verifier_traces")
            if trace_paths is None:
                trace_paths = [trace_out]
            if not (
                isinstance(trace_paths, list)
                and trace_paths
                and all(isinstance(x, str) for x in trace_paths)
            ):
                print(
                    f"job_id={job_id}: artifact.verifier_traces must be a non-empty array",
                    file=sys.stderr,
                )
                return 2

            pa_argv: list[str] = [
                "--evaluation-report",
                str(eval_report),
                "--out",
                str(artifact_out),
            ]
            for t in trace_paths:
                pa_argv.extend(["--verifier-trace", t])
            if bool(artifact.get("embed_evaluation_report", False)):
                pa_argv.append("--embed-evaluation-report")
            if isinstance(artifact.get("verify_json"), str):
                pa_argv.extend(["--verify-json", str(artifact.get("verify_json"))])
            if isinstance(artifact.get("invarlock_version"), str):
                pa_argv.extend(
                    ["--invarlock-version", str(artifact.get("invarlock_version"))]
                )
            if isinstance(artifact.get("git_commit"), str):
                pa_argv.extend(["--git-commit", str(artifact.get("git_commit"))])

            print(f"[job {job_id}] pilot_assemble_artifact.py {' '.join(pa_argv)}")
            if args.execute:
                rc = int(pilot_assemble_artifact.main(pa_argv))
                if rc != 0:
                    any_failed = True
                    print(
                        f"[job {job_id}] artifact assembly failed rc={rc}",
                        file=sys.stderr,
                    )
                    if not args.continue_on_error:
                        return rc
                    continue

            if bool(artifact.get("validate", False)):
                sv_argv = [str(artifact_out)]
                if bool(artifact.get("check_files", False)):
                    sv_argv.append("--check-files")
                schema_root = artifact.get("schema_root")
                if isinstance(schema_root, str) and schema_root:
                    sv_argv.extend(["--schema-root", schema_root])

                print(f"[job {job_id}] schema_verify.py {' '.join(sv_argv)}")
                if args.execute:
                    rc = int(schema_verify.main(sv_argv))
                    if rc != 0:
                        any_failed = True
                        print(
                            f"[job {job_id}] artifact validation failed rc={rc}",
                            file=sys.stderr,
                        )
                        if not args.continue_on_error:
                            return rc

    return 2 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
