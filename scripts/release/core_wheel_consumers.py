#!/usr/bin/env python3
"""Run the maintained core-wheel consumers outside the source checkout."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUDGE_FILES = (
    "wheel_smoke.py",
    "request.yaml",
    "plan.json",
    "measurements.json",
    "baseline_run.json",
    "subject_run.json",
    "analysis_policy.json",
    "collection.json",
)
APPROVAL = "examples/evaluator-qualification/signed-transactions/deployment-approval-inspect-ai"
FILES = (
    ("examples/quickstart/run.py", "run.py"),
    ("examples/captured-results/wheel_smoke.py", "captured-wheel-smoke.py"),
    ("examples/captured-results/scorer_wheel_smoke.py", "scorer-wheel-smoke.py"),
    *((f"examples/judge-measurements/{name}", f"judge/{name}") for name in JUDGE_FILES),
    (
        f"{APPROVAL}/verification.receipt.json",
        "approval/incoming/verification.receipt.json",
    ),
)
DIRECTORIES = (
    ("examples/acceptance-handoff/golden", "golden"),
    ("examples/ci/standalone-consumer", "approval"),
    (f"{APPROVAL}/evidence", "approval/incoming/evidence"),
)


def run_consumers(cli: str) -> None:
    executable = shutil.which(cli)
    if executable is None:
        raise RuntimeError(
            "Install the candidate wheel and supply its invarlock executable"
        )
    executable = str(Path(executable).absolute())
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONSAFEPATH="1")
    with tempfile.TemporaryDirectory(prefix="invarlock-core-consumers-") as directory:
        root = Path(directory).resolve()
        if root.is_relative_to(ROOT):
            raise RuntimeError(
                "Wheel consumers require a temporary directory outside the checkout"
            )
        for source, destination in DIRECTORIES:
            shutil.copytree(ROOT / source, root / destination)
        for source, destination in FILES:
            target = root / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, target)
        commands = (
            (".", ("run.py", "--fixture", "golden")),
            (".", ("captured-wheel-smoke.py", "--cli", executable)),
            (".", ("judge/wheel_smoke.py", "--fixture", "judge", "--cli", executable)),
            (".", ("scorer-wheel-smoke.py", "--fixture", "judge", "--cli", executable)),
            (
                "approval",
                (
                    "review/verify_deployment_receipt.py",
                    "--approval-inputs",
                    "review/inspect-ai-deployment-approval-inputs.json",
                    "--evidence",
                    "incoming/evidence",
                    "--policy",
                    "review/policy/acceptance.json",
                    "--receipt",
                    "incoming/verification.receipt.json",
                    "--output",
                    "deployment-approval.json",
                ),
            ),
        )
        # Reject source imports before any consumer can produce misleading evidence.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sysconfig; import invarlock; "
                "assert Path(invarlock.__file__).resolve().is_relative_to("
                "Path(sysconfig.get_path('purelib')).resolve())",
            ],
            cwd=root,
            env=environment,
            check=True,
        )
        for working_directory, arguments in commands:
            subprocess.run(
                [sys.executable, *arguments],
                cwd=root / working_directory,
                env=environment,
                check=True,
            )


def exit_on_signal(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True)
    args = parser.parse_args()
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, exit_on_signal)
    run_consumers(args.cli)


if __name__ == "__main__":
    main()
