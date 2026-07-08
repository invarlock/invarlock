from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_report_gate_action_threads_verify_result_into_exports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    action_path = (
        repo_root / ".github" / "actions" / "invarlock-report-gate" / "action.yml"
    )
    action = yaml.safe_load(action_path.read_text(encoding="utf-8"))

    inputs = action["inputs"]
    for name in (
        "assurance",
        "runtime-provenance",
        "warning-policy",
        "verify-output",
    ):
        assert name in inputs

    runs = [
        step.get("run", "")
        for step in action["runs"]["steps"]
        if isinstance(step, dict)
    ]
    steps = action["runs"]["steps"]
    verify_step = "\n".join(run for run in runs if "invarlock verify" in run)
    export_steps = "\n".join(run for run in runs if "report export" in run)
    fail_step = "\n".join(
        run for run in runs if "INVARLOCK_VERIFY_EXIT_CODE" in run and "exit" in run
    )

    assert "--assurance" in verify_step
    assert "--runtime-provenance" in verify_step
    assert "--warning-policy" in verify_step
    assert "${{ inputs.verify-output }}" in verify_step
    assert "INVARLOCK_VERIFY_EXIT_CODE" in verify_step
    assert 'exit "$status"' not in verify_step
    assert '--verify-result "${{ inputs.verify-output }}"' in export_steps
    assert "INVARLOCK_VERIFY_EXIT_CODE" in fail_step
    assert "${{ inputs.fail-on-verify }}" in fail_step

    upload_step = next(
        step
        for step in action["runs"]["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert "${{ inputs.verify-output }}" in upload_step["with"]["path"]

    upload_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    fail_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Fail on InvarLock verification result"
    )
    assert upload_index < fail_index


def test_report_gate_command_sequence_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report = (
        repo_root
        / "public_evidence"
        / "real_runs"
        / "tiny_gpt2_external_magnitude_prune"
        / "evidence_pack"
        / "reports"
        / "report-001"
        / "evaluation.report.json"
    )
    verify_out = tmp_path / "invarlock-verify.json"
    html_out = tmp_path / "evaluation.html"
    mlflow_out = tmp_path / "mlflow-tags.json"
    review_out = tmp_path / "release-review.md"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    with verify_out.open("w", encoding="utf-8") as stdout:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "invarlock",
                "verify",
                str(report),
                "--profile",
                "release",
                "--assurance",
                "strict",
                "--json",
            ],
            cwd=repo_root,
            env=env,
            stdout=stdout,
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "invarlock",
            "report",
            "html",
            "--input",
            str(report),
            "--output",
            str(html_out),
            "--force",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "invarlock",
            "report",
            "export",
            "--evaluation-report",
            str(report),
            "--format",
            "mlflow-tags",
            "--policy-profile",
            "release",
            "--verify-result",
            str(verify_out),
            "--output",
            str(mlflow_out),
            "--force",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "invarlock",
            "report",
            "export",
            "--evaluation-report",
            str(report),
            "--format",
            "release-review-md",
            "--policy-profile",
            "release",
            "--verify-result",
            str(verify_out),
            "--output",
            str(review_out),
            "--force",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    assert verify_out.is_file()
    assert html_out.is_file()
    assert mlflow_out.is_file()
    assert review_out.is_file()
    tags = json.loads(mlflow_out.read_text(encoding="utf-8"))["tags"]
    assert tags["invarlock.status"] == "pass"
    assert tags["invarlock.verifier_status"] == "pass"
