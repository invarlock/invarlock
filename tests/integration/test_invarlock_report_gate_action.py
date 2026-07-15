from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from tests.cli.verify._support_runtime_provenance import (
    _VALID_TEST_IMAGE_DIGEST,
    _bind_strict_baseline,
    _final_window_schedule_digest,
    _matching_strict_policy_pack,
    _matching_strict_ppl_baseline,
    _strict_provenance_gate_cert,
    _write_runtime_manifest,
    bind_runtime_policy_receipt,
)
from tests.core._support_assurance_contract import bind_noop_variance_evidence
from tests.core._support_guard_metric_impact import bind_guard_metric_impact_evidence


def test_report_gate_action_threads_verify_result_into_exports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    action_path = (
        repo_root / ".github" / "actions" / "invarlock-report-gate" / "action.yml"
    )
    action = yaml.safe_load(action_path.read_text(encoding="utf-8"))

    inputs = action["inputs"]
    for name in (
        "assurance",
        "baseline",
        "expected-runtime-image-digest",
        "policy-pack",
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
    assert "--baseline" in verify_step
    assert "--expected-runtime-image-digest" in verify_step
    assert "--policy-pack" in verify_step
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
    assert "${{ inputs.policy-pack }}" in upload_step["with"]["path"]

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
    report = tmp_path / "evaluation.report.json"
    report_payload = _strict_provenance_gate_cert()
    report_payload["assurance"]["profile"] = "release"
    report_payload["context"]["profile"] = "release"
    report_payload["meta"]["profile"] = "release"
    report_payload["primary_metric_tail"] = {
        "mode": "fail",
        "evaluated": True,
        "passed": True,
    }
    report_payload["validation"]["primary_metric_tail_acceptable"] = True
    release_windows = 400
    preview_ids = list(range(release_windows))
    final_ids = list(range(release_windows, release_windows * 2))
    final_schedule_digest = _final_window_schedule_digest(final_ids)
    window_stats = report_payload["dataset"]["windows"]["stats"]
    report_payload["dataset"]["windows"]["preview"] = release_windows
    report_payload["dataset"]["windows"]["final"] = release_windows
    window_stats["actual_preview"] = release_windows
    window_stats["actual_final"] = release_windows
    window_stats["paired_windows"] = release_windows
    for phase in ("preview", "final"):
        window_stats["coverage"][phase].update(
            {"used": release_windows, "required": 200, "ok": True}
        )
    window_stats["coverage"]["replicates"].update(
        {"used": 3200, "required": 3200, "ok": True}
    )
    window_stats["bootstrap"]["replicates"] = 3200
    window_stats["preview_final_slice_delta_summary"].update(
        {"preview_windows": release_windows, "final_windows": release_windows}
    )
    logloss = report_payload["evaluation_windows"]["final"]["logloss"][0]
    report_payload["evaluation_windows"] = {
        "preview": {
            "window_ids": preview_ids,
            "logloss": [logloss] * release_windows,
            "token_counts": [1] * release_windows,
        },
        "final": {
            "window_ids": final_ids,
            "logloss": [logloss] * release_windows,
            "token_counts": [1] * release_windows,
        },
    }
    report_payload["provenance"]["window_ids_digest"] = final_schedule_digest
    report_payload["provenance"]["window_plan_digest"] = final_schedule_digest
    bind_noop_variance_evidence(report_payload)
    bind_guard_metric_impact_evidence(report_payload)
    bind_runtime_policy_receipt(report_payload)
    baseline = tmp_path / "acceptance-baseline.json"
    baseline_payload = _matching_strict_ppl_baseline(report_payload)
    baseline_payload["context"]["profile"] = "release"
    _bind_strict_baseline(report_payload, baseline_payload)
    report.write_text(json.dumps(report_payload, sort_keys=True), encoding="utf-8")
    _write_runtime_manifest(report)
    baseline.write_text(
        json.dumps(baseline_payload, sort_keys=True),
        encoding="utf-8",
    )
    policy_pack = tmp_path / "acceptance-policy-pack.json"
    policy_pack.write_text(
        json.dumps(_matching_strict_policy_pack(report_payload), sort_keys=True),
        encoding="utf-8",
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
                "--baseline",
                str(baseline),
                "--policy-pack",
                str(policy_pack),
                "--expected-runtime-image-digest",
                _VALID_TEST_IMAGE_DIGEST,
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
    assert tags["invarlock.status"] == "receipt_bound_untrusted"
    assert tags["invarlock.report_local_status"] == "pass"
    assert tags["invarlock.verifier_status"] == "receipt_bound_untrusted"
    assert tags["invarlock.verifier_outcome"] == "pass"
    assert tags["invarlock.receipt_status"] == "bound_unsigned"
    verify_payload = json.loads(verify_out.read_text(encoding="utf-8"))
    runtime = verify_payload["results"][0]["verification"]["runtime_provenance"]
    assert runtime["expected_digest_matched"] is True
