from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.cli._support_verify_runtime_provenance import (
    _VALID_TEST_IMAGE_DIGEST,
    _matching_strict_policy_pack,
    _matching_strict_ppl_baseline,
    _strict_provenance_gate_cert,
    _write_runtime_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "integrations" / "public_e2e"
RUNNER = EXAMPLE_DIR / "run_public_e2e_release_review.sh"
README = EXAMPLE_DIR / "README.md"


def test_public_e2e_runner_smoke(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)

    report_payload = _strict_provenance_gate_cert()
    report = tmp_path / "source" / "evaluation.report.json"
    report.parent.mkdir()
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    _write_runtime_manifest(report)
    baseline = tmp_path / "acceptance-baseline.json"
    baseline.write_text(
        json.dumps(_matching_strict_ppl_baseline(report_payload)), encoding="utf-8"
    )
    policy_pack = tmp_path / "acceptance-policy-pack.json"
    policy_pack.write_text(
        json.dumps(_matching_strict_policy_pack(report_payload)), encoding="utf-8"
    )
    output_dir = tmp_path / "public-e2e"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            str(RUNNER),
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--policy-pack",
            str(policy_pack),
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--output-dir",
            str(output_dir),
            "--report-url",
            "https://example.test/evaluation.report.json",
            "--evidence-url",
            "https://example.test/evidence-pack",
            "--force",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "status: success" in result.stdout
    expected_files = [
        "evaluation.report.json",
        "baseline.report.json",
        "acceptance-policy-pack.json",
        "runtime.manifest.json",
        "invarlock-verify.json",
        "evaluation.html",
        "mlflow-tags.json",
        "model-card-invarlock.md",
        "release-review.md",
        "ci-summary.md",
        "run_command.txt",
        "run_summary.txt",
    ]
    for name in expected_files:
        assert (output_dir / name).is_file(), f"missing generated artifact: {name}"

    verify_payload = json.loads(
        (output_dir / "invarlock-verify.json").read_text(encoding="utf-8")
    )
    assert verify_payload["summary"]["ok"] is True
    runtime = verify_payload["results"][0]["verification"]["runtime_provenance"]
    assert runtime["expected_digest_matched"] is True
    assert verify_payload["results"][0]["id"] == str(
        output_dir / "evaluation.report.json"
    )

    tags = json.loads((output_dir / "mlflow-tags.json").read_text(encoding="utf-8"))[
        "tags"
    ]
    assert tags["invarlock.status"] == "receipt_bound_untrusted"
    assert tags["invarlock.report_local_status"] == "pass"
    assert tags["invarlock.verifier_status"] == "receipt_bound_untrusted"
    assert tags["invarlock.verifier_outcome"] == "pass"
    assert tags["invarlock.receipt_status"] == "bound_unsigned"
    assert tags["invarlock.policy_profile"] == "ci"

    html = (output_dir / "evaluation.html").read_text(encoding="utf-8")
    assert "<html" in html.lower()
    model_card = (output_dir / "model-card-invarlock.md").read_text(encoding="utf-8")
    assert "## InvarLock Evidence" in model_card
    assert "https://example.test/evaluation.report.json" in model_card
    assert "https://example.test/evidence-pack" in model_card
    release_review = (output_dir / "release-review.md").read_text(encoding="utf-8")
    assert "# InvarLock Release Review" in release_review
    assert "- Status: **RECEIPT_BOUND_UNTRUSTED**" in release_review
    ci_summary = (output_dir / "ci-summary.md").read_text(encoding="utf-8")
    assert ci_summary.startswith("### InvarLock\n\n# InvarLock Release Review")
    run_command = (output_dir / "run_command.txt").read_text(encoding="utf-8")
    assert "run_public_e2e_release_review.sh" in run_command
    assert not (output_dir / "source_run_command.txt").exists()


def test_public_e2e_readme_scopes_claims() -> None:
    text = README.read_text(encoding="utf-8")

    assert "Status: `reference-pattern`" in text
    assert "Script: requires caller-supplied current evidence." in text
    assert "copy/vendor `.github/actions/invarlock-report-gate/`" in text
    assert "does not regenerate the subject" in text
    assert "checkpoint" in text
    assert "push to MLflow" in text
    assert "update Hugging Face Hub" in text
    assert "approve a deployment" in text
    assert "--policy-pack /path/to/acceptance-policy-pack.json" in text
    assert "`baseline`, `policy-pack`, and expected image digest" in text
    assert "caller-supplied report" in text
    assert "The script accepts a current evaluation report" in text
    assert "public_evidence/byoe_examples" not in text


def test_public_e2e_strict_requires_independent_policy_pack(tmp_path: Path) -> None:
    report = tmp_path / "evaluation.report.json"
    report.write_text("{}\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(RUNNER),
            "--assurance",
            "strict",
            "--profile",
            "release",
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "requires --policy-pack from an independent source" in result.stderr
    runner_text = RUNNER.read_text(encoding="utf-8")
    assert 'policy_args=(--policy-pack "$bundle_policy_pack")' in runner_text


def test_public_e2e_strict_routes_and_preserves_all_acceptance_inputs(
    tmp_path: Path,
) -> None:
    report_payload = _strict_provenance_gate_cert()
    report = tmp_path / "source" / "evaluation.report.json"
    report.parent.mkdir()
    report.write_text(json.dumps(report_payload), encoding="utf-8")
    _write_runtime_manifest(report)
    baseline = tmp_path / "acceptance-baseline.json"
    baseline.write_text(
        json.dumps(_matching_strict_ppl_baseline(report_payload)), encoding="utf-8"
    )
    policy_pack = tmp_path / "acceptance-policy-pack.json"
    policy_pack.write_text(
        json.dumps(_matching_strict_policy_pack(report_payload)), encoding="utf-8"
    )
    output_dir = tmp_path / "strict-output"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(
        [
            str(RUNNER),
            "--report",
            str(report),
            "--baseline",
            str(baseline),
            "--policy-pack",
            str(policy_pack),
            "--profile",
            "ci",
            "--assurance",
            "strict",
            "--expected-runtime-image-digest",
            _VALID_TEST_IMAGE_DIGEST,
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "baseline.report.json").read_bytes() == baseline.read_bytes()
    assert (output_dir / "acceptance-policy-pack.json").read_bytes() == (
        policy_pack.read_bytes()
    )
    verify_payload = json.loads(
        (output_dir / "invarlock-verify.json").read_text(encoding="utf-8")
    )
    receipt = verify_payload["results"][0]["verification"]["receipt"]
    assert receipt["baseline_report_sha256"]
    assert receipt["policy_pack_sha256"]
    assert receipt["inputs"]["expected_runtime_image_digest"] == (
        _VALID_TEST_IMAGE_DIGEST
    )
