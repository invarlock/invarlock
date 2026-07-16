from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_PATH = REPO_ROOT / ".github/actions/invarlock-report-gate/action.yml"


def _action() -> dict[str, Any]:
    value = yaml.safe_load(ACTION_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _step(action: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in action["runs"]["steps"] if step.get("name") == name)


def test_action_exposes_only_the_evidence_gate_inputs() -> None:
    action = _action()
    inputs = action["inputs"]

    assert set(inputs) == {
        "evidence",
        "policy",
        "expected-baseline-artifact",
        "expected-subject-artifact",
        "expected-schedule",
        "expected-baseline-runtime",
        "expected-subject-runtime",
        "expected-signer",
        "verifier-signing-key",
        "verifier-identity",
        "python",
        "receipt-output",
        "verify-output",
        "html-output",
        "artifact-name",
        "fail-on-verify",
    }
    for required in (
        "evidence",
        "policy",
        "expected-baseline-artifact",
        "expected-subject-artifact",
        "expected-schedule",
        "expected-baseline-runtime",
        "expected-subject-runtime",
        "expected-signer",
        "verifier-signing-key",
        "verifier-identity",
    ):
        assert inputs[required]["required"] is True

    text = ACTION_PATH.read_text(encoding="utf-8")
    for retired in (
        "policy-pack",
        "profile",
        "assurance",
        "runtime-provenance",
        "warning-policy",
        "report html",
        "report export",
        "mlflow",
        "review-output",
    ):
        assert retired not in text.lower()


def test_action_calls_current_verify_and_report_transactions() -> None:
    action = _action()
    prepare = _step(action, "Prepare InvarLock output directories")["run"]
    verify = _step(action, "Verify InvarLock evidence")["run"]
    report = _step(action, "Render InvarLock HTML report")["run"]

    assert "output paths must be distinct" in prepare
    assert "output must remain outside evidence" in prepare
    assert "output already exists" in prepare
    assert '("policy", policy)' in prepare
    assert '("verifier signing key", verifier_key)' in prepare
    assert "must remain outside evidence" in prepare
    assert "set -o noclobber" in verify
    assert "-m invarlock verify" in verify
    for option in (
        "--policy",
        "--expected-baseline-artifact",
        "--expected-subject-artifact",
        "--expected-schedule",
        "--expected-baseline-runtime",
        "--expected-subject-runtime",
        "--expected-signer",
        "--receipt",
        "--verifier-signing-key",
        "--verifier-identity",
        "--json",
    ):
        assert option in verify

    assert "-m invarlock report" in report
    assert "--html" in report
    assert "--explain" in report
    assert "report html" not in report


def test_action_preserves_review_outputs_without_uploading_the_verifier_key() -> None:
    action = _action()
    upload = _step(action, "Upload InvarLock evidence")
    uploaded_paths = upload["with"]["path"]

    assert upload["if"] == (
        "${{ always() && env.INVARLOCK_ACTION_LAYOUT_SAFE == 'true' }}"
    )
    assert upload["uses"] == (
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
    )
    for expected in (
        "inputs.evidence",
        "inputs.verify-output",
        "inputs.receipt-output",
        "inputs.html-output",
    ):
        assert expected in uploaded_paths
    assert "verifier-signing-key" not in uploaded_paths
    assert "inputs.policy" not in uploaded_paths

    enforce = _step(action, "Enforce InvarLock verification result")["run"]
    assert "INVARLOCK_VERIFY_EXIT_CODE:-1" in enforce
    assert "INVARLOCK_ACTION_FAIL_ON_VERIFY" in enforce


def test_action_never_interpolates_untrusted_inputs_into_shell_source() -> None:
    action = _action()

    for step in action["runs"]["steps"]:
        run = step.get("run")
        if isinstance(run, str):
            assert "${{ inputs." not in run


def test_action_rejects_a_verifier_key_inside_the_uploaded_evidence(
    tmp_path: Path,
) -> None:
    action = _action()
    script = _step(action, "Prepare InvarLock output directories")["run"]
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    verifier_key = evidence / "verifier.pem"
    verifier_key.write_text("secret", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    github_env = tmp_path / "github.env"
    environment = dict(os.environ)
    environment.update(
        {
            "GITHUB_ENV": str(github_env),
            "INVARLOCK_ACTION_PYTHON": sys.executable,
            "INVARLOCK_ACTION_EVIDENCE": str(evidence),
            "INVARLOCK_ACTION_POLICY": str(policy),
            "INVARLOCK_ACTION_VERIFIER_SIGNING_KEY": str(verifier_key),
            "INVARLOCK_ACTION_RECEIPT_OUTPUT": str(tmp_path / "receipt.json"),
            "INVARLOCK_ACTION_VERIFY_OUTPUT": str(tmp_path / "verify.json"),
            "INVARLOCK_ACTION_HTML_OUTPUT": str(tmp_path / "report.html"),
            "INVARLOCK_ACTION_ARTIFACT_NAME": "review-artifact",
        }
    )

    completed = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "verifier signing key must remain outside evidence" in completed.stderr
    assert not github_env.exists()


def test_action_rejects_evidence_symlinks_before_artifact_upload(
    tmp_path: Path,
) -> None:
    action = _action()
    script = _step(action, "Prepare InvarLock output directories")["run"]
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    verifier_key = tmp_path / "verifier.pem"
    verifier_key.write_text("secret", encoding="utf-8")
    (evidence / "leaked-key.pem").symlink_to(verifier_key)
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    github_env = tmp_path / "github.env"
    environment = dict(os.environ)
    environment.update(
        {
            "GITHUB_ENV": str(github_env),
            "INVARLOCK_ACTION_PYTHON": sys.executable,
            "INVARLOCK_ACTION_EVIDENCE": str(evidence),
            "INVARLOCK_ACTION_POLICY": str(policy),
            "INVARLOCK_ACTION_VERIFIER_SIGNING_KEY": str(verifier_key),
            "INVARLOCK_ACTION_RECEIPT_OUTPUT": str(tmp_path / "receipt.json"),
            "INVARLOCK_ACTION_VERIFY_OUTPUT": str(tmp_path / "verify.json"),
            "INVARLOCK_ACTION_HTML_OUTPUT": str(tmp_path / "report.html"),
            "INVARLOCK_ACTION_ARTIFACT_NAME": "review-artifact",
        }
    )

    completed = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "evidence must not contain symbolic links" in completed.stderr
    assert not github_env.exists()


def test_action_rejects_newline_path_injection_before_artifact_upload(
    tmp_path: Path,
) -> None:
    action = _action()
    script = _step(action, "Prepare InvarLock output directories")["run"]
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    verifier_key = tmp_path / "verifier.pem"
    verifier_key.write_text("secret", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    github_env = tmp_path / "github.env"
    injected_output = f"{tmp_path / 'verify.json'}\n{verifier_key}"
    environment = dict(os.environ)
    environment.update(
        {
            "GITHUB_ENV": str(github_env),
            "INVARLOCK_ACTION_PYTHON": sys.executable,
            "INVARLOCK_ACTION_EVIDENCE": str(evidence),
            "INVARLOCK_ACTION_POLICY": str(policy),
            "INVARLOCK_ACTION_VERIFIER_SIGNING_KEY": str(verifier_key),
            "INVARLOCK_ACTION_RECEIPT_OUTPUT": str(tmp_path / "receipt.json"),
            "INVARLOCK_ACTION_VERIFY_OUTPUT": injected_output,
            "INVARLOCK_ACTION_HTML_OUTPUT": str(tmp_path / "report.html"),
            "INVARLOCK_ACTION_ARTIFACT_NAME": "review-artifact",
        }
    )

    completed = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "output path contains unsafe characters" in completed.stderr
    assert not github_env.exists()


@pytest.mark.parametrize(
    "pattern",
    ["*", "?", "[key]", "{key}", "!key", "~", r"key\.pem", "#key"],
)
def test_action_rejects_upload_pattern_syntax_before_artifact_upload(
    tmp_path: Path,
    pattern: str,
) -> None:
    action = _action()
    script = _step(action, "Prepare InvarLock output directories")["run"]
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    verifier_key = tmp_path / "verifier.pem"
    verifier_key.write_text("secret", encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    github_env = tmp_path / "github.env"
    environment = dict(os.environ)
    environment.update(
        {
            "GITHUB_ENV": str(github_env),
            "INVARLOCK_ACTION_PYTHON": sys.executable,
            "INVARLOCK_ACTION_EVIDENCE": str(evidence),
            "INVARLOCK_ACTION_POLICY": str(policy),
            "INVARLOCK_ACTION_VERIFIER_SIGNING_KEY": str(verifier_key),
            "INVARLOCK_ACTION_RECEIPT_OUTPUT": str(tmp_path / "receipt.json"),
            "INVARLOCK_ACTION_VERIFY_OUTPUT": str(tmp_path / pattern),
            "INVARLOCK_ACTION_HTML_OUTPUT": str(tmp_path / "report.html"),
            "INVARLOCK_ACTION_ARTIFACT_NAME": "review-artifact",
        }
    )

    completed = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "output path contains upload-pattern syntax" in completed.stderr
    assert not github_env.exists()


@pytest.mark.parametrize(
    ("status", "fail_on_verify", "expected"),
    [
        ("0", "true", 0),
        ("7", "true", 7),
        ("7", "false", 0),
        ("7", "TRUE", 2),
        ("invalid", "true", 2),
        ("256", "true", 2),
    ],
)
def test_action_enforcement_is_closed_and_validates_control_inputs(
    status: str,
    fail_on_verify: str,
    expected: int,
) -> None:
    action = _action()
    script = _step(action, "Enforce InvarLock verification result")["run"]
    environment = dict(os.environ)
    environment.update(
        {
            "INVARLOCK_VERIFY_EXIT_CODE": status,
            "INVARLOCK_ACTION_FAIL_ON_VERIFY": fail_on_verify,
        }
    )

    completed = subprocess.run(
        ["bash", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected
