from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_PATH = REPO_ROOT / ".github/actions/invarlock-report-gate/action.yml"
QUICKSTART_FIXTURE = REPO_ROOT / "examples/acceptance-handoff/golden"


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
        "expected-request-digest",
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
    assert inputs["expected-request-digest"]["required"] is False

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
        "--expected-request-digest",
        "--receipt",
        "--verifier-signing-key",
        "--verifier-identity",
        "--json",
    ):
        assert option in verify

    verify_step = _step(action, "Verify InvarLock evidence")
    assert verify_step["env"]["INVARLOCK_ACTION_EXPECTED_REQUEST_DIGEST"] == (
        "${{ inputs.expected-request-digest }}"
    )
    assert '[[ -n "$INVARLOCK_ACTION_EXPECTED_REQUEST_DIGEST" ]]' in verify

    assert "-m invarlock report" in report
    assert "--html" in report
    assert "--explain" in report
    assert "report html" not in report
    assert _step(action, "Render InvarLock HTML report")["if"] == (
        "${{ always() && env.INVARLOCK_ACTION_LAYOUT_SAFE == 'true' && "
        "env.INVARLOCK_VERIFY_EXIT_CODE == '0' }}"
    )


def test_action_preserves_review_outputs_without_uploading_the_verifier_key() -> None:
    action = _action()
    upload = _step(action, "Upload InvarLock evidence")
    uploaded_paths = upload["with"]["path"]

    assert upload["if"] == (
        "${{ always() && env.INVARLOCK_ACTION_LAYOUT_SAFE == 'true' && "
        "env.INVARLOCK_VERIFY_EXIT_CODE == '0' }}"
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

    failure_upload = _step(action, "Upload InvarLock verification failure")
    assert failure_upload["if"] == (
        "${{ always() && env.INVARLOCK_ACTION_LAYOUT_SAFE == 'true' && "
        "env.INVARLOCK_VERIFY_EXIT_CODE != '0' }}"
    )
    assert failure_upload["with"]["path"] == "${{ inputs.verify-output }}"
    assert failure_upload["with"]["name"].endswith("-verification-failure")
    for private_input in (
        "inputs.evidence",
        "inputs.receipt-output",
        "inputs.html-output",
        "inputs.verifier-signing-key",
    ):
        assert private_input not in failure_upload["with"]["path"]

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


def _consumer_environment(root: Path) -> dict[str, str]:
    anchors = json.loads(
        (root / "review/technical-anchors.json").read_text(encoding="utf-8")
    )
    key_path = root / "review/verifier.private.pem"
    key_path.write_bytes(
        ed25519.Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "GITHUB_ENV": str(root / "github.env"),
            "INVARLOCK_ACTION_PYTHON": sys.executable,
            "INVARLOCK_ACTION_EVIDENCE": "incoming/evidence",
            "INVARLOCK_ACTION_POLICY": "review/evaluated-policy.json",
            "INVARLOCK_ACTION_EXPECTED_BASELINE_ARTIFACT": anchors["artifact_digests"][
                "baseline"
            ],
            "INVARLOCK_ACTION_EXPECTED_SUBJECT_ARTIFACT": anchors["artifact_digests"][
                "subject"
            ],
            "INVARLOCK_ACTION_EXPECTED_SCHEDULE": anchors["schedule_digest"],
            "INVARLOCK_ACTION_EXPECTED_BASELINE_RUNTIME": anchors["runtime_digests"][
                "baseline"
            ],
            "INVARLOCK_ACTION_EXPECTED_SUBJECT_RUNTIME": anchors["runtime_digests"][
                "subject"
            ],
            "INVARLOCK_ACTION_EXPECTED_SIGNER": anchors["evidence_signer_fingerprint"],
            "INVARLOCK_ACTION_EXPECTED_REQUEST_DIGEST": "",
            "INVARLOCK_ACTION_VERIFIER_SIGNING_KEY": ("review/verifier.private.pem"),
            "INVARLOCK_ACTION_VERIFIER_IDENTITY": "consumer-verifier",
            "INVARLOCK_ACTION_RECEIPT_OUTPUT": (
                "reports/invarlock/verification.receipt.json"
            ),
            "INVARLOCK_ACTION_VERIFY_OUTPUT": (
                "reports/invarlock/verification.result.json"
            ),
            "INVARLOCK_ACTION_HTML_OUTPUT": "reports/invarlock/evidence.html",
            "INVARLOCK_ACTION_ARTIFACT_NAME": "candidate-invarlock-evidence",
            "INVARLOCK_ACTION_FAIL_ON_VERIFY": "true",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
    )
    return environment


def _load_github_environment(environment: dict[str, str]) -> None:
    for line in (
        Path(environment["GITHUB_ENV"]).read_text(encoding="utf-8").splitlines()
    ):
        name, value = line.split("=", 1)
        environment[name] = value


@pytest.mark.parametrize("tampered", [False, True])
def test_action_runs_from_an_isolated_consumer_and_rejects_tampering(
    tmp_path: Path, tampered: bool
) -> None:
    action = _action()
    consumer = tmp_path / "consumer"
    evidence = consumer / "incoming/evidence"
    review = consumer / "review"
    review.mkdir(parents=True)
    shutil.copytree(QUICKSTART_FIXTURE / "evidence", evidence)
    shutil.copy2(
        QUICKSTART_FIXTURE / "evaluated-policy.json",
        review / "evaluated-policy.json",
    )
    shutil.copy2(
        QUICKSTART_FIXTURE / "technical-anchors.json",
        review / "technical-anchors.json",
    )
    if tampered:
        report = evidence / "reports/evaluation.report.json"
        report.write_bytes(report.read_bytes() + b"\n")
    environment = _consumer_environment(consumer)

    for name in (
        "Prepare InvarLock output directories",
        "Verify InvarLock evidence",
    ):
        completed = subprocess.run(
            ["bash", "-c", _step(action, name)["run"]],
            cwd=consumer,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        _load_github_environment(environment)

    result = json.loads(
        (consumer / "reports/invarlock/verification.result.json").read_bytes()
    )
    enforce = subprocess.run(
        [
            "bash",
            "-c",
            _step(action, "Enforce InvarLock verification result")["run"],
        ],
        cwd=consumer,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if tampered:
        assert result["ok"] is False
        assert environment["INVARLOCK_VERIFY_EXIT_CODE"] != "0"
        assert enforce.returncode != 0
        assert not (consumer / "reports/invarlock/evidence.html").exists()
    else:
        rendered = subprocess.run(
            ["bash", "-c", _step(action, "Render InvarLock HTML report")["run"]],
            cwd=consumer,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert rendered.returncode == 0, rendered.stderr or rendered.stdout
        assert result["ok"] is True
        assert result["policy_verdict"] == "pass"
        assert environment["INVARLOCK_VERIFY_EXIT_CODE"] == "0"
        assert enforce.returncode == 0
        assert (consumer / "reports/invarlock/verification.receipt.json").is_file()
        assert (consumer / "reports/invarlock/evidence.html").is_file()
