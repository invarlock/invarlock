#!/usr/bin/env python3
"""Run one isolated evidence-signing-to-verifier handoff demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from invarlock.engine import verify_signed_verification_receipt

BASELINE_RUNTIME = "sha256:" + ("1" * 64)
SUBJECT_RUNTIME = "sha256:" + ("2" * 64)
VERIFIER_IDENTITY = "invarlock-verifier/trust-boundary-demo"


def _run(*arguments: str, expect_success: bool) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "invarlock", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    succeeded = completed.returncode == 0
    if succeeded != expect_success:
        outcome = "succeed" if expect_success else "fail closed"
        raise RuntimeError(
            f"expected {' '.join(arguments[:2])} to {outcome}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _generate_key(example_root: Path, destination: Path, role: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(example_root / "generate_keys.py"),
            "--output-dir",
            str(destination),
            "--role",
            role,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)


def _material_anchors(
    baseline_identity: Path,
    subject_identity: Path,
    schedule: Path,
) -> dict[str, str]:
    return {
        "baseline": "sha256:"
        + hashlib.sha256(baseline_identity.read_bytes()).hexdigest(),
        "subject": "sha256:"
        + hashlib.sha256(subject_identity.read_bytes()).hexdigest(),
        "dataset": "sha256:" + hashlib.sha256(schedule.read_bytes()).hexdigest(),
    }


def _copy_evaluation_inputs(example_root: Path, evaluation: Path) -> None:
    for directory in ("inputs", "policy", "import"):
        shutil.copytree(example_root / directory, evaluation / directory)
    for filename in ("request.yaml", "rejected-request.yaml"):
        shutil.copy2(example_root / filename, evaluation / filename)


def _verify_receipt(
    receipt: Path,
    evidence: Path,
    policy: Path,
    evidence_signer_fingerprint: str,
    verifier_fingerprint: str,
    input_anchors: dict[str, str],
    *,
    expected_acceptance: bool,
    expected_policy_verdict: str | None,
) -> None:
    verified = verify_signed_verification_receipt(
        receipt,
        evidence,
        policy_path=policy,
        expected_artifact_digests={
            "baseline": input_anchors["baseline"],
            "subject": input_anchors["subject"],
        },
        expected_schedule_digest=input_anchors["dataset"],
        expected_runtime_digests={
            "baseline": BASELINE_RUNTIME,
            "subject": SUBJECT_RUNTIME,
        },
        expected_pack_signer_fingerprint=evidence_signer_fingerprint,
        expected_verifier_identity=VERIFIER_IDENTITY,
        expected_verifier_fingerprint=verifier_fingerprint,
    )
    if not verified.ok:
        raise RuntimeError(
            "signed receipt verification failed: " + "; ".join(verified.errors)
        )
    if verified.statement is None:
        raise RuntimeError("signed receipt has no authenticated statement")
    verdict = verified.statement["verdict"]
    if not isinstance(verdict, dict):
        raise RuntimeError("signed receipt verdict is malformed")
    if verdict.get("ok") is not expected_acceptance:
        raise RuntimeError("signed receipt acceptance does not match the demo outcome")
    if verdict.get("policy_verdict") != expected_policy_verdict:
        raise RuntimeError(
            "signed receipt policy verdict does not match the demo outcome"
        )


def _verify_command(
    evidence: Path,
    policy: Path,
    evidence_signer_fingerprint: str,
    verifier_key: Path,
    receipt: Path,
    input_anchors: dict[str, str],
    *,
    expect_success: bool,
) -> subprocess.CompletedProcess[str]:
    return _run(
        "verify",
        str(evidence),
        "--policy",
        str(policy),
        "--expected-baseline-artifact",
        input_anchors["baseline"],
        "--expected-subject-artifact",
        input_anchors["subject"],
        "--expected-schedule",
        input_anchors["dataset"],
        "--expected-baseline-runtime",
        BASELINE_RUNTIME,
        "--expected-subject-runtime",
        SUBJECT_RUNTIME,
        "--expected-signer",
        evidence_signer_fingerprint,
        "--receipt",
        str(receipt),
        "--verifier-signing-key",
        str(verifier_key),
        "--verifier-identity",
        VERIFIER_IDENTITY,
        "--json",
        expect_success=expect_success,
    )


def run_demo(example_root: Path, workspace: Path) -> None:
    if workspace.exists() or workspace.is_symlink():
        raise RuntimeError(f"demo workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    evaluation = workspace / "evaluation"
    verifier = workspace / "verifier"
    evaluation.mkdir()
    verifier.mkdir()
    _copy_evaluation_inputs(example_root, evaluation)

    evidence_signer_private = evaluation / "private"
    verifier_private = verifier / "private"
    _generate_key(example_root, evidence_signer_private, "evidence-signer")
    _generate_key(example_root, verifier_private, "verifier")
    evidence_signer_key = evidence_signer_private / "evidence-signer.pem"
    verifier_key = verifier_private / "verifier.pem"
    evidence_signer_fingerprint = (
        (evidence_signer_private / "evidence-signer.fingerprint")
        .read_text(encoding="ascii")
        .strip()
    )
    verifier_fingerprint = (
        (verifier_private / "verifier.fingerprint").read_text(encoding="ascii").strip()
    )

    anchors = verifier / "trusted-inputs"
    anchors.mkdir()
    policy = anchors / "acceptance.json"
    shutil.copy2(example_root / "policy/acceptance.json", policy)
    (anchors / "evidence-signer.fingerprint").write_text(
        evidence_signer_fingerprint + "\n", encoding="ascii"
    )
    (anchors / "runtime-digests.txt").write_text(
        f"baseline={BASELINE_RUNTIME}\nsubject={SUBJECT_RUNTIME}\n",
        encoding="ascii",
    )
    accepted_input_anchors = _material_anchors(
        example_root / "import/baseline/model-artifact.identity.json",
        example_root / "import/subject/model-artifact.identity.json",
        example_root / "inputs/schedule.json",
    )
    rejected_input_anchors = _material_anchors(
        example_root / "import/baseline/model-artifact.identity.json",
        example_root / "import/rejected-subject/model-artifact.identity.json",
        example_root / "inputs/schedule.json",
    )
    (anchors / "input-digests.json").write_text(
        json.dumps(
            {
                "accepted": accepted_input_anchors,
                "rejected": rejected_input_anchors,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )

    _run(
        "evaluate",
        str(evaluation / "request.yaml"),
        "--signing-key",
        str(evidence_signer_key),
        expect_success=True,
    )
    _run(
        "evaluate",
        str(evaluation / "rejected-request.yaml"),
        "--signing-key",
        str(evidence_signer_key),
        expect_success=True,
    )

    submissions = verifier / "submissions"
    submissions.mkdir()
    accepted_evidence = submissions / "accepted-evidence"
    rejected_evidence = submissions / "rejected-evidence"
    # These immutable evidence directories are the only evaluation outputs handed to
    # the verifier. Policy, runtime digests, and signer authorization are provisioned
    # separately under verifier/trusted-inputs.
    shutil.copytree(evaluation / "artifacts/evidence", accepted_evidence)
    shutil.copytree(evaluation / "artifacts/rejected-evidence", rejected_evidence)

    receipts = verifier / "receipts"
    receipts.mkdir()
    accepted_receipt = receipts / "accepted.receipt.json"
    _verify_command(
        accepted_evidence,
        policy,
        evidence_signer_fingerprint,
        verifier_key,
        accepted_receipt,
        accepted_input_anchors,
        expect_success=True,
    )
    _verify_receipt(
        accepted_receipt,
        accepted_evidence,
        policy,
        evidence_signer_fingerprint,
        verifier_fingerprint,
        accepted_input_anchors,
        expected_acceptance=True,
        expected_policy_verdict="pass",
    )

    rejected_receipt = receipts / "policy-rejected.receipt.json"
    _verify_command(
        rejected_evidence,
        policy,
        evidence_signer_fingerprint,
        verifier_key,
        rejected_receipt,
        rejected_input_anchors,
        expect_success=False,
    )
    _verify_receipt(
        rejected_receipt,
        rejected_evidence,
        policy,
        evidence_signer_fingerprint,
        verifier_fingerprint,
        rejected_input_anchors,
        expected_acceptance=False,
        expected_policy_verdict="fail",
    )

    tampered_evidence = submissions / "tampered-evidence"
    shutil.copytree(accepted_evidence, tampered_evidence)
    tampered_report = tampered_evidence / "reports/evaluation.report.json"
    original_mode = stat.S_IMODE(tampered_report.stat().st_mode)
    tampered_report.chmod(original_mode | stat.S_IWUSR)
    tampered_report.write_bytes(tampered_report.read_bytes() + b"\n")
    tampered_report.chmod(original_mode)
    tampered_receipt = receipts / "tampered.receipt.json"
    _verify_command(
        tampered_evidence,
        policy,
        evidence_signer_fingerprint,
        verifier_key,
        tampered_receipt,
        accepted_input_anchors,
        expect_success=False,
    )
    _verify_receipt(
        tampered_receipt,
        tampered_evidence,
        policy,
        evidence_signer_fingerprint,
        verifier_fingerprint,
        accepted_input_anchors,
        expected_acceptance=False,
        expected_policy_verdict=None,
    )

    print(f"PASS accepted evidence and receipt: {accepted_receipt}")
    print(f"PASS authentic policy rejection: {rejected_receipt}")
    print(f"PASS byte-tamper rejection: {tampered_receipt}")
    print(f"Inspect the isolated workspaces under {workspace}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run an isolated evidence-signing/verifier handoff demonstration"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="new destination for evaluation and verifier workspaces",
    )
    args = parser.parse_args()
    run_demo(Path(__file__).resolve().parent, args.workspace.resolve())


if __name__ == "__main__":
    main()
