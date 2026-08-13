from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONSUMER = ROOT / "examples/ci/standalone-consumer"
SCRIPT = CONSUMER / "review/verify_deployment_receipt.py"
WORKFLOW = CONSUMER / ".github/workflows/deployment-approval.yml"
APPROVAL_INPUTS_PATH = CONSUMER / "review/inspect-ai-deployment-approval-inputs.json"
POLICY = CONSUMER / "review/policy/acceptance.json"
TRANSACTION = ROOT / "examples/evaluator-qualification/signed-transactions/inspect-ai"
APPROVAL_INPUTS = json.loads(APPROVAL_INPUTS_PATH.read_bytes())


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("deployment_receipt_example", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(
    inputs: Path, *, output: Path | None = None
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--approval-inputs",
        str(inputs),
        "--evidence",
        str(TRANSACTION / "evidence"),
        "--policy",
        str(POLICY),
        "--receipt",
        str(TRANSACTION / "verification.receipt.json"),
    ]
    if output is not None:
        command.extend(("--output", str(output)))
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_deployment_inputs_are_independent_anchors_for_retained_inspect_transaction() -> (
    None
):
    transaction = json.loads((TRANSACTION / "transaction.json").read_bytes())
    receipt = json.loads((TRANSACTION / "verification.receipt.json").read_bytes())[
        "statement"
    ]
    verification = transaction["verification"]
    policy_sha256 = "sha256:" + hashlib.sha256(POLICY.read_bytes()).hexdigest()

    assert APPROVAL_INPUTS["artifact_digests"] == verification["artifact_digests"]
    assert APPROVAL_INPUTS["runtime_digests"] == verification["runtime_digests"]
    assert APPROVAL_INPUTS["schedule_digest"] == verification["schedule_digest"]
    assert (
        APPROVAL_INPUTS["evidence_signer_fingerprint"]
        == verification["evidence_signer_fingerprint"]
    )
    assert APPROVAL_INPUTS["verifier_identity"] == verification["verifier_identity"]
    assert (
        APPROVAL_INPUTS["verifier_fingerprint"] == verification["verifier_fingerprint"]
    )
    assert (
        APPROVAL_INPUTS["trust_profile_digest"] == verification["trust_profile_digest"]
    )
    assert APPROVAL_INPUTS["policy_sha256"] == policy_sha256
    assert policy_sha256 == receipt["anchors"]["policy_digest"]


def test_deployment_receipt_gate_accepts_only_the_independent_signed_result(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "approval-inputs.json"
    inputs.write_text(json.dumps(APPROVAL_INPUTS), encoding="utf-8")

    accepted = _run(inputs)
    assert accepted.returncode == 0, accepted.stderr
    result = json.loads(accepted.stdout)
    assert result["accepted"] is True
    assert result["artifact_digests"] == APPROVAL_INPUTS["artifact_digests"]
    assert result["runtime_digests"] == APPROVAL_INPUTS["runtime_digests"]
    assert result["schedule_digest"] == APPROVAL_INPUTS["schedule_digest"]
    assert result["trust_profile_digest"] == APPROVAL_INPUTS["trust_profile_digest"]
    assert result["verifier_identity"] == APPROVAL_INPUTS["verifier_identity"]
    assert result["pack_manifest_digest"].startswith("sha256:")

    changed = dict(APPROVAL_INPUTS)
    changed["verifier_fingerprint"] = "sha256:" + "0" * 64
    inputs.write_text(json.dumps(changed), encoding="utf-8")
    rejected = _run(inputs)
    assert rejected.returncode != 0
    assert "receipt verifier key does not match caller expectation" in rejected.stderr


def test_deployment_receipt_gate_rejects_changed_policy_anchor(tmp_path: Path) -> None:
    inputs = tmp_path / "approval-inputs.json"
    changed = dict(APPROVAL_INPUTS)
    changed["policy_sha256"] = "sha256:" + "0" * 64
    inputs.write_text(json.dumps(changed), encoding="utf-8")

    rejected = _run(inputs)
    assert rejected.returncode != 0
    assert "policy digest does not match approval inputs" in rejected.stderr


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": True}, "fields are invalid"),
        ({"format": "wrong"}, "format is invalid"),
        ({"verifier_identity": " "}, "identity must be non-empty"),
        ({"schedule_digest": "wrong"}, "must be a sha256 digest"),
        ({"trust_profile_digest": "wrong"}, "must be a sha256 digest"),
        ({"artifact_digests": {}}, "exactly baseline and subject"),
        (
            {
                "runtime_digests": {
                    "baseline": "wrong",
                    "subject": "sha256:" + "2" * 64,
                }
            },
            "must be a sha256 digest",
        ),
    ],
)
def test_deployment_inputs_reject_malformed_trust_anchors(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    module = _module()
    inputs = tmp_path / "approval-inputs.json"
    value = dict(APPROVAL_INPUTS)
    value.update(change)
    inputs.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(module.DeploymentApprovalError, match=message):
        module.load_approval_inputs(inputs)


def test_deployment_inputs_reject_invalid_json_and_unreadable_policy(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = tmp_path / "approval-inputs.json"
    inputs.write_text('{"format":"one","format":"two"}', encoding="utf-8")
    with pytest.raises(module.DeploymentApprovalError, match="duplicate"):
        module.load_approval_inputs(inputs)

    inputs.write_text(json.dumps(APPROVAL_INPUTS), encoding="utf-8")
    with pytest.raises(module.DeploymentApprovalError, match="regular file"):
        module.approve(
            approval_inputs_path=inputs,
            evidence_path=TRANSACTION / "evidence",
            policy_path=tmp_path,
            receipt_path=TRANSACTION / "verification.receipt.json",
        )


def test_deployment_inputs_allow_explicit_anchor_receipts_without_trust_profile(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = tmp_path / "approval-inputs.json"
    value = dict(APPROVAL_INPUTS)
    value["trust_profile_digest"] = None
    inputs.write_text(json.dumps(value), encoding="utf-8")

    assert module.load_approval_inputs(inputs)["trust_profile_digest"] is None


@pytest.mark.parametrize(
    "verdict",
    [
        {"ok": False, "policy_verdict": "fail"},
        {"ok": True, "policy_verdict": None},
    ],
)
def test_deployment_gate_rejects_a_signed_non_authorizing_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verdict: dict[str, object],
) -> None:
    module = _module()
    inputs = tmp_path / "approval-inputs.json"
    inputs.write_text(json.dumps(APPROVAL_INPUTS), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "verify_signed_verification_receipt",
        lambda *_args, **_kwargs: SimpleNamespace(
            errors=(),
            ok=True,
            statement={"verdict": verdict},
            verifier_fingerprint=APPROVAL_INPUTS["verifier_fingerprint"],
        ),
    )

    with pytest.raises(module.DeploymentApprovalError, match="does not authorize"):
        module.approve(
            approval_inputs_path=inputs,
            evidence_path=TRANSACTION / "evidence",
            policy_path=POLICY,
            receipt_path=TRANSACTION / "verification.receipt.json",
        )


def test_deployment_output_is_canonical_and_no_clobber(tmp_path: Path) -> None:
    inputs = tmp_path / "approval-inputs.json"
    output = tmp_path / "reports/deployment-approval.json"
    inputs.write_text(json.dumps(APPROVAL_INPUTS), encoding="utf-8")

    accepted = _run(inputs, output=output)
    assert accepted.returncode == 0, accepted.stderr
    assert output.read_text(encoding="utf-8") == accepted.stdout

    repeated = _run(inputs, output=output)
    assert repeated.returncode == 2
    assert "output already exists" in repeated.stderr


def test_consumer_fixture_runs_from_an_independent_copy(tmp_path: Path) -> None:
    consumer = tmp_path / "consumer"
    shutil.copytree(CONSUMER, consumer)
    incoming = consumer / "incoming/evidence"
    incoming.parent.mkdir()
    shutil.copytree(TRANSACTION / "evidence", incoming)
    receipt = consumer / "incoming/verification.receipt.json"
    shutil.copy2(TRANSACTION / "verification.receipt.json", receipt)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"})

    completed = subprocess.run(
        [
            sys.executable,
            "review/verify_deployment_receipt.py",
            "--approval-inputs",
            "review/inspect-ai-deployment-approval-inputs.json",
            "--evidence",
            "incoming/evidence",
            "--policy",
            "review/policy/acceptance.json",
            "--receipt",
            "incoming/verification.receipt.json",
        ],
        cwd=consumer,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["accepted"] is True


def test_deployment_workflow_has_separate_verification_and_protected_deploy_jobs() -> (
    None
):
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    verify = workflow["jobs"]["verify-evidence"]
    deploy = workflow["jobs"]["deploy-candidate"]

    assert verify["environment"] == "release-review"
    assert deploy["needs"] == "verify-evidence"
    assert deploy["environment"] == "production"
    rendered = WORKFLOW.read_text(encoding="utf-8")
    assert "verify_deployment_receipt.py" in rendered
    assert "review/verify_deployment_receipt.py" in rendered
    assert "review/policy/acceptance.json" in rendered
    assert "examples/ci/" not in rendered
    assert "actions/download-artifact@" in rendered
    assert "fetch-approved-candidate.sh" in rendered
    assert "Deploy the exact approved candidate" in rendered
