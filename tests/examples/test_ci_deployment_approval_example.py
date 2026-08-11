from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/ci/verify_deployment_receipt.py"
WORKFLOW = ROOT / "examples/ci/github-actions/deployment-approval.yml"
GOLDEN = ROOT / "examples/acceptance-handoff/golden"

APPROVAL_INPUTS = {
    "artifact_digests": {
        "baseline": "sha256:1a06e3f2b3fdd505dcdf8b2aa7a8696a18be09881872742274525164567c5f53",
        "subject": "sha256:3028c4d2bd723a42aa88630b212996dde2ac62f6f25374b421a86066c600c930",
    },
    "evidence_signer_fingerprint": "sha256:fe2b99fd9afaa999241faf924364da249f769fd984813db0d42f389f30c65005",
    "format": "invarlock/deployment-approval-inputs-v1",
    "policy_sha256": "sha256:4df8b25e462f8173c7a4b329d24f9d286b0ad59bb0b5441fe700c1038ea4bcc1",
    "runtime_digests": {
        "baseline": "sha256:" + "1" * 64,
        "subject": "sha256:" + "2" * 64,
    },
    "schedule_digest": "sha256:adf6af826b0a72ee32a9d5a156144d716afe93bb280a763a56076f1554e223bf",
    "verifier_fingerprint": "sha256:74a97c1d8fe8d7d58faac074d3a3a9267d8db501d9e4aed77eaeb9ad4efb32ff",
    "verifier_identity": "verifier.example/release-qualification",
}


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
        str(GOLDEN / "evidence"),
        "--policy",
        str(GOLDEN / "evaluated-policy.json"),
        "--receipt",
        str(GOLDEN / "verification.receipt.json"),
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
            evidence_path=GOLDEN / "evidence",
            policy_path=tmp_path,
            receipt_path=GOLDEN / "verification.receipt.json",
        )


def test_deployment_gate_rejects_a_signed_non_authorizing_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
            statement={"verdict": {"ok": False}},
            verifier_fingerprint=APPROVAL_INPUTS["verifier_fingerprint"],
        ),
    )

    with pytest.raises(module.DeploymentApprovalError, match="does not authorize"):
        module.approve(
            approval_inputs_path=inputs,
            evidence_path=GOLDEN / "evidence",
            policy_path=GOLDEN / "evaluated-policy.json",
            receipt_path=GOLDEN / "verification.receipt.json",
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
    assert "actions/download-artifact@" in rendered
    assert "fetch-approved-candidate.sh" in rendered
    assert "Deploy the exact approved candidate" in rendered
