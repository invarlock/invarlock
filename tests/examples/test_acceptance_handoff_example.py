from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest

from invarlock.acceptance_attestation import verify_acceptance_attestation
from invarlock.evidence_pack_verification import verify_comparison_evidence
from invarlock.evidence_receipt import verify_signed_verification_receipt
from invarlock.public_contracts import load_recipient_acceptance_policy_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples/run_acceptance_handoff.py"
GOLDEN = REPO_ROOT / "examples/acceptance-handoff/golden"
REFERENCE_POLICY = (
    REPO_ROOT / "examples/acceptance-handoff/recipient-policy.example.json"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("acceptance_handoff", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_recipient_policy_is_schema_valid() -> None:
    policy = json.loads(REFERENCE_POLICY.read_bytes())

    jsonschema.Draft202012Validator(load_recipient_acceptance_policy_schema()).validate(
        policy
    )
    assert policy["expected_predicate_type"].endswith("/acceptance/v1")
    assert policy["required_technical_verdict"] == "pass"
    assert policy["trusted_signers"][0]["status"] == "active"


def test_committed_golden_package_verifies_end_to_end() -> None:
    anchors = json.loads((GOLDEN / "technical-anchors.json").read_bytes())
    recipient_policy = json.loads((GOLDEN / "recipient-policy.json").read_bytes())
    evidence_result = verify_comparison_evidence(
        GOLDEN / "evidence",
        policy_path=GOLDEN / "evaluated-policy.json",
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_signer_fingerprint=anchors["evidence_signer_fingerprint"],
    )
    receipt_result = verify_signed_verification_receipt(
        GOLDEN / "verification.receipt.json",
        GOLDEN / "evidence",
        policy_path=GOLDEN / "evaluated-policy.json",
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_pack_signer_fingerprint=anchors["evidence_signer_fingerprint"],
        expected_verifier_identity=anchors["verifier_identity"],
        expected_verifier_fingerprint=anchors["verifier_fingerprint"],
    )
    acceptance = verify_acceptance_attestation(
        GOLDEN / "acceptance.dsse.json",
        trusted_public_keys={
            anchors["envelope_signer_fingerprint"]: (GOLDEN / "producer.public.pem")
        },
        recipient_policy=recipient_policy,
        subject_artifact_path=GOLDEN / "artifact",
        now=datetime(2026, 7, 25, 12, 5, tzinfo=UTC),
    )

    assert evidence_result.payload["ok"] is True
    assert receipt_result.ok is True
    assert acceptance.accepted is True
    assert acceptance.envelope_authenticated is True
    assert acceptance.receipt_authenticated is True
    assert acceptance.subject_bound is True


def test_committed_golden_package_is_exact_generator_output(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "golden"
    _module().write_golden(generated)

    committed_files = {
        path.relative_to(GOLDEN): path.read_bytes()
        for path in GOLDEN.rglob("*")
        if path.is_file()
    }
    generated_files = {
        path.relative_to(generated): path.read_bytes()
        for path in generated.rglob("*")
        if path.is_file()
    }
    assert generated_files == committed_files


def test_offline_producer_recipient_handoff_covers_current_policy_failures(
    tmp_path: Path,
) -> None:
    module = _module()
    workspace = tmp_path / "handoff"

    module.run_handoff(workspace)

    results = json.loads((workspace / "results.json").read_bytes())
    assert results == {
        "format": "invarlock/producer-recipient-handoff-v1",
        "historical_technical_verification": True,
        "scenarios": {
            "accepted": True,
            "contradictory_receipt_envelope_rejected": True,
            "revoked_signer_rejected": True,
            "stale_evidence_rejected": True,
            "stricter_policy_rejected": True,
            "tampered_envelope_rejected": True,
            "tampered_evidence_rejected": True,
            "unknown_signer_rejected": True,
            "wrong_artifact_rejected": True,
        },
    }
    assert (workspace / "producer/artifacts/subject/model.safetensors").is_file()
    assert (workspace / "producer/evidence/manifest.json").is_file()
    assert (workspace / "producer/verification.receipt.json").is_file()
    assert (workspace / "producer/acceptance.dsse.json").is_file()
    assert (workspace / "recipient/policy.json").is_file()
    assert (workspace / "recipient/trust/producer.public.pem").is_file()


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_handoff_refuses_to_reuse_workspace(tmp_path: Path, kind: str) -> None:
    module = _module()
    workspace = tmp_path / "handoff"
    if kind == "directory":
        workspace.mkdir()
    else:
        workspace.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(RuntimeError, match="must be new"):
        module.run_handoff(workspace)
