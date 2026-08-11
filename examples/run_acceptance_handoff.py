#!/usr/bin/env python3
"""Generate and verify one service-free acceptance handoff."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import stat
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from examples.regenerate_fixtures import (
    BASELINE_RUNTIME,
    SUBJECT_RUNTIME,
    regenerate,
)
from invarlock.acceptance_attestation import (
    ACCEPTANCE_PREDICATE_TYPE,
    DSSE_PAYLOAD_TYPE,
    RECIPIENT_POLICY_FORMAT,
    verify_acceptance_attestation,
    write_acceptance_attestation,
)
from invarlock.core.checkpoint_identity import checkpoint_tree_sha256
from invarlock.core.evaluation_request import load_evaluation_request
from invarlock.core.registry import CoreRegistry
from invarlock.core.runtime_provider import HFSnapshotArtifactIdentity
from invarlock.evaluation_transaction import evaluate_request_file
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_pack_verification import verify_comparison_evidence
from invarlock.evidence_receipt import verify_signed_verification_receipt
from invarlock.evidence_verification import verify_evidence

ROOT = Path(__file__).resolve().parent
POLICY_SOURCE = ROOT / "policy/acceptance.json"
GOLDEN_ROOT = ROOT / "acceptance-handoff/golden"
ISSUED_AT = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
VERIFIER_IDENTITY = "verifier.example/release-qualification"
ENVELOPE_IDENTITY = "envelope-signer.example/release-assurance"
POLICY_IDENTITY = "evaluation.example/policies/release-regression-v1"
HANDOFF_FORMAT = "invarlock/acceptance-handoff-v1"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_key(
    private_path: Path,
    public_path: Path,
    *,
    seed: int,
) -> tuple[Path, Path, str]:
    key = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes((seed + offset) % 256 for offset in range(32))
    )
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    public_path.chmod(0o644)
    return private_path, public_path, public_key_fingerprint(key.public_key())


def _prepare_artifact(path: Path, *, role: str) -> HFSnapshotArtifactIdentity:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"architectures": ["FixtureModel"], "model_type": "fixture"}) + "\n",
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(
        f"invarlock-{role}-artifact-v1\n".encode("ascii")
    )
    tokenizer = b'{"model":{"type":"WordLevel","vocab":{"fixture":0}}}\n'
    (path / "tokenizer.json").write_bytes(tokenizer)
    return HFSnapshotArtifactIdentity(
        model_id=f"artifact.example/{role}",
        immutable_revision=None,
        checkpoint_tree_sha256=checkpoint_tree_sha256(path).removeprefix("sha256:"),
        tokenizer_metadata_sha256=hashlib.sha256(tokenizer).hexdigest(),
    )


def _side_request(identity: HFSnapshotArtifactIdentity) -> dict[str, object]:
    return {
        "artifact": {
            "model_id": identity.model_id,
            "locator": f"artifact://{identity.model_id}",
        },
        "runtime": {
            "provider": "hf_transformers",
            "settings": {
                "batch_size": 1,
                "checkpoint_tree_sha256": identity.checkpoint_tree_sha256,
                "context_length": 128,
                "max_output_tokens": 16,
                "offline": True,
                "seed": 0,
                "timeout_seconds": 30,
                "tokenizer_metadata_sha256": identity.tokenizer_metadata_sha256,
            },
        },
    }


def _import_side(role: str) -> dict[str, str]:
    return {
        "identity": f"import/{role}/model-artifact.identity.json",
        "receipt": f"import/{role}/runtime-provider.receipt.json",
        "observation": f"import/{role}/runtime-scoring.observation.json",
        "run_report": f"import/{role}/report.json",
        "runtime_manifest": f"import/{role}/runtime.manifest.json",
        "runtime_config": f"import/{role}/run.yaml",
    }


def _write_request(
    handoff: Path,
    identities: dict[str, HFSnapshotArtifactIdentity],
) -> Path:
    request = {
        "format_version": "invarlock/evaluation-request-v1",
        "comparison": {
            "baseline": _side_request(identities["baseline"]),
            "subject": _side_request(identities["subject"]),
            "dataset": "inputs/schedule.json",
            "policy": "policy/acceptance.json",
            "task": "text_causal",
            "metric": "exact_match",
        },
        "execution": {
            "mode": "import",
            "records": "import/paired-records.json",
            "schedule": "inputs/schedule.json",
            "baseline": _import_side("baseline"),
            "subject": _import_side("subject"),
        },
        "output": {"evidence": "evidence"},
    }
    path = handoff / "request.yaml"
    path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")
    return path


def _recipient_policy(
    envelope_fingerprint: str,
    receipt_verifier_fingerprint: str,
    *,
    status: str = "active",
    receipt_verifier_status: str = "active",
    max_envelope_age_seconds: int = 86400,
    max_evidence_age_seconds: int | None = None,
    versions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "format": RECIPIENT_POLICY_FORMAT,
        "expected_predicate_type": ACCEPTANCE_PREDICATE_TYPE,
        "trusted_signers": [
            {
                "identity": ENVELOPE_IDENTITY,
                "fingerprint": envelope_fingerprint,
                "status": status,
            }
        ],
        "trusted_receipt_verifiers": [
            {
                "identity": VERIFIER_IDENTITY,
                "fingerprint": receipt_verifier_fingerprint,
                "status": receipt_verifier_status,
            }
        ],
        "freshness": {
            "max_envelope_age_seconds": max_envelope_age_seconds,
            "max_evidence_age_seconds": max_evidence_age_seconds,
            "clock_skew_seconds": 0,
        },
        "allowed_contract_versions": versions or ["0.13.0"],
        "required_technical_verdict": "pass",
        "allow_countersigned_receipts": True,
    }


def _anchors(handoff: Path, evidence_signer: str) -> dict[str, Any]:
    return {
        "artifact_digests": {
            role: _digest(handoff / f"import/{role}/model-artifact.identity.json")
            for role in ("baseline", "subject")
        },
        "schedule_digest": _digest(handoff / "inputs/schedule.json"),
        "runtime_digests": {
            "baseline": BASELINE_RUNTIME,
            "subject": SUBJECT_RUNTIME,
        },
        "evidence_signer_fingerprint": evidence_signer,
        "verifier_identity": VERIFIER_IDENTITY,
    }


def _verify_historical_receipt(
    handoff: Path,
    anchors: dict[str, Any],
    *,
    verifier_fingerprint: str,
) -> bool:
    verified = verify_signed_verification_receipt(
        handoff / "verification.receipt.json",
        handoff / "evidence",
        policy_path=handoff / "policy/acceptance.json",
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_pack_signer_fingerprint=anchors["evidence_signer_fingerprint"],
        expected_verifier_identity=anchors["verifier_identity"],
        expected_verifier_fingerprint=verifier_fingerprint,
    )
    return bool(
        verified.ok
        and verified.statement is not None
        and verified.statement["verdict"]["policy_verdict"] == "pass"
    )


def _decision(
    envelope: Path,
    public_key: Path,
    fingerprint: str,
    policy: dict[str, Any],
    subject_artifact: Path,
    *,
    now: datetime,
) -> bool:
    return verify_acceptance_attestation(
        envelope,
        trusted_public_keys={fingerprint: public_key},
        recipient_policy=policy,
        subject_artifact_path=subject_artifact,
        now=now,
    ).accepted


def _tampered_envelope(source: Path, destination: Path) -> None:
    envelope = json.loads(source.read_bytes())
    envelope["payload"] = envelope["payload"][:-2] + "AA"
    destination.write_bytes(_canonical(envelope))


def _contradictory_envelope(
    source: Path,
    destination: Path,
    private_key: Path,
) -> None:
    envelope = json.loads(source.read_bytes())
    payload = base64.b64decode(envelope["payload"], validate=True)
    statement = json.loads(payload)
    statement["predicate"]["technical_verdict"]["policy_verdict"] = "fail"
    changed_payload = _canonical(statement)
    payload_type = DSSE_PAYLOAD_TYPE.encode("utf-8")
    pae = (
        b"DSSEv1 "
        + str(len(payload_type)).encode("ascii")
        + b" "
        + payload_type
        + b" "
        + str(len(changed_payload)).encode("ascii")
        + b" "
        + changed_payload
    )
    key = serialization.load_pem_private_key(private_key.read_bytes(), password=None)
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise RuntimeError("demo envelope key is not Ed25519")
    envelope["payload"] = base64.b64encode(changed_payload).decode("ascii")
    envelope["signatures"][0]["sig"] = base64.b64encode(key.sign(pae)).decode("ascii")
    destination.write_bytes(_canonical(envelope))


def _tampered_evidence_rejected(
    handoff: Path,
    recipient: Path,
    anchors: dict[str, Any],
) -> bool:
    tampered = recipient / "tampered-evidence"
    shutil.copytree(handoff / "evidence", tampered)
    report = tampered / "reports/evaluation.report.json"
    report.chmod(stat.S_IMODE(report.stat().st_mode) | stat.S_IWUSR)
    report.write_bytes(report.read_bytes() + b"\n")
    result = verify_comparison_evidence(
        tampered,
        policy_path=handoff / "policy/acceptance.json",
        expected_artifact_digests=anchors["artifact_digests"],
        expected_schedule_digest=anchors["schedule_digest"],
        expected_runtime_digests=anchors["runtime_digests"],
        expected_signer_fingerprint=anchors["evidence_signer_fingerprint"],
    )
    return not bool(result.payload.get("ok"))


def run_handoff(workspace: Path) -> None:
    """Run the deterministic offline acceptance transaction."""

    if workspace.exists() or workspace.is_symlink():
        raise RuntimeError("handoff workspace must be new")
    handoff = workspace / "handoff"
    recipient = workspace / "recipient"
    handoff.mkdir(parents=True)
    recipient.mkdir()
    (handoff / "policy").mkdir()
    (handoff / "inputs").mkdir()
    (handoff / "trusted-inputs").mkdir()
    shutil.copy2(POLICY_SOURCE, handoff / "policy/acceptance.json")

    identities = {
        role: _prepare_artifact(
            handoff / f"artifacts/{role}",
            role=role,
        )
        for role in ("baseline", "subject")
    }
    regenerate(handoff, identities=identities)
    request_path = _write_request(handoff, identities)

    evidence_key, _evidence_public, evidence_fingerprint = _write_key(
        handoff / "private/evidence.private.pem",
        handoff / "private/evidence.public.pem",
        seed=11,
    )
    verifier_key, verifier_public, verifier_fingerprint = _write_key(
        handoff / "private/verifier.private.pem",
        handoff / "private/verifier.public.pem",
        seed=53,
    )
    envelope_key, envelope_public, envelope_fingerprint = _write_key(
        handoff / "private/envelope.private.pem",
        recipient / "trust/envelope-signer.public.pem",
        seed=97,
    )
    loaded_request = load_evaluation_request(
        request_path,
        provider_resolver=CoreRegistry().get_runtime_provider,
    )
    evaluation = evaluate_request_file(
        loaded_request,
        signing_key_path=evidence_key,
        registry=CoreRegistry(),
    )
    if evaluation.evidence_path != (handoff / "evidence").resolve():
        raise RuntimeError("evaluation published to an unexpected destination")
    anchors = _anchors(handoff, evidence_fingerprint)
    verify_evidence(
        handoff / "evidence",
        policy_path=handoff / "policy/acceptance.json",
        expected_baseline_artifact=anchors["artifact_digests"]["baseline"],
        expected_subject_artifact=anchors["artifact_digests"]["subject"],
        expected_schedule=anchors["schedule_digest"],
        expected_baseline_runtime=BASELINE_RUNTIME,
        expected_subject_runtime=SUBJECT_RUNTIME,
        expected_signer=evidence_fingerprint,
        receipt_path=handoff / "verification.receipt.json",
        verifier_signing_key_path=verifier_key,
        verifier_identity=VERIFIER_IDENTITY,
    )
    historical_verified = _verify_historical_receipt(
        handoff,
        anchors,
        verifier_fingerprint=verifier_fingerprint,
    )
    write_acceptance_attestation(
        handoff / "verification.receipt.json",
        handoff / "evidence",
        handoff / "acceptance.dsse.json",
        signing_key_path=envelope_key,
        signer_identity=ENVELOPE_IDENTITY,
        policy_identity=POLICY_IDENTITY,
        issued_at=ISSUED_AT,
        evaluation_completed_at=ISSUED_AT - timedelta(minutes=5),
    )
    policy = _recipient_policy(envelope_fingerprint, verifier_fingerprint)
    (recipient / "policy.json").write_bytes(_canonical(policy))
    anchors["verifier_fingerprint"] = verifier_fingerprint
    anchors["envelope_signer_fingerprint"] = envelope_fingerprint
    anchors["envelope_signer_identity"] = ENVELOPE_IDENTITY
    (recipient / "trust/technical-anchors.json").write_bytes(_canonical(anchors))
    shutil.copy2(verifier_public, recipient / "trust/verifier.public.pem")

    subject_artifact = handoff / "artifacts/subject"
    envelope = handoff / "acceptance.dsse.json"
    scenarios: dict[str, bool] = {}
    scenarios["accepted"] = _decision(
        envelope,
        envelope_public,
        envelope_fingerprint,
        policy,
        subject_artifact,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    scenarios["stricter_policy_rejected"] = not _decision(
        envelope,
        envelope_public,
        envelope_fingerprint,
        _recipient_policy(
            envelope_fingerprint,
            verifier_fingerprint,
            versions=["0.14.0"],
        ),
        subject_artifact,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    wrong_artifact = recipient / "wrong-subject"
    shutil.copytree(subject_artifact, wrong_artifact)
    (wrong_artifact / "model.safetensors").write_bytes(b"wrong artifact\n")
    scenarios["wrong_artifact_rejected"] = not _decision(
        envelope,
        envelope_public,
        envelope_fingerprint,
        policy,
        wrong_artifact,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    scenarios["tampered_evidence_rejected"] = _tampered_evidence_rejected(
        handoff,
        recipient,
        anchors,
    )
    tampered_envelope = recipient / "tampered.dsse.json"
    _tampered_envelope(envelope, tampered_envelope)
    scenarios["tampered_envelope_rejected"] = not _decision(
        tampered_envelope,
        envelope_public,
        envelope_fingerprint,
        policy,
        subject_artifact,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    unknown_policy = _recipient_policy(
        "sha256:" + "9" * 64,
        verifier_fingerprint,
    )
    scenarios["unknown_signer_rejected"] = not _decision(
        envelope,
        envelope_public,
        envelope_fingerprint,
        unknown_policy,
        subject_artifact,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    scenarios["revoked_signer_rejected"] = not _decision(
        envelope,
        envelope_public,
        envelope_fingerprint,
        _recipient_policy(
            envelope_fingerprint,
            verifier_fingerprint,
            status="revoked",
        ),
        subject_artifact,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    scenarios["stale_envelope_rejected"] = not _decision(
        envelope,
        envelope_public,
        envelope_fingerprint,
        _recipient_policy(
            envelope_fingerprint,
            verifier_fingerprint,
            max_envelope_age_seconds=60,
        ),
        subject_artifact,
        now=ISSUED_AT + timedelta(seconds=61),
    )
    scenarios["unknown_receipt_verifier_rejected"] = not _decision(
        envelope,
        envelope_public,
        envelope_fingerprint,
        _recipient_policy(
            envelope_fingerprint,
            "sha256:" + "8" * 64,
        ),
        subject_artifact,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    scenarios["missing_evidence_timestamp_rejected"] = not _decision(
        envelope,
        envelope_public,
        envelope_fingerprint,
        _recipient_policy(
            envelope_fingerprint,
            verifier_fingerprint,
            max_evidence_age_seconds=60,
        ),
        subject_artifact,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    contradictory = recipient / "contradictory.dsse.json"
    _contradictory_envelope(envelope, contradictory, envelope_key)
    scenarios["contradictory_receipt_envelope_rejected"] = not _decision(
        contradictory,
        envelope_public,
        envelope_fingerprint,
        policy,
        subject_artifact,
        now=ISSUED_AT + timedelta(minutes=5),
    )
    results = {
        "format": HANDOFF_FORMAT,
        "historical_technical_verification": historical_verified,
        "scenarios": scenarios,
    }
    if not historical_verified or not all(scenarios.values()):
        raise RuntimeError("acceptance handoff did not satisfy every scenario")
    (workspace / "results.json").write_bytes(_canonical(results))


def _copy_golden(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("golden package destination must be new")
    destination.mkdir(parents=True)
    selected = {
        "artifact": source / "handoff/artifacts/subject",
        "evidence": source / "handoff/evidence",
        "verification.receipt.json": source / "handoff/verification.receipt.json",
        "acceptance.dsse.json": source / "handoff/acceptance.dsse.json",
        "evaluated-policy.json": source / "handoff/policy/acceptance.json",
        "recipient-policy.json": source / "recipient/policy.json",
        "envelope-signer.public.pem": (
            source / "recipient/trust/envelope-signer.public.pem"
        ),
        "verifier.public.pem": source / "recipient/trust/verifier.public.pem",
        "technical-anchors.json": (source / "recipient/trust/technical-anchors.json"),
        "results.json": source / "results.json",
    }
    for name, path in selected.items():
        target = destination / name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    for path in destination.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def write_golden(destination: Path = GOLDEN_ROOT) -> None:
    """Generate the committed public package from one fresh transaction."""

    with tempfile.TemporaryDirectory(prefix="invarlock-acceptance-handoff-") as tmp:
        workspace = Path(tmp).resolve() / "workspace"
        run_handoff(workspace)
        _copy_golden(workspace, destination)


def print_handoff_summary(workspace: Path) -> None:
    """Print the useful decision and signed outputs from a successful handoff."""

    results = json.loads((workspace / "results.json").read_bytes())
    scenarios = results.get("scenarios")
    if not isinstance(scenarios, dict) or scenarios.get("accepted") is not True:
        raise RuntimeError("acceptance handoff summary is invalid")
    rejection_results = {
        name: passed for name, passed in scenarios.items() if name != "accepted"
    }
    if not rejection_results or any(
        passed is not True for passed in rejection_results.values()
    ):
        raise RuntimeError("acceptance handoff rejection summary is invalid")

    print("PASS offline acceptance handoff")
    print("Fixture decision: accepted")
    print(
        "Fail-closed scenarios rejected: "
        f"{len(rejection_results)}/{len(rejection_results)}"
    )
    print(f"Signed evidence: {workspace / 'handoff/evidence'}")
    print(f"Signed verifier receipt: {workspace / 'handoff/verification.receipt.json'}")
    print(f"Acceptance envelope: {workspace / 'handoff/acceptance.dsse.json'}")
    print(f"Scenario results: {workspace / 'results.json'}")
    print(f"Workspace: {workspace}")


def main() -> int:
    parser = argparse.ArgumentParser(description="run the offline acceptance handoff")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="generate the committed golden package at its fixed example path",
    )
    args = parser.parse_args()
    if args.write_golden:
        if args.workspace is not None:
            parser.error("--workspace cannot be mixed with --write-golden")
        write_golden()
        print(f"PASS generated golden handoff package: {GOLDEN_ROOT}")
        return 0
    workspace = (
        args.workspace.resolve()
        if args.workspace is not None
        else Path(tempfile.mkdtemp(prefix="invarlock-acceptance-handoff-")).resolve()
        / "workspace"
    )
    run_handoff(workspace)
    print_handoff_summary(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
