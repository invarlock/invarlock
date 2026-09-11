"""Neutral test imports and builders for recovered evaluation behavior coverage."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.captured_contracts import (
    PAYLOADS,
    CapturedSnapshot,
    captured_snapshot,
    json_object,
)
from invarlock.captured_evidence_publication import publish_captured_evidence
from invarlock.captured_verification import verify_captured_receipt
from invarlock.evaluation_comparison import comparison
from invarlock.evaluation_comparison.capacity import DEFAULT_MAX_BOOTSTRAP_DRAWS
from invarlock.evaluation_comparison.comparison import compare_runs, make_run
from invarlock.evaluation_record_contracts import contracts
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
    read_json,
    validate,
    write_directory,
    write_new,
)
from invarlock.evaluation_records.adapters import load_run
from invarlock.evaluation_records.cases import (
    CASE_SET_FORMAT,
    canonical_case_set,
    case_set_digest,
    validate_run_case_set,
)
from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_verification import EvidenceVerificationError, verify_evidence


def pack_json(snapshot: CapturedSnapshot, role: str) -> dict[str, Any]:
    return json_object(snapshot.files[PAYLOADS.get(role, role)], role)


def write_snapshot(root: Path, snapshot: CapturedSnapshot) -> None:
    for name, raw in snapshot.files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def rebind_pack(snapshot, key, **payloads):
    """Author adversarial pack bytes with the existing real pack-signing fixture."""
    from tests.core.test_captured_security_contracts import _rebind

    with TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        write_snapshot(root, snapshot)
        for role, value in payloads.items():
            (root / PAYLOADS[role]).write_bytes(canonical_json_bytes(value))
        _rebind(root, key)
        with captured_snapshot(root) as rebound:
            return rebound


def build_pack(
    baseline,
    subject,
    policy,
    key=None,
    *,
    max_bootstrap_draws=DEFAULT_MAX_BOOTSTRAP_DRAWS,
):
    """Build the production pack, then detach its bounded immutable snapshot."""
    report = compare_runs(
        baseline, subject, policy, max_bootstrap_draws=max_bootstrap_draws
    )
    normalized = {
        "format": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {"adapter": "invarlock", "run_digest": digest(baseline)},
            "subject": {"adapter": "invarlock", "run_digest": digest(subject)},
            "policy_digest": digest(policy),
        },
    }
    with TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        signing_key = root / "signer.pem" if key is not None else None
        if signing_key is not None:
            signing_key.write_bytes(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
        pack = publish_captured_evidence(
            root / "pack",
            baseline=baseline,
            subject=subject,
            policy=policy,
            comparison=report,
            normalized_request=normalized,
            request_digest=digest(normalized),
            signing_key_path=signing_key,
            unsigned=key is None,
        )
        with captured_snapshot(pack) as snapshot:
            return snapshot


def replay_pack(
    snapshot,
    *,
    public_key,
    expected_baseline_run,
    expected_subject_run,
    policy,
    max_bootstrap_draws=DEFAULT_MAX_BOOTSTRAP_DRAWS,
):
    """Use independent inputs and a separate verifier key/receipt, never a fake replay."""
    normalized = {
        "format": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {"adapter": "invarlock", "run_digest": expected_baseline_run},
            "subject": {"adapter": "invarlock", "run_digest": expected_subject_run},
            "policy_digest": digest(policy),
        },
    }
    with TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        pack = root / "pack"
        write_snapshot(pack, snapshot)
        policy_path = root / "approved-policy.json"
        policy_path.write_bytes(canonical_json_bytes(policy))
        verifier = Ed25519PrivateKey.generate()
        verifier_path = root / "verifier.pem"
        verifier_path.write_bytes(
            verifier.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        receipt = root / "receipt.json"
        anchors = {
            "policy_path": policy_path,
            "expected_baseline_run": expected_baseline_run,
            "expected_subject_run": expected_subject_run,
            "expected_request_digest": digest(normalized),
            "expected_signer": public_key_fingerprint(public_key),
        }
        try:
            result = verify_evidence(
                pack,
                **anchors,
                receipt_path=receipt,
                verifier_signing_key_path=verifier_path,
                verifier_identity="record-tests",
                max_bootstrap_draws=max_bootstrap_draws,
            ).payload
        except EvidenceVerificationError as exc:
            if exc.exit_code != 7:
                raise
            result = exc.payload
        assert result["integrity_ok"] is True
        authenticated = verify_captured_receipt(
            receipt,
            pack,
            **anchors,
            expected_verifier_identity="record-tests",
            expected_verifier_fingerprint=public_key_fingerprint(verifier.public_key()),
        )
        assert authenticated.ok, authenticated.errors
        assert (
            json_object(receipt.read_bytes(), "receipt")["statement"]["verdict"][
                "decision"
            ]
            == result["decision"]
        )
        return pack_json(snapshot, "report")


def materialize_captured_request(
    root: Path,
    baseline: dict[str, Any],
    subject: dict[str, Any],
    policy: dict[str, Any],
    *,
    evidence: str = "evidence",
) -> Path:
    """Write one closed captured request using only neutral record paths."""
    root.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("baseline.json", baseline),
        ("subject.json", subject),
        ("policy.json", policy),
    ):
        (root / name).write_bytes(canonical_json_bytes(value))
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {"path": "baseline.json", "adapter": "invarlock"},
            "subject": {"path": "subject.json", "adapter": "invarlock"},
            "policy": "policy.json",
        },
        "output": {"evidence": evidence},
    }
    path = root / "request.yaml"
    path.write_bytes(canonical_json_bytes(request))
    return path


def captured_request_digest(
    baseline: dict[str, Any], subject: dict[str, Any], policy: dict[str, Any]
) -> str:
    """Return the normalized request binding emitted into captured evidence."""
    return digest(
        {
            "format": "invarlock/evaluation-request-v2",
            "execution": {"mode": "captured"},
            "comparison": {
                "baseline": {
                    "adapter": "invarlock",
                    "run_digest": digest(baseline),
                },
                "subject": {
                    "adapter": "invarlock",
                    "run_digest": digest(subject),
                },
                "policy_digest": digest(policy),
            },
        }
    )


__all__ = [
    "CASE_SET_FORMAT",
    "captured_request_digest",
    "EvaluationRecordsError",
    "compare_runs",
    "contracts",
    "case_set_digest",
    "canonical_case_set",
    "comparison",
    "build_pack",
    "pack_json",
    "replay_pack",
    "rebind_pack",
    "write_snapshot",
    "digest",
    "example_project",
    "load_run",
    "make_run",
    "materialize_captured_request",
    "read_json",
    "validate",
    "validate_run_case_set",
    "write_directory",
    "write_new",
]
