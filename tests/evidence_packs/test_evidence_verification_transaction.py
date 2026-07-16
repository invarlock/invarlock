from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from invarlock.evidence_pack_support import EvidencePackStatus
from invarlock.evidence_verification import (
    EvidenceVerificationError,
    verify_evidence,
)
from tests.evidence_packs.test_evidence_pack import _publish


def _verifier_key(tmp_path: Path) -> Path:
    path = tmp_path / "verifier-key.pem"
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return path


def _verify_kwargs(
    tmp_path: Path,
    *,
    delta_min_pp: float = -100.0,
) -> tuple[Path, dict[str, object]]:
    pack, policy, signer, runtimes, _evidence_key, arguments = _publish(
        tmp_path, delta_min_pp=delta_min_pp
    )
    baseline = arguments["baseline"]
    subject = arguments["subject"]
    dataset = arguments["dataset"]
    return pack, {
        "policy_path": policy,
        "expected_baseline_artifact": baseline.digest,
        "expected_subject_artifact": subject.digest,
        "expected_schedule": dataset.digest,
        "expected_baseline_runtime": runtimes["baseline"],
        "expected_subject_runtime": runtimes["subject"],
        "expected_signer": signer,
        "receipt_path": tmp_path / "verification.receipt.json",
        "verifier_signing_key_path": _verifier_key(tmp_path),
        "verifier_identity": "invarlock-verifier/release",
    }


def test_independent_verification_user_journey_emits_a_signed_external_receipt(
    tmp_path: Path,
) -> None:
    pack, kwargs = _verify_kwargs(tmp_path)

    verified = verify_evidence(pack, **kwargs)  # type: ignore[arg-type]

    receipt = Path(kwargs["receipt_path"])
    assert verified.evidence_path == pack.resolve()
    assert verified.receipt_path == receipt.resolve()
    assert receipt.is_file()
    assert verified.payload["ok"] is True
    assert verified.payload["verifier_identity"] == "invarlock-verifier/release"
    assert "Evidence:" in verified.summary
    assert "Comparison: single-comparison" in verified.summary
    assert "Evidence signer:" in verified.summary
    assert "Verifier signer:" in verified.summary
    assert "Receipt:" in verified.summary
    assert '"ok":true' in verified.as_json()


def test_failed_policy_is_reported_after_signing_the_verification_receipt(
    tmp_path: Path,
) -> None:
    pack, kwargs = _verify_kwargs(tmp_path, delta_min_pp=-10.0)

    with pytest.raises(EvidenceVerificationError) as caught:
        verify_evidence(pack, **kwargs)  # type: ignore[arg-type]

    assert caught.value.exit_code == int(EvidencePackStatus.REPORTS)
    assert caught.value.payload["integrity_ok"] is True
    assert caught.value.payload["policy_verdict"] == "fail"
    assert Path(kwargs["receipt_path"]).is_file()
    assert '"ok":false' in caught.value.as_json()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"policy_path": None}, "independent policy is required"),
        ({"expected_baseline_artifact": None}, "baseline artifact anchor"),
        ({"expected_subject_artifact": ""}, "subject artifact anchor"),
        ({"expected_schedule": None}, "schedule anchor"),
        ({"expected_baseline_runtime": None}, "baseline runtime anchor"),
        ({"expected_subject_runtime": ""}, "subject runtime anchor"),
        ({"expected_signer": None}, "evidence signer"),
        ({"receipt_path": None}, "receipt destination"),
        ({"verifier_signing_key_path": None}, "signing key is required"),
        ({"verifier_identity": " "}, "verifier identity"),
    ],
)
def test_verification_requires_every_independent_root(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    pack, kwargs = _verify_kwargs(tmp_path)
    kwargs.update(mutation)

    with pytest.raises(EvidenceVerificationError, match=message):
        verify_evidence(pack, **kwargs)  # type: ignore[arg-type]


def test_verification_rejects_unsafe_evidence_policy_key_and_receipt_paths(
    tmp_path: Path,
) -> None:
    pack, kwargs = _verify_kwargs(tmp_path)
    evidence_link = tmp_path / "evidence-link"
    evidence_link.symlink_to(pack, target_is_directory=True)
    with pytest.raises(EvidenceVerificationError, match="real directory"):
        verify_evidence(evidence_link, **kwargs)  # type: ignore[arg-type]

    policy = Path(kwargs["policy_path"])
    policy_link = tmp_path / "policy-link.json"
    policy_link.symlink_to(policy)
    with pytest.raises(EvidenceVerificationError, match="real regular file"):
        verify_evidence(
            pack,
            **{**kwargs, "policy_path": policy_link},  # type: ignore[arg-type]
        )

    key = Path(kwargs["verifier_signing_key_path"])
    key_link = tmp_path / "key-link.pem"
    key_link.symlink_to(key)
    with pytest.raises(EvidenceVerificationError, match="real regular file"):
        verify_evidence(
            pack,
            **{**kwargs, "verifier_signing_key_path": key_link},  # type: ignore[arg-type]
        )

    with pytest.raises(EvidenceVerificationError, match="outside"):
        verify_evidence(
            pack,
            **{**kwargs, "receipt_path": pack / "receipt.json"},  # type: ignore[arg-type]
        )


def test_verification_maps_receipt_write_failure_to_public_error(
    tmp_path: Path,
) -> None:
    pack, kwargs = _verify_kwargs(tmp_path)
    receipt = Path(kwargs["receipt_path"])
    receipt.write_text("caller-owned\n", encoding="utf-8")

    with pytest.raises(EvidenceVerificationError, match="already exists") as caught:
        verify_evidence(pack, **kwargs)  # type: ignore[arg-type]

    assert caught.value.exit_code == 2
    assert caught.value.payload["errors"] == [str(caught.value)]


def test_verification_error_default_payload_is_stable_json() -> None:
    error = EvidenceVerificationError("unsafe evidence")

    assert error.exit_code == 2
    assert error.payload == {
        "format_version": "invarlock/evidence-verification-error-v1",
        "ok": False,
        "errors": ["unsafe evidence"],
        "warnings": [],
    }
    assert error.as_json() == (
        '{"errors":["unsafe evidence"],'
        '"format_version":"invarlock/evidence-verification-error-v1",'
        '"ok":false,"warnings":[]}'
    )
