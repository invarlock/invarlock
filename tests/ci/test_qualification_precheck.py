from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from scripts import qualification_precheck
from scripts.qualification_precheck import validate


def _digest(marker: str) -> str:
    return "sha256:" + marker * 64


def _trust_profile(tmp_path: Path) -> tuple[Path, bytes]:
    root = tmp_path / "trust"
    root.mkdir()
    policy = b'{"thresholds":{}}\n'
    (root / "policy.json").write_bytes(policy)
    key = ed25519.Ed25519PrivateKey.generate()
    (root / "verifier.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    profile = root / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "format": "invarlock/trust-inputs-v1",
                "policy": {"path": "policy.json"},
                "anchors": {
                    "baseline_artifact_digest": _digest("a"),
                    "subject_artifact_digest": _digest("b"),
                    "schedule_digest": _digest("c"),
                    "baseline_runtime_digest": _digest("d"),
                    "subject_runtime_digest": _digest("e"),
                    "evidence_signer_fingerprint": _digest("f"),
                },
                "verifier": {
                    "identity": "invarlock-verifier/qualification",
                    "signing_key_path": "verifier.pem",
                },
                "allow_installed_scorers": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return profile, policy


def _preflight(policy: bytes) -> dict[str, object]:
    return {
        "ok": True,
        "artifact_digests": {
            "baseline": _digest("a"),
            "subject": _digest("b"),
        },
        "evidence_signer_fingerprint": _digest("f"),
        "request_digest": _digest("1"),
        "schedule_digest": _digest("c"),
        "policy_digest": f"sha256:{hashlib.sha256(policy).hexdigest()}",
        "runtime_image_digests": {
            "baseline": _digest("d"),
            "subject": _digest("e"),
        },
    }


def test_qualification_precheck_loads_trust_and_reserves_fresh_receipt(
    tmp_path: Path,
) -> None:
    profile, policy = _trust_profile(tmp_path)

    result = validate(
        preflight=_preflight(policy),
        trust_profile=profile,
        receipt=tmp_path / "receipt.json",
    )

    assert result["ok"] is True
    assert result["receipt"] == str(tmp_path / "receipt.json")
    assert result["artifact_digests"] == {
        "baseline": _digest("a"),
        "subject": _digest("b"),
    }
    assert result["evidence_signer_fingerprint"] == _digest("f")
    assert result["request_digest"] == _digest("1")
    assert result["policy_digest"] == f"sha256:{hashlib.sha256(policy).hexdigest()}"
    assert result["runtime_digests"] == {
        "baseline": _digest("d"),
        "subject": _digest("e"),
    }
    assert result["schedule_digest"] == _digest("c")
    assert str(result["trust_profile_digest"]).startswith("sha256:")
    assert str(result["verifier_fingerprint"]).startswith("sha256:")
    assert result["verifier_identity"] == "invarlock-verifier/qualification"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schedule_digest": _digest("9")}, "schedule digest"),
        ({"policy_digest": _digest("9")}, "policy does not match"),
        (
            {
                "artifact_digests": {
                    "baseline": _digest("9"),
                    "subject": _digest("b"),
                }
            },
            "baseline artifact digest",
        ),
        (
            {"evidence_signer_fingerprint": _digest("9")},
            "evidence signer fingerprint",
        ),
        ({"request_digest": "invalid"}, "normalized request digest"),
        (
            {
                "runtime_image_digests": {
                    "baseline": _digest("9"),
                    "subject": _digest("e"),
                }
            },
            "baseline runtime digest",
        ),
    ],
)
def test_qualification_precheck_rejects_trust_mismatch_before_execution(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    profile, policy = _trust_profile(tmp_path)
    preflight = _preflight(policy)
    preflight.update(mutation)

    with pytest.raises(ValueError, match=message):
        validate(
            preflight=preflight,
            trust_profile=profile,
            receipt=tmp_path / "receipt.json",
        )


def test_qualification_precheck_rejects_existing_receipt_before_execution(
    tmp_path: Path,
) -> None:
    profile, policy = _trust_profile(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        validate(
            preflight=_preflight(policy),
            trust_profile=profile,
            receipt=receipt,
        )


def test_qualification_precheck_expands_referenced_policy_and_verifier_key(
    tmp_path: Path,
) -> None:
    profile, policy = _trust_profile(tmp_path)
    first = validate(
        preflight=_preflight(policy),
        trust_profile=profile,
        receipt=tmp_path / "first-receipt.json",
    )

    changed_policy = b'{"thresholds":{"changed":true}}\n'
    profile.parent.joinpath("policy.json").write_bytes(changed_policy)
    key = ed25519.Ed25519PrivateKey.generate()
    profile.parent.joinpath("verifier.pem").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    second = validate(
        preflight=_preflight(changed_policy),
        trust_profile=profile,
        receipt=tmp_path / "second-receipt.json",
    )

    assert second["trust_profile_digest"] == first["trust_profile_digest"]
    assert second["policy_digest"] != first["policy_digest"]
    assert second["verifier_fingerprint"] != first["verifier_fingerprint"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"ok": False}, "not successful"),
        ({"artifact_digests": None}, "artifact digests are missing"),
        ({"runtime_image_digests": None}, "runtime image digests are missing"),
    ),
)
def test_qualification_precheck_rejects_missing_preflight_units(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    profile, policy = _trust_profile(tmp_path)
    preflight = _preflight(policy)
    preflight.update(mutation)

    with pytest.raises(ValueError, match=message):
        validate(
            preflight=preflight,
            trust_profile=profile,
            receipt=tmp_path / "receipt.json",
        )


def test_qualification_precheck_main_emits_validated_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile, policy = _trust_profile(tmp_path)
    stdin = io.TextIOWrapper(io.BytesIO(json.dumps(_preflight(policy)).encode()))
    monkeypatch.setattr(sys, "stdin", stdin)

    result = qualification_precheck.main(
        [
            "--trust-profile",
            str(profile),
            "--receipt",
            str(tmp_path / "receipt.json"),
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
