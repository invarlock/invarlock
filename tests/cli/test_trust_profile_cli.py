from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.evidence_pack_contract import InputIdentity
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_receipt import verify_signed_verification_receipt
from invarlock.trust_inputs import load_trust_inputs
from tests.evidence_packs.test_evidence_pack import _publish


def _verifier_key(path: Path) -> tuple[Path, str]:
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)
    return path, public_key_fingerprint(key.public_key())


def _case(tmp_path: Path) -> dict[str, object]:
    pack, policy, signer, runtimes, _evidence_key, arguments = _publish(tmp_path)
    baseline = arguments["baseline"]
    subject = arguments["subject"]
    dataset = arguments["dataset"]
    assert isinstance(baseline, InputIdentity)
    assert isinstance(subject, InputIdentity)
    assert isinstance(dataset, InputIdentity)
    trust_root = tmp_path / "trust"
    trust_root.mkdir()
    profile_policy = trust_root / "policy.json"
    profile_policy.write_bytes(policy.read_bytes())
    verifier_key, verifier_fingerprint = _verifier_key(trust_root / "verifier.pem")
    request_digest = (
        "sha256:" + hashlib.sha256((pack / "request.json").read_bytes()).hexdigest()
    )
    profile = trust_root / "trust-inputs.json"
    profile.write_text(
        json.dumps(
            {
                "format": "invarlock/trust-inputs-v1",
                "policy": {"path": profile_policy.name},
                "anchors": {
                    "baseline_artifact_digest": baseline.digest,
                    "subject_artifact_digest": subject.digest,
                    "schedule_digest": dataset.digest,
                    "baseline_runtime_digest": runtimes["baseline"],
                    "subject_runtime_digest": runtimes["subject"],
                    "evidence_signer_fingerprint": signer,
                    "request_digest": request_digest,
                },
                "verifier": {
                    "identity": "invarlock-verifier/release",
                    "signing_key_path": verifier_key.name,
                },
                "allow_installed_scorers": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "pack": pack,
        "policy": profile_policy,
        "signer": signer,
        "runtimes": runtimes,
        "baseline": baseline,
        "subject": subject,
        "dataset": dataset,
        "profile": profile,
        "verifier_key": verifier_key,
        "verifier_fingerprint": verifier_fingerprint,
        "request_digest": request_digest,
    }


def test_profile_and_explicit_verification_have_the_same_decision_and_anchors(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    explicit_receipt = tmp_path / "explicit.receipt.json"
    profile_receipt = tmp_path / "profile.receipt.json"
    runner = CliRunner()
    baseline = case["baseline"]
    subject = case["subject"]
    dataset = case["dataset"]
    runtimes = case["runtimes"]
    assert isinstance(baseline, InputIdentity)
    assert isinstance(subject, InputIdentity)
    assert isinstance(dataset, InputIdentity)
    assert isinstance(runtimes, dict)

    explicit = runner.invoke(
        app,
        [
            "verify",
            str(case["pack"]),
            "--policy",
            str(case["policy"]),
            "--expected-baseline-artifact",
            baseline.digest,
            "--expected-subject-artifact",
            subject.digest,
            "--expected-schedule",
            dataset.digest,
            "--expected-baseline-runtime",
            str(runtimes["baseline"]),
            "--expected-subject-runtime",
            str(runtimes["subject"]),
            "--expected-signer",
            str(case["signer"]),
            "--expected-request-digest",
            str(case["request_digest"]),
            "--receipt",
            str(explicit_receipt),
            "--verifier-signing-key",
            str(case["verifier_key"]),
            "--verifier-identity",
            "invarlock-verifier/release",
            "--json",
        ],
    )
    profiled = runner.invoke(
        app,
        [
            "verify",
            str(case["pack"]),
            "--trust-profile",
            str(case["profile"]),
            "--receipt",
            str(profile_receipt),
            "--json",
        ],
    )

    assert explicit.exit_code == profiled.exit_code == 0
    explicit_payload = json.loads(explicit.stdout)
    profile_payload = json.loads(profiled.stdout)
    for field in ("ok", "integrity_ok", "policy_verdict", "anchors"):
        assert profile_payload[field] == explicit_payload[field]
    loaded = load_trust_inputs(Path(case["profile"]))
    assert profile_payload["trust_profile_digest"] == loaded.profile_digest
    profile_statement = json.loads(profile_receipt.read_text())["statement"]
    explicit_statement = json.loads(explicit_receipt.read_text())["statement"]
    assert (
        profile_statement["verifier"]["trust_profile_digest"] == loaded.profile_digest
    )
    assert explicit_statement["verifier"]["trust_profile_digest"] is None

    verified_receipt = verify_signed_verification_receipt(
        profile_receipt,
        Path(case["pack"]),
        policy_path=Path(case["policy"]),
        expected_artifact_digests={
            "baseline": baseline.digest,
            "subject": subject.digest,
        },
        expected_schedule_digest=dataset.digest,
        expected_runtime_digests={
            "baseline": str(runtimes["baseline"]),
            "subject": str(runtimes["subject"]),
        },
        expected_pack_signer_fingerprint=str(case["signer"]),
        expected_request_digest=str(case["request_digest"]),
        expected_verifier_identity="invarlock-verifier/release",
        expected_verifier_fingerprint=str(case["verifier_fingerprint"]),
        expected_trust_profile_digest=loaded.profile_digest,
    )
    assert verified_receipt.ok is True


def test_profile_ignores_environment_anchors_and_rejects_cli_mixing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    runner = CliRunner()
    monkeypatch.setenv("INVARLOCK_POLICY", "/missing/from/environment")
    monkeypatch.setenv("INVARLOCK_EXPECTED_BASELINE_ARTIFACT", "sha256:bad")
    monkeypatch.setenv("INVARLOCK_EXPECTED_REQUEST_DIGEST", "sha256:bad")
    monkeypatch.setenv("INVARLOCK_ALLOW_INSTALLED_SCORERS", "1")
    receipt = tmp_path / "environment.receipt.json"

    profiled = runner.invoke(
        app,
        [
            "verify",
            str(case["pack"]),
            "--trust-profile",
            str(case["profile"]),
            "--receipt",
            str(receipt),
        ],
    )
    assert profiled.exit_code == 0, profiled.stdout

    mixed = runner.invoke(
        app,
        [
            "verify",
            str(case["pack"]),
            "--trust-profile",
            str(case["profile"]),
            "--policy",
            str(case["policy"]),
            "--receipt",
            str(tmp_path / "mixed.receipt.json"),
        ],
    )
    assert mixed.exit_code == 2
    assert "cannot be mixed with --policy" in mixed.stdout


def test_trust_profile_must_remain_outside_submitted_evidence(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    pack = Path(case["pack"])
    submitted_profile = pack / "manifest.json"

    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(pack),
            "--trust-profile",
            str(submitted_profile),
            "--receipt",
            str(tmp_path / "submitted-profile.receipt.json"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "independent trust profile must remain outside" in payload["errors"][0]


@pytest.mark.parametrize(
    ("material", "message"),
    [
        ("policy", "independent policy"),
        ("key", "verifier Ed25519 signing key"),
    ],
)
def test_external_profile_cannot_reference_trust_material_inside_evidence(
    tmp_path: Path,
    material: str,
    message: str,
) -> None:
    case = _case(tmp_path)
    pack = Path(case["pack"])
    profile_payload = json.loads(Path(case["profile"]).read_text(encoding="utf-8"))
    external_policy = tmp_path / "independent-policy.json"
    external_key = tmp_path / "independent-verifier.pem"
    external_policy.write_bytes(Path(case["policy"]).read_bytes())
    external_key.write_bytes(Path(case["verifier_key"]).read_bytes())
    profile_payload["policy"]["path"] = external_policy.name
    profile_payload["verifier"]["signing_key_path"] = external_key.name
    submitted = pack / "manifest.json"
    if material == "policy":
        profile_payload["policy"]["path"] = submitted.relative_to(tmp_path).as_posix()
    else:
        profile_payload["verifier"]["signing_key_path"] = submitted.relative_to(
            tmp_path
        ).as_posix()
    profile = tmp_path / "external-trust-inputs.json"
    profile.write_text(
        json.dumps(profile_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(pack),
            "--trust-profile",
            str(profile),
            "--receipt",
            str(tmp_path / f"submitted-{material}.receipt.json"),
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert f"{message} must remain outside" in payload["errors"][0]


def test_profile_verification_consumes_captured_policy_and_key_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invarlock import evidence_verification

    case = _case(tmp_path)
    policy_path = Path(case["policy"])
    key_path = Path(case["verifier_key"])
    original_policy = policy_path.read_bytes()
    original_verify = evidence_verification.verify_evidence
    replacement_key = ed25519.Ed25519PrivateKey.generate()
    replacement_key_bytes = replacement_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    def swap_paths_then_verify(*args: Any, **kwargs: Any) -> object:
        policy_path.write_text("{}\n", encoding="utf-8")
        key_path.write_bytes(replacement_key_bytes)
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(
        evidence_verification,
        "verify_evidence",
        swap_paths_then_verify,
    )
    receipt = tmp_path / "captured.receipt.json"

    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(case["pack"]),
            "--trust-profile",
            str(case["profile"]),
            "--receipt",
            str(receipt),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    statement = json.loads(receipt.read_text(encoding="utf-8"))["statement"]
    assert statement["anchors"]["policy_digest"] == (
        "sha256:" + hashlib.sha256(original_policy).hexdigest()
    )
    assert (
        statement["verifier"]["signing_key_fingerprint"] == case["verifier_fingerprint"]
    )
    assert statement["verifier"]["signing_key_fingerprint"] != (
        public_key_fingerprint(replacement_key.public_key())
    )


def test_malformed_trust_profile_fails_closed_before_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from invarlock import evidence_verification

    case = _case(tmp_path)
    profile = Path(case["profile"])
    profile.write_text("[]\n", encoding="utf-8")
    receipt = tmp_path / "malformed.receipt.json"
    monkeypatch.setattr(
        evidence_verification,
        "verify_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "malformed trust input must fail before pack verification"
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(case["pack"]),
            "--trust-profile",
            str(profile),
            "--receipt",
            str(receipt),
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "must decode to a JSON object" in payload["errors"][0]
    assert not receipt.exists()
