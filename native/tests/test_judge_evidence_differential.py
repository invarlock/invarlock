"""Native judge acceptance and transport against independent Python replay."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization

from invarlock.judge_measurements.acceptance import (
    verify_judge_evidence,
    verify_signed_judge_verification_receipt,
    write_signed_judge_verification_receipt,
)
from tests.judge_measurements.test_evidence_acceptance import (
    KEY,
    _publish,
    _verifier_key_path,
)


def native(command, config):
    binary = os.environ.get("INVARLOCK_NATIVE_VERIFY")
    assert binary and Path(binary).is_file(), "set INVARLOCK_NATIVE_VERIFY"
    process = subprocess.run(
        [binary, command, "--request", "-"],
        input=json.dumps(config).encode(),
        capture_output=True,
        timeout=60,
    )
    assert process.stdout, process.stderr.decode()
    return process.returncode, json.loads(process.stdout)


def plain(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize(
    "case",
    [
        "pass",
        "regression",
        "incomplete",
        "advisory",
        "unsigned",
        "wrong_signer",
        "wrong_subject",
    ],
)
def test_acceptance_and_signed_transport(tmp_path, case):
    root = tmp_path.resolve()
    options = {}
    if case == "regression":
        options.update(baseline=1, subject=0)
    elif case == "incomplete":
        options["incomplete"] = True
    elif case == "advisory":
        options["role"] = "advisory"
    elif case == "unsigned":
        options["signed"] = False
    publication, policy_path = _publish(root, **options)
    if case in {"wrong_signer", "wrong_subject"}:
        policy = json.loads(policy_path.read_bytes())
        if case == "wrong_signer":
            policy["trusted_signer"]["identity"] = "another-signer"
        else:
            policy["intended_subject"] = "sha256:" + "0" * 64
        policy_path.write_text(json.dumps(policy))
    config = {
        "evidence_path": str(publication.path),
        "recipient_policy_path": str(policy_path),
    }
    expected = verify_judge_evidence(
        publication.path, recipient_policy_path=policy_path
    )
    status, result = native("verify-judge", config)
    assert result == expected.to_dict()
    assert status == (0 if expected.accepted else 7 if expected.verified else 4)
    key_path = _verifier_key_path(root)
    reference_receipt, native_receipt = (
        root / "python-receipt.json",
        root / "rust-receipt.json",
    )
    fingerprint = write_signed_judge_verification_receipt(
        publication.path,
        expected,
        reference_receipt,
        recipient_policy_path=policy_path,
        verifier_identity="receipt-verifier",
        verifier_signing_key_path=key_path,
    )
    status, written = native(
        "write-judge-receipt",
        {
            **config,
            "result": result,
            "receipt_path": str(native_receipt),
            "verifier_identity": "receipt-verifier",
            "verifier_signing_key_path": str(key_path),
        },
    )
    assert status == 0, written
    assert written["verifier_fingerprint"] == fingerprint
    assert native_receipt.read_bytes() == reference_receipt.read_bytes()
    for selected_fingerprint in (fingerprint, "sha256:" + "0" * 64):
        args = {
            "expected_verifier_identity": "receipt-verifier",
            "expected_verifier_fingerprint": selected_fingerprint,
            "expected_recipient_policy_sha256": expected.recipient_policy_sha256,
        }
        reference = verify_signed_judge_verification_receipt(native_receipt, **args)
        status, received = native(
            "verify-judge-receipt", {"receipt_path": str(native_receipt), **args}
        )
        expected_receipt = plain(dataclasses.asdict(reference))
        if reference.result is not None:
            expected_receipt["result"] = reference.result.to_dict()
        assert received == expected_receipt
        assert status == (0 if reference.ok else 4)


def test_native_publication_matches_python(tmp_path):
    root = tmp_path.resolve()
    publication, policy_path = _publish(root)
    key = root / "signer.pem"
    key.write_bytes(
        KEY.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    names = {
        "plan": "plan.json",
        "measurements": "measurements.json",
        "baseline_run": "baseline_run.json",
        "subject_run": "subject_run.json",
        "analysis_policy": "analysis_policy.json",
    }
    request = {
        name: json.loads((publication.path / path).read_bytes())
        for name, path in names.items()
    }
    destination = root / "native-evidence"
    status, result = native(
        "publish-judge",
        {
            **request,
            "destination": str(destination),
            "signing_key_path": str(key),
            "signer_identity": "example-signer",
        },
    )
    assert status == 0, result
    assert sorted(p.name for p in destination.iterdir()) == sorted(
        p.name for p in publication.path.iterdir()
    )
    for reference in publication.path.iterdir():
        assert (destination / reference.name).read_bytes() == reference.read_bytes()
    assert verify_judge_evidence(
        destination, recipient_policy_path=policy_path
    ).accepted
