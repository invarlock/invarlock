"""Native offline replay parity for retained signed evaluator transactions.

These are historical observations, not fresh model runs. Reference anchors are
read from the repository's separate retained receipts for this compatibility
check; the test does not establish an independent recipient trust policy.
"""

import json
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[2]
CORPUS = sorted(
    (ROOT / "examples/evaluator-qualification/signed-transactions").glob(
        "*/verification.receipt.json"
    )
)
pytestmark = pytest.mark.skipif(
    not os.environ.get("INVARLOCK_NATIVE_REPORT_BINDINGS"),
    reason="native extension qualification is opt-in",
)


@pytest.mark.parametrize("receipt", CORPUS, ids=lambda p: p.parent.name)
def test_retained_native_replay(receipt, tmp_path):
    from invarlock.evidence_verification import (
        EvidenceVerificationError,
        verify_evidence,
    )

    directory = receipt.parent
    anchors = json.loads(receipt.read_bytes())["statement"]["anchors"]
    key = tmp_path / "verifier.pem"
    key.write_bytes(
        Ed25519PrivateKey.from_private_bytes(bytes(range(32))).private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key.chmod(0o600)
    options = {
        "policy_path": directory / "policy.json",
        "expected_baseline_artifact": anchors["artifact_digests"]["baseline"],
        "expected_subject_artifact": anchors["artifact_digests"]["subject"],
        "expected_schedule": anchors["schedule_digest"],
        "expected_baseline_runtime": anchors["runtime_digests"]["baseline"],
        "expected_subject_runtime": anchors["runtime_digests"]["subject"],
        "expected_signer": anchors["pack_signer_fingerprint"],
        "verifier_signing_key_path": key,
        "verifier_identity": "retained-replay-fixture",
    }
    if "request_digest" in anchors:
        options["expected_request_digest"] = anchors["request_digest"]
    results = []
    for backend in ["python", "rust"]:
        (tmp_path / backend).mkdir()
        output = tmp_path / backend / "receipt.json"
        try:
            result = verify_evidence(
                directory / "evidence", receipt_path=output, backend=backend, **options
            )
            status = 0
        except EvidenceVerificationError as error:
            result = error
            status = error.exit_code
        results.append(
            (status, result.payload, output.read_bytes() if output.exists() else None)
        )
    difference = {
        key: (results[0][1].get(key), results[1][1].get(key))
        for key in set(results[0][1]) | set(results[1][1])
        if results[0][1].get(key) != results[1][1].get(key)
    }
    assert not difference, difference
    assert results[1][0] == results[0][0]
    assert results[1][2] == results[0][2]
    assert results[0][1]["integrity_ok"] is True
