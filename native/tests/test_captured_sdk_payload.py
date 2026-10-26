"""Captured SDK collection types remain compatible across explicit backends."""

import json
from pathlib import Path

import pytest

from invarlock.evidence_verification import EvidenceVerificationError, verify_evidence
from invarlock.native_backend import transact, verify_captured_evidence
from native.tests.test_evidence_differential import config_for


@pytest.mark.parametrize("scenario", ["pass", "regression"])
def test_captured_raw_sdk_payload_and_wire_types(tmp_path, monkeypatch, scenario):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "experiments/rust/python")
    )
    from invarlock_rust_experiment.fixtures import make_fixture

    root = tmp_path.resolve()
    fixture = root / "fixture"
    make_fixture(fixture, 12, scenario)
    options = config_for(fixture, root / "unused.json")
    evidence = Path(options.pop("evidence_path"))
    options.pop("receipt_path")
    for key in ("policy_path", "verifier_signing_key_path"):
        options[key] = Path(options[key])
    for name in ("python", "rust", "direct", "wire"):
        (root / name).mkdir()

    results = []
    for backend in ("python", "rust"):
        try:
            result = verify_evidence(
                evidence,
                **options,
                receipt_path=root / backend / "receipt.json",
                **({"backend": "rust"} if backend == "rust" else {}),
            )
            status, payload = 0, result.payload
        except EvidenceVerificationError as exc:
            status, payload = exc.exit_code, exc.payload
        assert type(payload["metric_summaries"]) is tuple
        results.append((status, payload))
    assert results[0] == results[1]
    assert results[0][0] == (0 if scenario == "pass" else 7)

    direct = verify_captured_evidence(
        evidence, **options, receipt_path=root / "direct/receipt.json"
    )
    assert type(direct["metric_summaries"]) is tuple
    assert direct == results[0][1]

    wire = transact(
        "verify-captured",
        {
            "evidence_path": evidence,
            **options,
            "receipt_path": root / "wire/receipt.json",
        },
    )
    assert type(wire["payload"]["metric_summaries"]) is list
    assert wire["status"] == results[0][0]
    assert wire["payload"] == json.loads(json.dumps(results[0][1]))
    receipts = [
        (root / name / "receipt.json").read_bytes()
        for name in ("python", "rust", "direct", "wire")
    ]
    assert all(receipt == receipts[0] for receipt in receipts)
