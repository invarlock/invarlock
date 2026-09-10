"""The bounded full comparison survives signed replay without splitting its sample."""

import json
import subprocess
import sys
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.evaluation_record_contracts.contracts import digest
from invarlock.evidence_pack_integrity import public_key_fingerprint
from tests._evaluation_support import (
    EvaluationRecordsError,
    build_pack,
    captured_request_digest,
    case_set_digest,
    compare_runs,
    comparison,
    make_run,
    pack_json,
    write_snapshot,
)


def material(count=12000):
    baseline = make_run(
        [
            {
                "id": f"case-{i:05d}",
                "input": f"question {i}",
                "expected": "yes",
                "output": "yes",
                "metadata": {"group": "first" if i < 6000 else "second"},
            }
            for i in range(count)
        ],
        source={"name": "evaluation", "version": "1"},
        run_id="baseline",
        artifact_digest="sha256:" + "a" * 64,
    )
    candidate = deepcopy(baseline)
    candidate["run_id"] = "candidate"
    for i, row in enumerate(candidate["records"]):
        if i % 1000 == 0:
            row["output"] = "no"
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [
            {
                "name": "quality",
                "kind": "exact_match",
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 6000,
                "maximum_regression": 0.01,
                "maximum_interval_width": 0.02,
            }
        ],
        "slices": [
            {"name": group, "where": {"group": group}} for group in ("first", "second")
        ],
        "expected_case_set_digest": case_set_digest(
            {
                "format": "invarlock/evaluation-case-set-v1",
                "cases": [
                    {k: row[k] for k in ("id", "input", "expected", "metadata")}
                    for row in baseline["records"]
                ],
            }
        ),
    }
    return baseline, candidate, policy


@pytest.mark.parametrize("count", [12000, 50000])
def test_full_capacity_signed_independent_recipient(tmp_path, count):
    baseline, candidate, policy = material(count)
    key = Ed25519PrivateKey.generate()
    evidence = build_pack(baseline, candidate, policy, key)
    report = pack_json(evidence, "report")
    assert report["decision"] == "pass"
    assert [m["count"] for m in report["metrics"]] == [
        count,
        6000,
        count - 6000,
    ]
    write_snapshot(tmp_path / "pack", evidence)
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    verifier = Ed25519PrivateKey.generate()
    (tmp_path / "verifier.pem").write_bytes(
        verifier.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    # The recipient receives the policy and run digests independently of evidence.
    script = """
import json, sys
from pathlib import Path
from invarlock.engine import verify_evidence
from invarlock.captured_verification import verify_captured_receipt
root = Path(sys.argv[1])
anchors = dict(
    expected_baseline_run=sys.argv[2], expected_subject_run=sys.argv[3],
    expected_request_digest=sys.argv[4], expected_signer=sys.argv[5],
    policy_path=root / 'policy.json',
)
result = verify_evidence(
    root / 'pack', **anchors, receipt_path=root / 'receipt.json',
    verifier_signing_key_path=root / 'verifier.pem', verifier_identity='capacity-test',
)
assert result.payload['decision'] == 'pass' and result.payload['integrity_ok'] is True
receipt = verify_captured_receipt(
    root / 'receipt.json', root / 'pack', **anchors,
    expected_verifier_identity='capacity-test', expected_verifier_fingerprint=sys.argv[6],
)
assert receipt.ok, receipt.errors
print((root / 'pack/reports/evaluation.report.json').read_text())
"""
    received = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(tmp_path),
            digest(baseline),
            digest(candidate),
            captured_request_digest(baseline, candidate, policy),
            public_key_fingerprint(key.public_key()),
            public_key_fingerprint(verifier.public_key()),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert json.loads(received.stdout) == report


def test_capacity_policy_minimum_and_mutual_omission_before_arithmetic(monkeypatch):
    baseline, candidate, policy = material()
    policy["slices"] = []
    policy["metrics"][0]["minimum_count"] = 12000
    assert compare_runs(baseline, candidate, policy)["decision"] == "pass"
    baseline["records"].pop()
    candidate["records"].pop()

    def forbidden(*args, **kwargs):
        pytest.fail("metric arithmetic must not start after planned membership changes")

    monkeypatch.setattr(comparison, "score", forbidden)
    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(EvaluationRecordsError, match="planned case set"):
        compare_runs(baseline, candidate, policy)


def test_capacity_plus_one_rejected_before_metric_arithmetic(monkeypatch):
    baseline, candidate, policy = material(50000)
    for run in (baseline, candidate):
        run["records"].append({**deepcopy(run["records"][0]), "id": "extra"})

    def forbidden(*args, **kwargs):
        pytest.fail("metric arithmetic must not start above the record bound")

    monkeypatch.setattr(comparison, "score", forbidden)
    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(EvaluationRecordsError, match="too long"):
        compare_runs(baseline, candidate, policy)


def test_capacity_nonconstant_recorded_scalar_interval():
    baseline, candidate, policy = material()
    provenance = {
        "kind": "measurement",
        "unit": "seconds",
        "source": "timer",
        "version": "1",
        "rubric_digest": None,
    }
    for run in (baseline, candidate):
        run["score_provenance"] = {"latency": provenance}
        for i, row in enumerate(run["records"]):
            row["scores"] = {"latency": 1.0 + (i % 7) / 10}
    for i, row in enumerate(candidate["records"]):
        row["scores"]["latency"] += 0.125 if i % 2 else -0.125
    policy["slices"] = []
    policy["metrics"] = [
        {
            "name": "latency",
            "kind": "recorded",
            "score_key": "latency",
            "accepted_provenance": provenance,
            "configuration": {},
            "direction": "lower",
            "unit": "seconds",
            "aggregation": "mean",
            "minimum_count": 12000,
            "maximum_regression": 0.01,
            "maximum_interval_width": 0.02,
        }
    ]
    result = compare_runs(baseline, candidate, policy)
    metric = result["metrics"][0]
    assert result["decision"] == "pass"
    assert metric["count"] == 12000
    assert metric["delta"] == pytest.approx(0)
    assert metric["interval"]["lower"] < 0 < metric["interval"]["upper"]
    assert metric["interval"]["replicates"] == 2048
    assert metric["scoring_assurance"] == "recorded"
