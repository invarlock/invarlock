"""The bounded full comparison survives signed replay without splitting its sample."""

import json
import subprocess
import sys
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.pipeline import (
    PipelineError,
    case_set_digest,
    compare_runs,
    comparison,
    create_evidence,
    make_run,
)
from invarlock.pipeline.contracts import digest


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
        "format": "invarlock/pipeline-policy-v1",
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
                "format": "invarlock/pipeline-case-set-v1",
                "cases": [
                    {k: row[k] for k in ("id", "input", "expected", "metadata")}
                    for row in baseline["records"]
                ],
            }
        ),
    }
    return baseline, candidate, policy


def test_full_capacity_signed_independent_recipient(tmp_path):
    baseline, candidate, policy = material()
    key = Ed25519PrivateKey.generate()
    evidence = create_evidence(baseline, candidate, policy, key)
    assert evidence["comparison"]["decision"] == "pass"
    assert [m["count"] for m in evidence["comparison"]["metrics"]] == [
        12000,
        6000,
        6000,
    ]
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    (tmp_path / "public.pem").write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    # The recipient receives the policy and run digests independently of evidence.
    script = """
import json, sys
from pathlib import Path
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from invarlock.pipeline import verify_evidence
root = Path(sys.argv[1])
result = verify_evidence(
    json.loads((root / 'evidence.json').read_text()),
    public_key=load_pem_public_key((root / 'public.pem').read_bytes()),
    expected_baseline=sys.argv[2], expected_candidate=sys.argv[3],
    policy=json.loads((root / 'policy.json').read_text()),
)
assert result['decision'] == 'pass'
print(json.dumps(result))
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
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert json.loads(received.stdout) == evidence["comparison"]


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
    with pytest.raises(PipelineError, match="planned case set"):
        compare_runs(baseline, candidate, policy)


def test_capacity_plus_one_rejected_before_metric_arithmetic(monkeypatch):
    baseline, candidate, policy = material()
    for run in (baseline, candidate):
        run["records"].append({**deepcopy(run["records"][0]), "id": "extra"})

    def forbidden(*args, **kwargs):
        pytest.fail("metric arithmetic must not start above the record bound")

    monkeypatch.setattr(comparison, "score", forbidden)
    monkeypatch.setattr(comparison, "_interval", forbidden)
    with pytest.raises(PipelineError, match="too long"):
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
