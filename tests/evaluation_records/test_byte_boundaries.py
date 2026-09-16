"""Canonical artifact limits and physical input limits are separate boundaries."""

import json
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock import captured_contracts
from invarlock.captured_contracts import CapturedContractError
from invarlock.evaluation_record_contracts.contracts import digest
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_verification import EvidenceVerificationError
from tests._evaluation_support import (
    EvaluationRecordsError,
    build_pack,
    case_set_digest,
    compare_runs,
    comparison,
    contracts,
    make_run,
    pack_json,
    replay_pack,
)


@pytest.fixture
def artifacts():
    baseline = make_run(
        [
            {
                "id": f"case-{i}",
                "input": {"question": "Español 日本語 🧪 e\u0301"},
                "expected": answer,
                "output": answer,
            }
            for i, answer in enumerate(("yes", "no"))
        ],
        source={"name": "byte boundary", "version": "1"},
        run_id="baseline",
        artifact_digest="sha256:" + "a" * 64,
    )
    candidate = deepcopy(baseline)
    candidate["run_id"] = "candidate"
    cases = {
        "format": "invarlock/evaluation-case-set-v1",
        "cases": [
            {key: row[key] for key in ("id", "input", "expected", "metadata")}
            for row in baseline["records"]
        ],
    }
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "expected_case_set_digest": case_set_digest(cases),
        "metrics": [
            {
                "name": "exact-é-🧪",
                "kind": "exact_match",
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 2,
                "maximum_regression": 1.0,
                "maximum_interval_width": 2.0,
            }
        ],
        "slices": [],
    }
    key = Ed25519PrivateKey.generate()
    evidence = build_pack(baseline, candidate, policy, key)
    return {
        "run": baseline,
        "candidate": candidate,
        "case_set": cases,
        "policy": policy,
        "comparison": pack_json(evidence, "report"),
        "pack": evidence,
        "key": key,
    }


@pytest.mark.parametrize("kind", ["run", "case_set", "policy", "comparison", "pack"])
def test_canonical_utf8_exact_limit_and_one_byte_over(artifacts, monkeypatch, kind):
    value = artifacts[kind]
    wire = (
        b"".join(value.files.values())
        if kind == "pack"
        else canonical_json_bytes(value)
    )
    assert wire.endswith(b"\n")
    assert len(wire) > len(wire.decode("utf-8"))
    if kind == "pack":
        monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", 0)
        monkeypatch.setattr(captured_contracts, "TOTAL_LIMIT", len(wire))
        captured_contracts.check_sizes(value.files)
        monkeypatch.setattr(captured_contracts, "TOTAL_LIMIT", len(wire) - 1)
        with pytest.raises(
            CapturedContractError, match="inventory exceeds total byte limit"
        ):
            captured_contracts.check_sizes(value.files)
    else:
        monkeypatch.setattr(captured_contracts, "TOTAL_LIMIT", 0)
        monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", len(wire))
        contracts.validate(value, kind)
        # The same valid artifact is exactly one byte above this smaller allowance.
        monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", len(wire) - 1)
        with pytest.raises(
            EvaluationRecordsError, match=rf"{kind} exceeds the .* byte limit"
        ):
            contracts.validate(value, kind)


def test_physical_read_limit_is_not_normalized_canonical_size(
    artifacts, monkeypatch, tmp_path
):
    value = artifacts["run"]
    canonical_size = len(canonical_json_bytes(value))
    physical = (json.dumps(value, ensure_ascii=True, indent=4) + "\n").encode()
    assert len(physical) > canonical_size
    path = tmp_path / "run.json"
    path.write_bytes(physical)

    monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", canonical_size)
    loaded = contracts.read_json(path, max_bytes=len(physical))
    assert loaded == value
    contracts.validate(loaded, "run")

    def forbidden(*args, **kwargs):
        pytest.fail("an oversized physical file must be rejected before JSON parsing")

    monkeypatch.setattr(contracts, "parse_json_bytes", forbidden)
    with pytest.raises(EvaluationRecordsError, match="exceeds the .*byte size limit"):
        contracts.read_json(path, max_bytes=len(physical) - 1)


def test_signed_evidence_at_exact_limit_preserves_pinned_replay(artifacts, monkeypatch):
    baseline, candidate, policy = (
        artifacts[name] for name in ("run", "candidate", "policy")
    )
    evidence = artifacts["pack"]
    pins = {
        "expected_baseline_run": digest(baseline),
        "expected_subject_run": digest(candidate),
    }
    original_wire = dict(evidence.files)
    total = sum(map(len, original_wire.values()))
    monkeypatch.setattr(captured_contracts, "TOTAL_LIMIT", total)

    recreated = build_pack(baseline, candidate, policy, artifacts["key"])
    assert recreated.files == original_wire
    assert (
        replay_pack(
            recreated,
            public_key=artifacts["key"].public_key(),
            policy=policy,
            **pins,
        )
        == artifacts["comparison"]
    )
    with pytest.raises(EvidenceVerificationError, match="subject.*expected anchor"):
        replay_pack(
            recreated,
            public_key=artifacts["key"].public_key(),
            policy=policy,
            **{**pins, "expected_subject_run": "sha256:" + "0" * 64},
        )

    def forbidden(*args, **kwargs):
        pytest.fail("oversized evidence must be rejected before any metric replay")

    monkeypatch.setattr(comparison, "_metric_result", forbidden)
    monkeypatch.setattr(captured_contracts, "TOTAL_LIMIT", total - 1)
    with pytest.raises(
        EvidenceVerificationError, match="inventory exceeds total byte limit"
    ):
        replay_pack(
            recreated,
            public_key=artifacts["key"].public_key(),
            policy=policy,
            **pins,
        )
    assert recreated.files == original_wire


@pytest.mark.parametrize("oversized_part", ["run", "candidate", "policy"])
def test_malformed_oversized_input_rejected_before_scoring(
    artifacts, monkeypatch, oversized_part
):
    limit = max(
        len(canonical_json_bytes(artifacts[name]))
        for name in ("run", "candidate", "policy", "case_set", "comparison")
    )
    malformed = deepcopy(artifacts[oversized_part])
    malformed["format"] = "wrong-format"
    malformed["oversized_unknown_field"] = "🧪" * limit
    inputs = {name: artifacts[name] for name in ("run", "candidate", "policy")}
    inputs[oversized_part] = malformed

    def forbidden(*args, **kwargs):
        pytest.fail("oversized malformed input must not reach metric arithmetic")

    monkeypatch.setattr(comparison, "_metric_result", forbidden)
    monkeypatch.setattr(contracts, "MAX_INPUT_BYTES", limit)
    with pytest.raises(EvaluationRecordsError, match="exceeds the .* byte limit"):
        compare_runs(inputs["run"], inputs["candidate"], inputs["policy"])
