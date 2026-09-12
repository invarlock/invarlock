"""Reject malformed and oversized native provenance before replaying answers."""

import base64
import copy
import hashlib
import json
from dataclasses import replace

import pytest

from invarlock.evidence_pack_contract import (
    EvidenceObservation,
    RuntimeSideEvidence,
    canonical_json_bytes,
)
from invarlock.judge_measurements import native_capture as capture_contract
from tests.judge_measurements.test_native_capture import _capture


@pytest.fixture
def captured(tmp_path):
    return _capture(tmp_path)


def _create_again(capture, *, policy=None, baseline=None, observations=()):
    sides = {
        role: RuntimeSideEvidence(
            **{name: base64.b64decode(value) for name, value in capture[role].items()}
        )
        for role in ("baseline", "subject")
    }
    return capture_contract.create_native_capture(
        capture["normalized_request"],
        capture["schedule"],
        base64.b64decode(capture["policy_base64"]) if policy is None else policy,
        capture["recipe"],
        sides["baseline"] if baseline is None else baseline,
        sides["subject"],
        observations=observations,
    )


@pytest.mark.parametrize(
    "value, message",
    [
        (None, "exceeds its byte limit or is invalid"),
        ("not-valid-base64", "is not valid base64"),
        ("e31=", "not bounded canonical base64"),
    ],
)
def test_policy_encoding_must_be_canonical_base64(captured, value, message):
    captured["policy_base64"] = value
    with pytest.raises(ValueError, match=message):
        capture_contract.validate_native_capture(captured)


def test_decoded_policy_cannot_exceed_the_byte_limit(captured, monkeypatch):
    captured["policy_base64"] = base64.b64encode(b"{}").decode()
    monkeypatch.setattr(capture_contract, "_POLICY_MAX_BYTES", 1)
    with pytest.raises(ValueError, match="not bounded canonical base64"):
        capture_contract.validate_native_capture(captured)


def test_encoded_policy_is_bounded_before_decoding(captured, monkeypatch):
    monkeypatch.setattr(capture_contract, "_POLICY_MAX_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds its byte limit"):
        capture_contract.validate_native_capture(captured)


@pytest.mark.parametrize(
    "limit, message",
    [
        ("_POLICY_MAX_BYTES", "policy exceeds"),
        ("MAX_EVIDENCE_BYTES", "side file exceeds"),
        ("NATIVE_CAPTURE_MAX_BYTES", "aggregate byte limit"),
    ],
)
def test_capture_creation_enforces_each_resource_boundary(
    captured, monkeypatch, limit, message
):
    monkeypatch.setattr(capture_contract, limit, 1)
    with pytest.raises(ValueError, match=message):
        _create_again(captured)


def test_capture_creation_requires_exact_side_bytes(captured):
    baseline = RuntimeSideEvidence(
        **{
            name: base64.b64decode(value)
            for name, value in captured["baseline"].items()
        }
    )
    with pytest.raises(
        ValueError, match="side file exceeds its byte limit or is invalid"
    ):
        _create_again(captured, baseline=replace(baseline, runtime_config="{}"))


def test_capture_creation_bounds_observation_count(captured):
    observation = EvidenceObservation("note", "comparison", "diagnostic.note", b"{}\n")
    with pytest.raises(ValueError, match="observation count limit"):
        _create_again(
            captured,
            observations=(observation,) * (capture_contract.MAX_OBSERVATIONS + 1),
        )


def test_capture_replay_enforces_total_retained_size(captured, monkeypatch):
    monkeypatch.setattr(
        capture_contract,
        "NATIVE_CAPTURE_MAX_BYTES",
        len(canonical_json_bytes(captured, newline=False)) - 1,
    )
    with pytest.raises(ValueError, match="aggregate byte limit"):
        capture_contract.validate_native_capture(captured)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("format", "invarlock/unsupported-capture-v1", "format is unsupported"),
        ("normalized_request", None, "normalized request must be an object"),
        ("normalized_request", {}, "normalized request schema failed"),
        ("observations", None, "observations must be a bounded array"),
        ("observations", [{}], "observation fields are invalid"),
        ("observations", [None], "observation fields are invalid"),
        ("baseline", {}, "retain all six original files"),
        ("baseline", None, "retain all six original files"),
    ],
)
def test_capture_replay_rejects_open_or_missing_fields(captured, field, value, message):
    captured[field] = value
    with pytest.raises(ValueError, match=message):
        capture_contract.validate_native_capture(captured)


def test_capture_replay_bounds_retained_observation_count(captured):
    captured["observations"] = [{}] * (capture_contract.MAX_OBSERVATIONS + 1)
    with pytest.raises(ValueError, match="observations must be a bounded array"):
        capture_contract.validate_native_capture(captured)


def test_request_dataset_selection_must_match_native_schedule(captured):
    captured["normalized_request"]["comparison"]["dataset"]["selected_record_count"] = 1
    with pytest.raises(ValueError, match="selected_record_count does not match"):
        capture_contract.validate_native_capture(captured)


def test_native_capture_rejects_multiple_authenticated_text_parts(captured):
    row = captured["schedule"]["records"][0]
    part = copy.deepcopy(row["input_parts"][0])
    part["role"] = "context"
    row["input_parts"].append(part)
    row["input_sha256"] = hashlib.sha256(
        canonical_json_bytes(row["input_parts"], newline=False)
    ).hexdigest()
    with pytest.raises(ValueError, match="exactly one text input part"):
        capture_contract.validate_native_capture(captured)


def test_capture_cannot_be_transplanted_to_an_exact_match_request(captured):
    comparison = captured["normalized_request"]["comparison"]
    comparison["metric"] = "exact_match"
    del comparison["judge"]
    with pytest.raises(ValueError, match="requires a judge metric request"):
        capture_contract.validate_native_capture(captured)


def test_capture_requires_canonical_policy_reference(captured):
    captured["normalized_request"]["comparison"]["policy"] = "other-policy.json"
    with pytest.raises(ValueError, match="bind the canonical policy"):
        capture_contract.validate_native_capture(captured)


def test_capture_requires_strict_outer_runtime_identity(captured):
    receipt = json.loads(base64.b64decode(captured["baseline"]["provider_receipt"]))
    receipt["outer_image_digest"] = None
    captured["baseline"]["provider_receipt"] = base64.b64encode(
        canonical_json_bytes(receipt)
    ).decode()
    with pytest.raises(ValueError, match="lacks strict runtime identity"):
        capture_contract.validate_native_capture(captured)


def test_capture_model_label_must_match_authenticated_artifact(captured):
    captured["normalized_request"]["comparison"]["baseline"]["artifact"]["model_id"] = (
        "substituted-model"
    )
    with pytest.raises(ValueError, match="model identity differs from request"):
        capture_contract.validate_native_capture(captured)
