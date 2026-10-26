"""Original native answer provenance and recipe replay through the Rust core."""

import copy
import json

import pytest

from invarlock.evaluation_transaction import evaluate_request_file
from invarlock.judge_measurements import native_workflow
from invarlock.judge_measurements.native_capture import validate_native_capture
from invarlock.judge_measurements.native_recipe import prepare_native_judge
from native.tests.test_judge_evidence_differential import native
from tests.core.test_native_judge_transaction import _completed_measurements, _import


@pytest.fixture
def capture(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(
        native_workflow,
        "collection_preflight",
        lambda *a, **k: {"credential_available": True, "network_calls": 0},
    )
    monkeypatch.setattr(native_workflow, "collect_frozen", _completed_measurements)
    request, key, _ = _import(root)
    result = evaluate_request_file(request, signing_key_path=key)
    return json.loads(
        (root / "judge-workspace/native_capture.json").read_bytes()
    ), result


def test_native_capture_and_recipe(capture):
    value, publication = capture
    expected = validate_native_capture(value)
    status, actual = native("validate-native-capture", value)
    assert status == 0, actual
    assert actual == dict(zip(("baseline", "subject"), expected, strict=True))
    policy = json.dumps(value["recipe"]).encode()
    expected = prepare_native_judge(policy, value["schedule"])
    status, actual = native(
        "prepare-native-judge",
        {"policy_json": policy.decode(), "schedule": value["schedule"]},
    )
    assert status == 0, actual
    assert actual == expected
    status, actual = native(
        "replay-judge", {"evidence_path": publication.payload["evidence"]}
    )
    assert status == 0, actual


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "bad"),
        ("observations", {}),
        ("baseline", {}),
        ("policy_base64", "!"),
        ("subject", None),
    ],
)
def test_native_capture_rejects_malformed(capture, field, value):
    original, _ = capture
    changed = copy.deepcopy(original)
    changed[field] = value
    with pytest.raises(ValueError) as failure:
        validate_native_capture(changed)
    status, actual = native("validate-native-capture", changed)
    assert status != 0
    assert actual["error"] == str(failure.value)


def test_judge_derivation_preserves_original_request_identity(capture):
    import base64

    from invarlock.core.runtime_provider import build_runtime_behavioral_schedule
    from invarlock.evaluation_transaction import _comparison_id
    from invarlock.evidence_pack_contract import (
        RuntimeSideEvidence,
        derive_paired_records,
        schedule_bytes,
        sha256_digest,
    )
    from invarlock.native_backend import runtime_material_config

    value, _ = capture
    schedule = build_runtime_behavioral_schedule(value["schedule"])
    sides = {
        side: RuntimeSideEvidence(
            **{
                name: base64.b64decode(raw, validate=True)
                for name, raw in value[side].items()
            }
        )
        for side in ("baseline", "subject")
    }
    runtimes = {
        side: json.loads(material.provider_receipt)["outer_image_digest"]
        for side, material in sides.items()
    }
    artifacts = {
        side: sha256_digest(material.artifact_identity)
        for side, material in sides.items()
    }
    policy = base64.b64decode(value["policy_base64"], validate=True)
    expected = derive_paired_records(
        schedule=schedule,
        metric="exact_match",
        baseline=sides["baseline"],
        subject=sides["subject"],
        baseline_identity_digest=artifacts["baseline"],
        subject_identity_digest=artifacts["subject"],
        baseline_runtime_digest=runtimes["baseline"],
        subject_runtime_digest=runtimes["subject"],
    )
    config = runtime_material_config(
        normalized_request=value["normalized_request"],
        schedule=schedule,
        policy_bytes=policy,
        baseline_evidence=sides["baseline"],
        subject_evidence=sides["subject"],
    )
    status, actual = native("derive-runtime-transaction", config)
    assert status == 0, actual
    assert actual == {
        "paired_records": expected,
        "artifact_digests": artifacts,
        "runtime_digests": runtimes,
        "comparison_id": _comparison_id(
            value["normalized_request"],
            baseline_identity=sides["baseline"].artifact_identity,
            subject_identity=sides["subject"].artifact_identity,
            schedule_payload=schedule_bytes(schedule),
            policy_payload=policy,
            baseline_runtime_digest=runtimes["baseline"],
            subject_runtime_digest=runtimes["subject"],
            paired_records=expected,
        ),
    }
