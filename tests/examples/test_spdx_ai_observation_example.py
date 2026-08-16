from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from examples.integrations import spdx_ai_observation as example
from invarlock import evaluation_transaction as transaction
from invarlock.evidence_pack_contract import (
    EvidenceObservation,
    build_comparison_report,
    canonical_json_bytes,
    evidence_observation_bytes,
    evidence_observation_errors,
    sha256_digest,
)

ROOT = Path(__file__).resolve().parents[2]


def _fixture_bytes() -> tuple[bytes, bytes]:
    return (
        example.DEFAULT_SOURCE.read_bytes(),
        example.DEFAULT_ARTIFACT_IDENTITY.read_bytes(),
    )


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def test_committed_spdx_observation_rebuilds_exactly() -> None:
    source, identity = _fixture_bytes()

    payload = example.build_observation_payload(source, identity)

    assert not source.endswith(b"\n")
    assert not identity.endswith(b"\n")
    assert example.spdx_canonical_json_bytes(json.loads(source)) == source
    assert canonical_json_bytes(payload) == example.DEFAULT_EXPECTED.read_bytes()
    assert payload["artifact_cross_binding"] == {
        "binding_basis": "spdx.ai_AIPackage.verifiedUsing.Hash.sha256",
        "invarlock_artifact_content_digest": (
            "sha256:684888c0ebb17f374298b65ee2807526c066094c7014e8a778ce62e346331b90"
        ),
        "invarlock_artifact_identity_digest": (
            "sha256:7dd5332927812f7fa60c117ac1900285352e78e1ff99fc88282499963c0a7fce"
        ),
        "invarlock_artifact_name": "example-model-q5_k_m.gguf",
        "spdx_package_id": "https://example.invalid/spdx/model/example-q5-k-m",
        "status": "matched",
    }
    validation = payload["validation"]
    assert isinstance(validation, dict)
    assert validation["example_subset_checks"]["status"] == "passed"
    assert validation["official_json_schema"]["status"] == "not_evaluated"
    assert validation["owl_shacl_semantics"]["status"] == "not_evaluated"
    assert validation["spdx_profile_conformance"]["status"] == "not_evaluated"
    assert example.main(["--check"]) == 0


def test_mapper_rejects_reformatting_and_artifact_digest_mismatch() -> None:
    source, identity = _fixture_bytes()
    with pytest.raises(example.SpdxObservationError, match="no final line feed"):
        example.load_spdx_document(source + b"\n")

    identity_payload = json.loads(identity)
    identity_payload["sha256"] = "f" * 64
    changed_identity = example.spdx_canonical_json_bytes(identity_payload)
    with pytest.raises(example.SpdxObservationError, match="does not match"):
        example.build_observation_payload(source, changed_identity)


def test_spdx_canonicalizer_rejects_non_ascii_names_and_floats() -> None:
    with pytest.raises(example.SpdxObservationError, match="canonical ASCII"):
        example.spdx_canonical_json_bytes({"mødel": "value"})
    with pytest.raises(example.SpdxObservationError, match="canonical JSON subset"):
        example.spdx_canonical_json_bytes({"metric": 1.0})


def test_mapper_rejects_incomplete_ai_license_relationships() -> None:
    source, _identity = _fixture_bytes()
    document = json.loads(source)
    document["@graph"] = [
        item
        for item in document["@graph"]
        if item.get("relationshipType") != "hasDeclaredLicense"
    ]

    with pytest.raises(
        example.SpdxObservationError, match="declared and one concluded"
    ):
        example.load_spdx_document(example.spdx_canonical_json_bytes(document))


def test_spdx_validation_status_is_distinct_from_envelope_integrity() -> None:
    source, identity = _fixture_bytes()
    payload_bytes = canonical_json_bytes(
        example.build_observation_payload(source, identity)
    )
    observation = EvidenceObservation(
        observation_id="subject-spdx-ai",
        kind="spdx.ai",
        scope="subject",
        payload=payload_bytes,
    )
    bindings = {
        "comparison_id": "spdx-observation-comparison",
        "schedule_digest": _digest("a"),
        "policy_digest": _digest("b"),
        "artifact_digests": {
            "baseline": _digest("c"),
            "subject": sha256_digest(identity),
        },
    }
    encoded = evidence_observation_bytes(observation, **bindings)
    envelope = json.loads(encoded)
    reference = {
        "path": "observations/subject-spdx-ai.json",
        "digest": sha256_digest(encoded),
        "kind": "spdx.ai",
        "scope": "subject",
    }

    assert envelope["authority"] == "observation"
    assert (
        envelope["payload"]["artifact_cross_binding"][
            "invarlock_artifact_identity_digest"
        ]
        == envelope["bindings"]["artifact_digests"]["subject"]
    )
    assert (
        envelope["payload"]["validation"]["spdx_profile_conformance"]["status"]
        == "not_evaluated"
    )
    assert (
        evidence_observation_errors(
            envelope,
            observation_id="subject-spdx-ai",
            reference=reference,
            **bindings,
        )
        == []
    )

    rebound = copy.deepcopy(envelope)
    rebound["authority"] = "acceptance"
    errors = evidence_observation_errors(
        rebound,
        observation_id="subject-spdx-ai",
        reference=reference,
        **bindings,
    )
    assert any("authority" in error for error in errors)
    assert (
        rebound["payload"]["validation"]["spdx_profile_conformance"]["status"]
        == "not_evaluated"
    )


def test_observation_changes_transaction_identity_but_not_decision_fields() -> None:
    source, identity = _fixture_bytes()
    payload = canonical_json_bytes(example.build_observation_payload(source, identity))
    paired_records = {
        "format": "invarlock/paired-records-v1",
        "metric": "exact_match",
        "schedule_sha256": "e" * 64,
        "records": [
            {
                "record_id": "one",
                "baseline": {"score": 1.0},
                "subject": {"score": 1.0},
            },
            {
                "record_id": "two",
                "baseline": {"score": 1.0},
                "subject": {"score": 0.0},
            },
            {
                "record_id": "three",
                "baseline": {"score": 0.0},
                "subject": {"score": 1.0},
            },
            {
                "record_id": "four",
                "baseline": {"score": 0.0},
                "subject": {"score": 1.0},
            },
        ],
    }
    policy = {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": -100.0}}}}
    normalized_without: dict[str, object] = {"format": "invarlock/example-request-v1"}
    normalized_with = {
        **normalized_without,
        "observations": [
            {
                "id": "subject-spdx-ai",
                "kind": "spdx.ai",
                "payload_digest": sha256_digest(payload),
                "scope": "subject",
            }
        ],
    }
    comparison_inputs = {
        "baseline_identity": b"baseline-identity",
        "subject_identity": identity,
        "schedule_payload": canonical_json_bytes({"schedule": "same"}),
        "policy_payload": canonical_json_bytes(policy),
        "baseline_runtime_digest": _digest("1"),
        "subject_runtime_digest": _digest("2"),
        "paired_records": paired_records,
    }
    without_id = transaction._comparison_id(normalized_without, **comparison_inputs)
    with_id = transaction._comparison_id(normalized_with, **comparison_inputs)
    assert without_id != with_id

    report_without = build_comparison_report(
        comparison_id=without_id,
        paired_records=paired_records,
        policy=policy,
        policy_digest=_digest("b"),
    )
    report_with = build_comparison_report(
        comparison_id=with_id,
        paired_records=paired_records,
        policy=policy,
        policy_digest=_digest("b"),
    )
    assert report_without != report_with
    for field in (
        "metric",
        "record_count",
        "baseline",
        "subject",
        "comparison",
        "uncertainty",
        "paired_binary",
        "policy_digest",
        "verdict",
    ):
        assert report_without[field] == report_with[field]


def test_example_is_listed_and_has_a_cpu_only_make_target() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("example-spdx-ai-observation:", 1)[1].split("\n\n", 1)[0]
    integrations = (ROOT / "examples/integrations/README.md").read_text(
        encoding="utf-8"
    )

    assert "examples.integrations.spdx_ai_observation --check" in target
    assert "uv run" not in target
    assert "SPDX 3.0.1 AI observation" in integrations
