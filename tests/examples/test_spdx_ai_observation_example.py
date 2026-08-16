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


def _spdx_document() -> dict[str, object]:
    source, _identity = _fixture_bytes()
    document = json.loads(source)
    assert isinstance(document, dict)
    return document


def _graph_item(document: dict[str, object], type_name: str) -> dict[str, object]:
    graph = document["@graph"]
    assert isinstance(graph, list)
    item = next(
        candidate
        for candidate in graph
        if isinstance(candidate, dict) and candidate.get("type") == type_name
    )
    return item


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


def test_spdx_canonicalizer_wraps_encoder_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_json(*_args: object, **_kwargs: object) -> str:
        raise ValueError("encoder rejected value")

    monkeypatch.setattr(example.json, "dumps", reject_json)
    with pytest.raises(example.SpdxObservationError, match="not canonical JSON"):
        example.spdx_canonical_json_bytes({"valid": True})


def test_spdx_subset_helpers_reject_invalid_shapes() -> None:
    with pytest.raises(example.SpdxObservationError, match="must be an object"):
        example._object([], label="value")
    with pytest.raises(example.SpdxObservationError, match="non-empty array"):
        example._object_array([], label="array")
    with pytest.raises(example.SpdxObservationError, match="element example limit"):
        example._object_array([{}] * 65, label="array")
    with pytest.raises(example.SpdxObservationError, match=r"array\[0\] must"):
        example._object_array([[]], label="array")
    with pytest.raises(example.SpdxObservationError, match="string array"):
        example._string_array([""], label="strings")
    with pytest.raises(example.SpdxObservationError, match="exactly one Thing"):
        example._one_by_type([], "Thing")
    with pytest.raises(example.SpdxObservationError, match="non-empty spdxId"):
        example._element_id({}, label="element")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("extra_root_field", "only @context and @graph"),
        ("wrong_context", "context is not pinned"),
        ("duplicate_id", "duplicate ID"),
        ("duplicate_creation", "exactly one CreationInfo"),
        ("wrong_spec_version", "specVersion"),
        ("invalid_created_by", "createdBy"),
        ("invalid_created", "must be a timestamp"),
        ("missing_profile", "missing the example profiles"),
        ("missing_package_field", "missing required example fields"),
        ("wrong_creation_binding", "creationInfo binding"),
        ("wrong_primary_purpose", "primary purpose"),
        ("invalid_model_types", "ai_typeOfModel"),
        ("invalid_supplier_type", "suppliedBy must"),
        ("missing_supplier", "agent is missing"),
        ("multiple_hashes", "exactly one example integrity"),
        ("invalid_hash_fields", "Hash fields"),
        ("invalid_hash", "lowercase SHA-256"),
        ("wrong_root", "does not root"),
        ("wrong_relationship_source", "declared and one concluded"),
        ("irrelevant_relationship_type", "declared and one concluded"),
        ("wrong_license_target", "must point to NoAssertion"),
        ("unlisted_relationship", "not listed in document elements"),
    ],
)
def test_spdx_subset_rejects_bounded_structural_mutations(
    case: str, message: str
) -> None:
    document = _spdx_document()
    graph = document["@graph"]
    assert isinstance(graph, list)
    creation = _graph_item(document, "CreationInfo")
    spdx_document = _graph_item(document, "SpdxDocument")
    package = _graph_item(document, "ai_AIPackage")
    relationships = [
        item
        for item in graph
        if isinstance(item, dict) and item.get("type") == "Relationship"
    ]

    if case == "extra_root_field":
        document["unexpected"] = True
    elif case == "wrong_context":
        document["@context"] = "https://example.invalid/context"
    elif case == "duplicate_id":
        package["spdxId"] = _graph_item(document, "SoftwareAgent")["spdxId"]
    elif case == "duplicate_creation":
        duplicate = copy.deepcopy(creation)
        duplicate["@id"] = "_:other-creation-info"
        graph.append(duplicate)
    elif case == "wrong_spec_version":
        creation["specVersion"] = "3.0.0"
    elif case == "invalid_created_by":
        creation["createdBy"] = "agent"
    elif case == "invalid_created":
        creation["created"] = 1
    elif case == "missing_profile":
        spdx_document["profileConformance"] = ["core"]
    elif case == "missing_package_field":
        package.pop("name")
    elif case == "wrong_creation_binding":
        package["creationInfo"] = "_:other"
    elif case == "wrong_primary_purpose":
        package["software_primaryPurpose"] = "application"
    elif case == "invalid_model_types":
        package["ai_typeOfModel"] = []
    elif case == "invalid_supplier_type":
        package["suppliedBy"] = 1
    elif case == "missing_supplier":
        package["suppliedBy"] = "https://example.invalid/spdx/agent/missing"
    elif case == "multiple_hashes":
        hashes = package["verifiedUsing"]
        assert isinstance(hashes, list)
        hashes.append(copy.deepcopy(hashes[0]))
    elif case == "invalid_hash_fields":
        hashes = package["verifiedUsing"]
        assert isinstance(hashes, list) and isinstance(hashes[0], dict)
        hashes[0]["unexpected"] = True
    elif case == "invalid_hash":
        hashes = package["verifiedUsing"]
        assert isinstance(hashes, list) and isinstance(hashes[0], dict)
        hashes[0]["hashValue"] = "A" * 64
    elif case == "wrong_root":
        spdx_document["rootElement"] = ["https://example.invalid/spdx/model/other"]
    elif case == "wrong_relationship_source":
        relationships[0]["from"] = "https://example.invalid/spdx/model/other"
    elif case == "irrelevant_relationship_type":
        relationships[0]["relationshipType"] = "describes"
    elif case == "wrong_license_target":
        relationships[0]["to"] = ["https://example.invalid/license"]
    elif case == "unlisted_relationship":
        elements = spdx_document["element"]
        assert isinstance(elements, list)
        elements.remove(relationships[0]["spdxId"])
    else:  # pragma: no cover - the parameter list is closed above
        raise AssertionError(case)

    encoded = example.spdx_canonical_json_bytes(document)
    with pytest.raises(example.SpdxObservationError, match=message):
        example.load_spdx_document(encoded)


def test_mapper_wraps_strict_json_and_identity_errors() -> None:
    source, identity = _fixture_bytes()
    document, _checks = example.load_spdx_document(source)

    with pytest.raises(example.SpdxObservationError, match="valid JSON"):
        example.load_spdx_document(b"{")
    with pytest.raises(example.SpdxObservationError, match="valid JSON"):
        example.build_observation_payload(source, b"{")

    hf_identity = (
        ROOT / "examples/import/baseline/model-artifact.identity.json"
    ).read_bytes()
    with pytest.raises(example.SpdxObservationError, match="one GGUF artifact"):
        example._gguf_cross_binding(document, hf_identity)

    pretty_identity = json.dumps(json.loads(identity), indent=2).encode("utf-8")
    with pytest.raises(example.SpdxObservationError, match="canonical JSON bytes"):
        example._gguf_cross_binding(document, pretty_identity)


def test_cli_reports_output_mismatch_and_read_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert example.main([]) == 0
    assert (
        json.loads(capsys.readouterr().out)["format"]
        == example.OBSERVATION_PAYLOAD_FORMAT
    )

    expected = tmp_path / "expected.json"
    expected.write_bytes(b"{}\n")
    assert example.main(["--check", "--expected", str(expected)]) == 2
    assert "does not match rebuilt bytes" in capsys.readouterr().err

    missing = tmp_path / "missing.json"
    assert example.main(["--source", str(missing)]) == 2
    assert "SPDX source document is unavailable" in capsys.readouterr().err


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
