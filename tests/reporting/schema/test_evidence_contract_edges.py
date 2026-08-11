from __future__ import annotations

import hashlib
import math

import pytest

from invarlock import evidence_pack_contract as contract
from invarlock.core.runtime_provider import RuntimeScoringRecord
from invarlock.evidence_pack_contract import (
    COMPARISON_REPORT_FORMAT,
    COMPARISON_REPORT_FORMAT_V2,
    EVIDENCE_INPUT_IDENTITY_FORMAT,
    PAIRED_RECORDS_FORMAT,
    EvidenceObservation,
    EvidencePackError,
    InputIdentity,
    build_comparison_report,
    canonical_json_bytes,
    identity_payload,
    normalize_digest,
    parse_json_object,
    request_metric,
)


def _digest(marker: str = "a") -> str:
    return "sha256:" + marker * 64


def _pairs(
    *, metric: str = "exact_match", baseline: float = 1.0, subject: float = 0.0
) -> dict[str, object]:
    second_baseline = baseline if metric == "normalized_nll_per_utf8_byte" else 1.0
    second_subject = subject if metric == "normalized_nll_per_utf8_byte" else 1.0
    payload: dict[str, object] = {
        "format": PAIRED_RECORDS_FORMAT,
        "metric": metric,
        "schedule_sha256": "0" * 64,
        "records": [
            {
                "record_id": "one",
                "baseline": {"score": baseline},
                "subject": {"score": subject},
            },
            {
                "record_id": "two",
                "baseline": {"score": second_baseline},
                "subject": {"score": second_subject},
            },
        ],
    }
    if metric == "normalized_nll_per_utf8_byte":
        payload["derived_measurements"] = {
            "perplexity_ratio": {
                "status": "unavailable",
                "basis": "authenticated_target_likelihood",
                "method": "target_token_weighted_perplexity_ratio_v1",
                "reason": "target_token_counts_unavailable",
            }
        }
    return payload


def test_canonical_json_and_digest_normalization_are_closed() -> None:
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    assert canonical_json_bytes({"a": 1}, newline=False) == b'{"a":1}'
    assert normalize_digest(f"  {_digest().upper()}  ", label="artifact") == _digest()

    for value in (float("nan"), {"not", "json"}):
        with pytest.raises(EvidencePackError, match="canonical JSON"):
            canonical_json_bytes(value)
    for value in ("", "a" * 64, "sha256:" + "g" * 64):
        with pytest.raises(EvidencePackError, match="sha256"):
            normalize_digest(value, label="artifact")


def test_input_identity_emits_only_authenticated_optional_metadata() -> None:
    payload = identity_payload(
        "baseline",
        InputIdentity(
            _digest(),
            locator="  artifact://model  ",
            media_type="  application/vnd.model  ",
        ),
    )

    assert payload == {
        "format": EVIDENCE_INPUT_IDENTITY_FORMAT,
        "role": "baseline",
        "digest": _digest(),
        "locator": "artifact://model",
        "media_type": "application/vnd.model",
    }
    assert identity_payload("subject", InputIdentity(_digest("b"))) == {
        "format": EVIDENCE_INPUT_IDENTITY_FORMAT,
        "role": "subject",
        "digest": _digest("b"),
    }


@pytest.mark.parametrize(
    ("role", "identity", "message"),
    [
        ("unknown", InputIdentity(_digest()), "unsupported input role"),
        ("baseline", InputIdentity(_digest(), locator=" "), "locator is invalid"),
        (
            "baseline",
            InputIdentity(_digest(), locator="x" * 4097),
            "locator is invalid",
        ),
        (
            "baseline",
            InputIdentity(_digest(), media_type=" "),
            "media_type is invalid",
        ),
        (
            "baseline",
            InputIdentity(_digest(), media_type="x" * 256),
            "media_type is invalid",
        ),
    ],
)
def test_input_identity_rejects_open_or_ambiguous_metadata(
    role: str, identity: InputIdentity, message: str
) -> None:
    with pytest.raises(EvidencePackError, match=message):
        identity_payload(role, identity)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("exact_match", "exact_match"),
        ("normalized_nll_per_utf8_byte", "normalized_nll_per_utf8_byte"),
    ],
)
def test_request_metric_accepts_only_canonical_metrics(
    metric: str, expected: str
) -> None:
    assert request_metric({"comparison": {"metric": metric}}) == expected


def test_request_metric_explains_retired_and_unknown_metrics() -> None:
    with pytest.raises(EvidencePackError, match="not yet canonical"):
        request_metric({"comparison": {"metric": "multiple_choice_accuracy"}})
    for request in ({}, {"comparison": []}, {"comparison": {"metric": "other"}}):
        with pytest.raises(EvidencePackError, match="unsupported"):
            request_metric(request)


def test_parse_json_object_rejects_ambiguous_and_nonobject_payloads() -> None:
    assert parse_json_object(b'{"ok":true}', label="report") == {"ok": True}
    with pytest.raises(EvidencePackError, match="duplicate key"):
        parse_json_object(b'{"ok":true,"ok":false}', label="report")
    with pytest.raises(EvidencePackError, match="JSON object"):
        parse_json_object(b"[]", label="report")


def test_exact_match_report_replays_policy_pass_and_fail() -> None:
    policy = {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": -100}}}}

    report = build_comparison_report(
        comparison_id="comparison-1",
        paired_records=_pairs(),
        policy=policy,
        policy_digest=_digest(),
    )

    assert report["format"] == COMPARISON_REPORT_FORMAT
    assert report["comparison"] == {
        "kind": "exact_match_delta_pp",
        "value": -50.0,
        "minimum": -100.0,
    }
    assert report["uncertainty"] == {
        "method": "newcombe_hybrid_score_paired_v2",
        "scope": "paired_binary_outcomes",
        "interval_mass": 0.95,
        "lower": pytest.approx(-90.54687942657694),
        "upper": pytest.approx(27.257278511318162),
    }
    assert report["verdict"] == "pass"

    failing_policy = {
        "resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": -90}}}
    }
    assert (
        build_comparison_report(
            comparison_id="comparison-1",
            paired_records=_pairs(),
            policy=failing_policy,
            policy_digest=_digest(),
        )["verdict"]
        == "fail"
    )


def test_normalized_nll_report_replays_ratio_policy() -> None:
    report = build_comparison_report(
        comparison_id="comparison-1",
        paired_records=_pairs(
            metric="normalized_nll_per_utf8_byte", baseline=2.0, subject=2.2
        ),
        policy={
            "resolved_policy": {
                "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.2}}
            }
        },
        policy_digest=_digest(),
    )

    assert report["comparison"] == {
        "kind": "normalized_nll_ratio",
        "value": pytest.approx(1.1),
        "maximum": 1.2,
    }
    assert report["verdict"] == "pass"


def test_sample_qualification_fails_closed_on_count_or_precision() -> None:
    report = build_comparison_report(
        comparison_id="comparison-1",
        paired_records=_pairs(),
        policy={
            "resolved_policy": {
                "metrics": {
                    "exact_match": {
                        "delta_min_pp": -100.0,
                        "minimum_record_count": 400,
                        "maximum_interval_width_pp": 10.0,
                    }
                }
            }
        },
        policy_digest=_digest(),
    )

    assert report["sample_qualification"] == {
        "record_count": {"minimum": 400, "observed": 2, "passed": False},
        "interval_width": {
            "maximum": 10.0,
            "observed": pytest.approx(117.804158),
            "unit": "percentage_points",
            "passed": False,
        },
        "passed": False,
    }
    assert report["verdict"] == "fail"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minimum_record_count", True, "minimum_record_count"),
        ("minimum_record_count", 0, "minimum_record_count"),
        ("minimum_record_count", 10_001, "minimum_record_count"),
        ("maximum_interval_width_pp", 0.0, "maximum_interval_width_pp"),
        ("maximum_interval_width_pp", 201.0, "maximum_interval_width_pp"),
        ("minimum_side_accuracy", True, "minimum_side_accuracy"),
        ("minimum_side_accuracy", -0.1, "minimum_side_accuracy"),
        ("minimum_side_accuracy", 1.1, "minimum_side_accuracy"),
    ],
)
def test_sample_qualification_policy_rejects_invalid_limits(
    field: str, value: object, message: str
) -> None:
    selected: dict[str, object] = {"delta_min_pp": -10.0, field: value}
    with pytest.raises(EvidencePackError, match=message):
        build_comparison_report(
            comparison_id="comparison-1",
            paired_records=_pairs(),
            policy={"resolved_policy": {"metrics": {"exact_match": selected}}},
            policy_digest=_digest(),
        )


def test_signed_side_accuracy_policy_controls_verdict_and_report() -> None:
    policy = {
        "resolved_policy": {
            "metrics": {
                "exact_match": {
                    "delta_min_pp": -100.0,
                    "minimum_side_accuracy": 0.75,
                }
            }
        }
    }
    report = build_comparison_report(
        comparison_id="comparison-1",
        paired_records=_pairs(baseline=1.0, subject=0.0),
        policy=policy,
        policy_digest=_digest(),
    )
    assert report["verdict"] == "fail"
    assert report["side_accuracy"] == {
        "minimum": 0.75,
        "baseline": {"observed": 1.0, "passed": True},
        "subject": {"observed": 0.5, "passed": False},
        "passed": False,
    }

    with pytest.raises(EvidencePackError, match="comparison-report-v3"):
        build_comparison_report(
            comparison_id="comparison-1",
            paired_records=_pairs(baseline=1.0, subject=0.0),
            policy=policy,
            policy_digest=_digest(),
            report_format=COMPARISON_REPORT_FORMAT_V2,
        )


def test_paired_interval_is_deterministic_and_controls_the_verdict() -> None:
    pairs = {
        "format": PAIRED_RECORDS_FORMAT,
        "metric": "exact_match",
        "schedule_sha256": "a" * 64,
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
        ],
    }
    policy = {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": -75.0}}}}

    first = build_comparison_report(
        comparison_id="comparison-1",
        paired_records=pairs,
        policy=policy,
        policy_digest=_digest(),
    )
    second = build_comparison_report(
        comparison_id="comparison-1",
        paired_records=pairs,
        policy=policy,
        policy_digest=_digest(),
    )

    assert first == second
    assert first["comparison"] == {
        "kind": "exact_match_delta_pp",
        "value": -50.0,
        "minimum": -75.0,
    }
    assert first["uncertainty"]["lower"] == pytest.approx(  # type: ignore[index]
        -90.54687942657694
    )
    assert first["uncertainty"]["upper"] == pytest.approx(  # type: ignore[index]
        27.257278511318162
    )
    assert first["verdict"] == "fail"


@pytest.mark.parametrize(
    ("pairs", "policy", "message"),
    [
        ({}, {}, "paired records fields"),
        ({**_pairs(), "format": "other"}, {}, "paired records format"),
        ({**_pairs(), "metric": "other"}, {}, "metric is unsupported"),
        ({**_pairs(), "records": []}, {}, "non-empty array"),
        ({**_pairs(), "records": ["bad"]}, {}, "must be an object"),
        (
            {**_pairs(), "records": [{"baseline": {}, "subject": {"score": 1}}]},
            {},
            "baseline.score",
        ),
        (
            _pairs(),
            {},
            "exactly resolved_policy",
        ),
        (
            _pairs(),
            {"resolved_policy": {"metrics": {}}},
            "metrics.exact_match",
        ),
        (
            _pairs(),
            {"resolved_policy": {"metrics": {"accuracy": {"delta_min_pp": 0.0}}}},
            "metrics.exact_match",
        ),
        (
            _pairs(metric="normalized_nll_per_utf8_byte"),
            {"resolved_policy": {"metrics": {}}},
            "metrics.normalized_nll_per_utf8_byte",
        ),
        (
            _pairs(metric="normalized_nll_per_utf8_byte", baseline=0.0),
            {
                "resolved_policy": {
                    "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 1.0}}
                }
            },
            "greater than zero",
        ),
        (
            _pairs(),
            {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": -100.1}}}},
            "between -100 and 100",
        ),
        (
            _pairs(metric="normalized_nll_per_utf8_byte"),
            {
                "resolved_policy": {
                    "metrics": {"normalized_nll_per_utf8_byte": {"ratio_max": 0.0}}
                }
            },
            "ratio_max must be positive",
        ),
        (
            _pairs(baseline=math.inf),
            {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": 0.0}}}},
            "finite number",
        ),
    ],
)
def test_comparison_report_rejects_malformed_records_and_policy(
    pairs: dict[str, object], policy: dict[str, object], message: str
) -> None:
    with pytest.raises(EvidencePackError, match=message):
        build_comparison_report(
            comparison_id="comparison-1",
            paired_records=pairs,
            policy=policy,
            policy_digest=_digest(),
        )


@pytest.mark.parametrize(
    "policy",
    [
        {
            "resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": 0.0}}},
            "required_human_approval": True,
        },
        {
            "resolved_policy": {
                "metrics": {"exact_match": {"delta_min_pp": 0.0}},
                "deny": True,
            }
        },
        {
            "resolved_policy": {
                "metrics": {
                    "exact_match": {"delta_min_pp": 0.0},
                    "normalized_nll_per_utf8_byte": {"ratio_max": 1.0},
                }
            }
        },
        {
            "resolved_policy": {
                "metrics": {
                    "exact_match": {
                        "delta_min_pp": 0.0,
                        "required_human_approval": True,
                    }
                }
            }
        },
    ],
)
def test_comparison_report_rejects_ignored_policy_controls(
    policy: dict[str, object],
) -> None:
    with pytest.raises(EvidencePackError, match="exactly"):
        build_comparison_report(
            comparison_id="comparison-1",
            paired_records=_pairs(),
            policy=policy,
            policy_digest=_digest(),
        )


def test_comparison_report_rejects_bad_identifier_and_digest() -> None:
    policy = {"resolved_policy": {"metrics": {"exact_match": {"delta_min_pp": 0}}}}
    with pytest.raises(EvidencePackError, match="comparison_id"):
        build_comparison_report(
            comparison_id="contains spaces",
            paired_records=_pairs(),
            policy=policy,
            policy_digest=_digest(),
        )
    with pytest.raises(EvidencePackError, match="policy digest"):
        build_comparison_report(
            comparison_id="comparison-1",
            paired_records=_pairs(),
            policy=policy,
            policy_digest="invalid",
        )


def _output_record(text: str, *, digest: str | None = None) -> RuntimeScoringRecord:
    return RuntimeScoringRecord(
        record_id="record-1",
        input_sha256="0" * 64,
        status="ok",
        output_text=text,
        output_sha256=digest or hashlib.sha256(text.encode()).hexdigest(),
    )


def test_observation_record_validation_rejects_errors_digest_drift_and_empty_sets() -> (
    None
):
    failed = RuntimeScoringRecord(
        record_id="record-1",
        input_sha256="0" * 64,
        status="error",
        error_code="backend_error",
    )
    with pytest.raises(EvidencePackError, match="not successful"):
        contract._validate_observation_records((failed,), side="subject")
    with pytest.raises(EvidencePackError, match="output digest"):
        contract._validate_observation_records(
            (_output_record("answer", digest="1" * 64),), side="subject"
        )
    with pytest.raises(EvidencePackError, match="has no records"):
        contract._validate_observation_records((), side="subject")

    contract._validate_observation_records((_output_record("answer"),), side="subject")


def test_record_scoring_requires_metric_specific_authenticated_facts() -> None:
    exact = _output_record("answer")
    nll = RuntimeScoringRecord(
        record_id="record-1",
        input_sha256="0" * 64,
        status="ok",
        logprob_sum=-4.0,
        token_count=2,
        utf8_byte_count=2,
    )
    invalid_nll = RuntimeScoringRecord(
        record_id="record-1",
        input_sha256="0" * 64,
        status="ok",
        logprob_sum=4.0,
        token_count=2,
        utf8_byte_count=2,
    )

    assert (
        contract._score_record(
            exact, expected_output="answer", metric="exact_match", side="baseline"
        )
        == 1.0
    )
    assert (
        contract._score_record(
            exact, expected_output="other", metric="exact_match", side="baseline"
        )
        == 0.0
    )
    assert (
        contract._score_record(
            nll,
            expected_output="é",
            metric="normalized_nll_per_utf8_byte",
            side="baseline",
        )
        == 2.0
    )
    with pytest.raises(EvidencePackError, match="lacks output text"):
        contract._score_record(
            nll, expected_output="answer", metric="exact_match", side="baseline"
        )
    with pytest.raises(EvidencePackError, match="lacks normalized NLL facts"):
        contract._score_record(
            exact,
            expected_output="answer",
            metric="normalized_nll_per_utf8_byte",
            side="baseline",
        )
    with pytest.raises(EvidencePackError, match="utf8_byte_count"):
        contract._score_record(
            nll,
            expected_output="answer",
            metric="normalized_nll_per_utf8_byte",
            side="baseline",
        )
    with pytest.raises(EvidencePackError, match="invalid normalized NLL"):
        contract._score_record(
            invalid_nll,
            expected_output="é",
            metric="normalized_nll_per_utf8_byte",
            side="baseline",
        )


def _unchecked_observation(**values: object) -> EvidenceObservation:
    observation = object.__new__(EvidenceObservation)
    for key, value in values.items():
        object.__setattr__(observation, key, value)
    return observation


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {
                "observation_id": "bad id",
                "kind": "diagnostic",
                "scope": "comparison",
                "payload": b"{}\n",
            },
            "observation_id is invalid",
        ),
        (
            {
                "observation_id": "valid",
                "kind": "bad kind",
                "scope": "comparison",
                "payload": b"{}\n",
            },
            "kind is invalid",
        ),
        (
            {
                "observation_id": "valid",
                "kind": "diagnostic",
                "scope": "invalid",
                "payload": b"{}\n",
            },
            "scope is invalid",
        ),
        (
            {
                "observation_id": "valid",
                "kind": "diagnostic",
                "scope": "comparison",
                "payload": b"[",
            },
            "not valid JSON",
        ),
        (
            {
                "observation_id": "valid",
                "kind": "diagnostic",
                "scope": "comparison",
                "payload": b"[]\n",
            },
            "JSON object",
        ),
        (
            {
                "observation_id": "valid",
                "kind": "diagnostic",
                "scope": "comparison",
                "payload": b'{"value": 1}\n',
            },
            "canonical JSON",
        ),
    ],
)
def test_observation_envelope_revalidates_untrusted_instances(
    values: dict[str, object], message: str
) -> None:
    observation = _unchecked_observation(**values)
    with pytest.raises(EvidencePackError, match=message):
        contract.evidence_observation_bytes(
            observation,
            comparison_id="comparison-1",
            schedule_digest=_digest(),
            policy_digest=_digest("b"),
            artifact_digests={"baseline": _digest("c"), "subject": _digest("d")},
        )


def test_observation_envelope_requires_both_artifact_bindings() -> None:
    observation = EvidenceObservation(
        observation_id="valid",
        kind="diagnostic",
        scope="comparison",
        payload=b"{}\n",
    )
    with pytest.raises(EvidencePackError, match="baseline and subject"):
        contract.evidence_observation_bytes(
            observation,
            comparison_id="comparison-1",
            schedule_digest=_digest(),
            policy_digest=_digest("b"),
            artifact_digests={"baseline": _digest("c")},
        )


def test_observation_binding_errors_cover_manifest_and_envelope_drift() -> None:
    observation = EvidenceObservation(
        observation_id="valid",
        kind="diagnostic",
        scope="comparison",
        payload=b"{}\n",
    )
    encoded = contract.evidence_observation_bytes(
        observation,
        comparison_id="comparison-1",
        schedule_digest=_digest(),
        policy_digest=_digest("b"),
        artifact_digests={"baseline": _digest("c"), "subject": _digest("d")},
    )
    payload = contract.parse_json_object(encoded, label="observation")
    reference = {
        "path": "wrong",
        "digest": _digest("e"),
        "kind": "other",
        "scope": "subject",
    }
    payload.update(
        observation_id="other",
        kind="diagnostic",
        scope="comparison",
        authority="acceptance",
        bindings={},
    )
    errors = contract.evidence_observation_errors(
        payload,
        observation_id="valid",
        reference=reference,
        comparison_id="comparison-1",
        schedule_digest=_digest(),
        policy_digest=_digest("b"),
        artifact_digests={"baseline": _digest("c"), "subject": _digest("d")},
    )
    for fragment in (
        "manifest reference",
        "identifier binding",
        "kind binding",
        "scope binding",
        "does not bind",
        "authority is invalid",
    ):
        assert any(fragment in error for error in errors), (fragment, errors)


def test_runtime_side_config_rejects_parse_shape_binding_and_encoding() -> None:
    arguments = {
        "role": "baseline",
        "provider_name": "fixture",
        "artifact_identity_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
        "policy_digest": _digest("c"),
    }
    assert "valid JSON" in contract.runtime_side_config_errors(b"{", **arguments)[0]
    assert "JSON object" in contract.runtime_side_config_errors(b"[]", **arguments)[0]
    assert "does not bind" in contract.runtime_side_config_errors(b"{}", **arguments)[0]
    expected = {
        "format": contract.RUNTIME_SIDE_CONFIG_FORMAT,
        "role": "baseline",
        "provider": "fixture",
        "artifact_identity_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
        "policy_digest": _digest("c"),
    }
    noncanonical = contract.canonical_json_bytes(expected).replace(b'":', b'": ')
    assert (
        "canonical JSON"
        in contract.runtime_side_config_errors(noncanonical, **arguments)[0]
    )
