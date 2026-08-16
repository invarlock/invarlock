"""Adversarial semantic replay tests for evidence-pack contracts."""

from __future__ import annotations

import copy
import dataclasses
from types import SimpleNamespace

import pytest

from invarlock import evidence_pack_contract as contract
from invarlock.core.runtime_provider import (
    build_runtime_behavioral_schedule_from_material,
)
from invarlock.evidence_pack_contract import EvidencePackError


def _exact_paired_records() -> dict[str, object]:
    return {
        "format": contract.PAIRED_RECORDS_FORMAT,
        "metric": "exact_match",
        "schedule_sha256": "a" * 64,
        "records": [
            {
                "record_id": "one",
                "baseline": {"score": 1.0},
                "subject": {"score": 1.0},
            }
        ],
    }


def _exact_policy(**updates: object) -> dict[str, object]:
    selected: dict[str, object] = {"delta_min_pp": 0.0}
    selected.update(updates)
    return {"resolved_policy": {"metrics": {"exact_match": selected}}}


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ({"resolved_policy": {"metrics": []}}, "metrics must be an object"),
        (
            {"resolved_policy": {"metrics": {"exact_match": []}}},
            "exact_match must be an object",
        ),
        (
            _exact_policy(
                minimum_record_count=True,
                maximum_interval_width_pp=100.0,
            ),
            "minimum_record_count must be an integer",
        ),
        (
            _exact_policy(
                minimum_record_count=1,
                maximum_interval_width_pp=0.0,
            ),
            "maximum_interval_width_pp must be positive",
        ),
    ],
)
def test_metric_policy_rejects_invalid_structural_and_sampling_bounds(
    policy: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(EvidencePackError, match=message):
        contract._resolved_metric_policy(policy, metric="exact_match")


def _available_perplexity() -> dict[str, object]:
    return {
        "status": "available",
        "basis": "authenticated_target_likelihood",
        "method": contract.DERIVED_PERPLEXITY_METHOD,
        "tokenizer_metadata_sha256": "a" * 64,
        "target_token_count": 2,
        "baseline_perplexity": 2.0,
        "subject_perplexity": 2.2,
        "ratio": 1.1,
    }


@pytest.mark.parametrize(
    ("measurement", "message"),
    [
        ({**_available_perplexity(), "basis": "untrusted"}, "basis is invalid"),
        ({**_available_perplexity(), "method": "other"}, "method is invalid"),
        (
            {
                "status": "unavailable",
                "basis": "authenticated_target_likelihood",
                "method": contract.DERIVED_PERPLEXITY_METHOD,
                "reason": "unknown",
            },
            "unavailability is invalid",
        ),
        (
            {
                "status": "available",
                "basis": "authenticated_target_likelihood",
                "method": contract.DERIVED_PERPLEXITY_METHOD,
            },
            "fields are invalid",
        ),
        (
            {**_available_perplexity(), "tokenizer_metadata_sha256": "bad"},
            "bindings are invalid",
        ),
        (
            {**_available_perplexity(), "ratio": 1.2},
            "values are invalid",
        ),
    ],
)
def test_derived_perplexity_contract_rejects_false_or_unbound_claims(
    measurement: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(EvidencePackError, match=message):
        contract.validated_derived_measurements({"perplexity_ratio": measurement})


def test_derived_perplexity_rejects_nonfinite_math_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract.math, "exp", lambda _value: float("inf"))

    measurement = contract._derived_perplexity_measurement(
        baseline_records=(SimpleNamespace(token_count=1, logprob_sum=-1.0),),
        subject_records=(SimpleNamespace(token_count=1, logprob_sum=-1.0),),
        baseline_tokenizer_sha256="a" * 64,
        subject_tokenizer_sha256="a" * 64,
    )

    assert measurement["reason"] == "derived_value_non_finite"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["records"][0].update(baseline=[]),
            "baseline is invalid",
        ),
        (
            lambda value: value["records"][0].update(weight=1.0),
            "weight is invalid",
        ),
        (
            lambda value: value.update(schedule_sha256=None),
            "schedule_sha256 is invalid",
        ),
    ],
)
def test_report_builder_rejects_untrusted_record_and_schedule_surfaces(
    mutation: object,
    message: str,
) -> None:
    paired = copy.deepcopy(_exact_paired_records())
    assert callable(mutation)
    mutation(paired)

    with pytest.raises(EvidencePackError, match=message):
        contract.build_comparison_report(
            comparison_id="comparison",
            paired_records=paired,
            policy=_exact_policy(),
            policy_digest="sha256:" + "b" * 64,
        )


def test_schedule_bytes_recomputes_and_rejects_claimed_digest_drift() -> None:
    schedule = build_runtime_behavioral_schedule_from_material(
        dataset_identity={
            "provider": "local",
            "dataset_name": "contract-edge",
            "config_name": "evaluation-records-jsonl-v1",
            "revision": "c" * 64,
            "split": "validation",
        },
        records=[
            {
                "record_id": "one",
                "input_text": "prompt",
                "expected_output": "answer",
            }
        ],
    )
    drifted = dataclasses.replace(schedule, schedule_sha256="0" * 64)

    with pytest.raises(EvidencePackError, match="digest does not match"):
        contract.schedule_bytes(drifted)
