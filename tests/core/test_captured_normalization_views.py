"""Callable portable identities reject invalid intent without filesystem access."""

from copy import deepcopy
from types import MappingProxyType
from unittest.mock import Mock

import pytest

from invarlock import captured_normalization as normalization
from invarlock.evaluation_record_contracts.contracts import EvaluationRecordsError
from tests.core.test_captured_contract_freeze import _digest, _value


@pytest.mark.parametrize(
    "change,message",
    [
        ({"configuration": {"ignored": True}}, "without configuration"),
        ({"score_key": None}, "invalid policy"),
        ({"subject_minimum": 3, "subject_maximum": 2}, "minimum exceeds maximum"),
    ],
)
def test_recorded_policy_semantics_are_not_just_a_hash(change, message):
    policy = _value("policy.json")
    policy["metrics"][0].update(change)
    with pytest.raises(EvaluationRecordsError, match=message):
        normalization.comparison_policy_digest(policy)


@pytest.mark.parametrize("field", ["score_key", "accepted_provenance"])
def test_recorded_policy_requires_both_provenance_and_key(field):
    policy = _value("policy.json")
    del policy["metrics"][0][field]
    with pytest.raises(
        EvaluationRecordsError, match="requires provenance and score_key"
    ):
        normalization.comparison_policy_digest(policy)


@pytest.mark.parametrize("kind", ["judge", "human", "measurement"])
def test_recorded_provenance_units_and_rubrics_are_checked(kind):
    policy = _value("policy.json")
    provenance = policy["metrics"][0]["accepted_provenance"]
    provenance.update(kind=kind, rubric_digest=None)
    if kind == "measurement":
        provenance["unit"] = "wrong unit"
    with pytest.raises(EvaluationRecordsError, match="provenance is inconsistent"):
        normalization.comparison_policy_digest(policy)


@pytest.mark.parametrize("mutation", ["metrics", "slices", "overall"])
def test_policy_scope_names_must_be_unique_and_overall_reserved(mutation):
    policy = _value("policy.json")
    if mutation == "metrics":
        policy["metrics"].append(deepcopy(policy["metrics"][0]))
    else:
        policy["slices"] = [{"name": mutation, "where": {"group": "a"}}]
        if mutation == "slices":
            policy["slices"] *= 2
    with pytest.raises(EvaluationRecordsError, match="unique; overall is reserved"):
        normalization.comparison_policy_digest(policy)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"score_key": "external"}, "cannot accept recorded provenance"),
        ({"direction": "lower"}, "higher-is-better"),
        ({"unit": "seconds"}, "higher-is-better"),
        ({"configuration": {"unexpected": True}}, "configuration"),
    ],
)
def test_deterministic_policy_rejects_incompatible_options(change, message):
    policy = _value("policy.json")
    metric = policy["metrics"][0]
    del metric["score_key"], metric["accepted_provenance"]
    metric.update(kind="exact_match", unit="score", direction="higher")
    metric.update(change)
    with pytest.raises(EvaluationRecordsError, match=message):
        normalization.comparison_policy_digest(policy)


@pytest.mark.parametrize(
    "reference",
    ["../escape", "/absolute", "a/../b", "./run", "a//b", "a\\b", "https://host/run"],
)
def test_portable_normalization_rejects_unsafe_authored_references(reference):
    authored = _value("authored-pass.json")
    authored["comparison"]["baseline"]["path"] = reference
    with pytest.raises(EvaluationRecordsError, match="invalid captured request"):
        normalization.normalize_captured_request(
            authored,
            baseline=_value("baseline.json"),
            subject=_value("subject-pass.json"),
            policy=_value("policy.json"),
        )


@pytest.mark.parametrize("mutation", ["control", "oversized", "list", "pin", "path"])
def test_digest_refuses_invalid_normalized_intents(mutation):
    value = _value("request-pass.json")
    source = value["comparison"]["subject"]
    if mutation == "control":
        value["comparison"]["policy_digest"] += "\x7f"
        message = "control characters"
    elif mutation == "oversized":
        value["extra"] = "x" * (1024 * 1024)
        message = "1 MiB"
    elif mutation == "list":
        value["extra"] = [{"nested": True}, None, 1]
        message = "invalid captured request"
    elif mutation == "pin":
        source["expected_run_digest"] = "sha256:" + "0" * 64
        message = "subject normalized run pin is inconsistent"
    else:
        source["path"] = "source.json"
        message = "invalid captured request"
    with pytest.raises(EvaluationRecordsError, match=message):
        normalization.captured_request_digest(MappingProxyType(value))


@pytest.mark.parametrize("mutation", ["override", "identity", "physical", "provenance"])
def test_normalizer_checks_import_intent_before_binding(mutation):
    authored, baseline = _value("authored-pass.json"), _value("baseline.json")
    source = authored["comparison"]["baseline"]
    source.pop("expected_run_digest", None)
    if mutation == "override":
        source["run_id"] = baseline["run_id"]
        message = "canonical run identities cannot be overridden"
    else:
        source.update(
            adapter="jsonl",
            **{
                k: deepcopy(baseline[k])
                for k in ("source", "run_id", "artifact_digest", "score_provenance")
            },
        )
        baseline["source_digest"] = "sha256:" + "a" * 64
        if mutation == "identity":
            del source["run_id"]
        elif mutation == "physical":
            baseline["source_digest"] = None
        else:
            source["score_provenance"] = {}
        message = (
            "native import requires" if mutation != "provenance" else "source intent"
        )
    with pytest.raises(EvaluationRecordsError, match=message):
        normalization.normalize_captured_request(
            authored,
            baseline=baseline,
            subject=_value("subject-pass.json"),
            policy=_value("policy.json"),
        )


def test_frozen_identities_are_portable_and_do_not_touch_host_or_scorer(monkeypatch):
    forbidden = Mock(
        side_effect=AssertionError("portable identity performed I/O or scoring")
    )
    monkeypatch.setattr("invarlock.captured_contracts.read_file", forbidden)
    monkeypatch.setattr("invarlock.core.scoring.score", forbidden)
    authored = _value("authored-pass.json")
    authored["output"]["evidence"] = "nonexistent/output"
    actual = normalization.normalize_captured_request(
        MappingProxyType(authored),
        baseline=_value("baseline.json"),
        subject=_value("subject-pass.json"),
        policy=_value("policy.json"),
    )
    assert actual == _value("request-pass.json")
    assert normalization.captured_request_digest(actual) == _digest("request-pass.json")
    forbidden.assert_not_called()
