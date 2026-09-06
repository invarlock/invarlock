"""Whole-document decisions preserve strict JSON meaning and prior scorer identities."""

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.core.builtin_scorers import BuiltinScorer
from invarlock.core.scorer_extension import scorer_descriptor_sha256
from invarlock.pipeline import create_evidence, verify_evidence
from invarlock.pipeline.contracts import digest
from invarlock.pipeline.metrics import MetricError, score
from invarlock.pipeline.templates import example_project


@pytest.mark.parametrize(
    "reference,answer,expected",
    [
        ('{"a":1,"b":[true,null]}', ' { "b": [true, null], "a": 1 } ', 1),
        ('{"a":1}', '{"a":1,"extra":0}', 0),
        ('{"a":1}', "{}", 0),
        ('{"a":{"b":1}}', '{"a":{"b":2}}', 0),
        ('{"a":true}', '{"a":1}', 0),
        ('{"a":1}', '{"a":1.0}', 0),
        ("[1,2]", "[2,1]", 0),
        ('{"a":1}', '{"a":1,"a":1}', 0),
        ('{"a":1}', '{"a":NaN}', 0),
        ('{"a":1}', '```json\n{"a":1}\n```', 0),
        ('{"a":1}', '{"a":1', 0),
        ({"a": [1, None]}, {"a": [1, None]}, 1),
        ("null", None, 1),
    ],
)
def test_whole_json_semantics(reference, answer, expected):
    assert score("json_exact", reference, answer, {}) == expected


@pytest.mark.parametrize(
    "reference", ['{"a":1,"a":2}', "NaN", "not JSON", float("inf")]
)
def test_invalid_gold_blocks_scoring(reference):
    with pytest.raises(MetricError, match="JSON reference"):
        score("json_exact", reference, "{}", {})


def test_whole_json_configuration_is_closed():
    with pytest.raises(MetricError, match="unsupported configuration"):
        score("json_exact", "{}", "{}", {"fields": ["/a"]})


def test_whole_json_signed_replay_uses_binary_interval():
    baseline, candidate, policy = example_project("extraction")
    metric = policy["metrics"][0]
    metric.update(kind="json_exact", configuration={})
    key = Ed25519PrivateKey.generate()
    evidence = create_evidence(baseline, candidate, policy, key)
    result = verify_evidence(
        evidence,
        public_key=key.public_key(),
        policy=policy,
        expected_baseline=digest(baseline),
        expected_candidate=digest(candidate),
    )
    assert result["decision"] == "pass"
    assert "newcombe" in result["metrics"][0]["interval"]["method"].lower()
    assert result["metrics"][0]["candidate_mean"] == 1


@pytest.mark.parametrize(
    "kind,expected",
    [
        (
            "json_fields",
            "f3de13998e33cdc766f3e199e5bf5ffbe6f797eff493d43bb29102c5d143e26c",
        ),
        (
            "normalized_match",
            "bf5db1886082e8c5c87b096a8f93c127c701c15a037219f1a3b8c13b914ceeae",
        ),
        (
            "numeric_tolerance",
            "3843f034b517f9fd14b44efaa0d40f5a9a730266c748bf022e3fbd04f5083eb6",
        ),
        (
            "token_f1",
            "b6094f4ffd3d2bac99096768771dc177412b5452cd44d65c419a280105879686",
        ),
    ],
)
def test_legacy_scorer_descriptors_remain_replay_compatible(kind, expected):
    assert scorer_descriptor_sha256(BuiltinScorer(kind).descriptor()) == expected
