"""Unicode semantics are explicit policy inputs, never local runtime defaults."""

import copy
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock.core.builtin_scorers import BuiltinScorer
from invarlock.core.scorer_extension import (
    AuthenticatedScorerRecord,
    ScorerExtensionError,
    ScorerExtensionRegistry,
    ScorerReplayRequest,
    build_scorer_binding,
)
from invarlock.pipeline import PipelineError, create_evidence, metrics, verify_evidence
from invarlock.pipeline.contracts import digest, validate
from invarlock.pipeline.metrics import UNICODE_VERSION, MetricError, score
from invarlock.pipeline.templates import example_project

OTHER_VERSION = "16.0.0" if UNICODE_VERSION != "16.0.0" else "15.0.0"


@pytest.mark.parametrize("kind", ["normalized_match", "token_f1"])
@pytest.mark.parametrize("configuration", [{}, {"casefold": False}])
def test_unicode_metrics_require_an_explicit_version(kind, configuration):
    with pytest.raises(MetricError, match="explicit unicode_version"):
        score(kind, "same", "same", configuration)


@pytest.mark.parametrize("kind", ["normalized_match", "token_f1"])
@pytest.mark.parametrize("version", [OTHER_VERSION, None, 15, ""])
def test_unsupported_unicode_version_fails_before_normalization(
    monkeypatch, kind, version
):
    def unexpected_normalization(*args):
        raise AssertionError("unsupported semantics must not score a record")

    monkeypatch.setattr(metrics, "_normalize", unexpected_normalization)
    with pytest.raises(MetricError, match="this runtime provides"):
        score(kind, "same", "same", {"unicode_version": version})


@pytest.mark.parametrize("kind", ["normalized_match", "token_f1"])
def test_cyrillic_case_pair_uses_the_declared_unicode_version(kind):
    # Unicode 16 introduced the U+1C89/U+1C8A uppercase/lowercase pair. Identical
    # text therefore has different scores under Python's Unicode 15 and 16 data.
    expected = float(tuple(map(int, UNICODE_VERSION.split("."))) >= (16, 0, 0))
    assert (
        score(kind, "\u1c8a", "\u1c89", {"unicode_version": UNICODE_VERSION})
        == expected
    )


@pytest.mark.parametrize("kind", ["normalized_match", "token_f1"])
def test_unicode_version_is_bound_in_the_core_scorer_configuration(kind):
    scorer = BuiltinScorer(kind)
    binding = build_scorer_binding(
        scorer.descriptor(), {"unicode_version": UNICODE_VERSION}
    )
    other = build_scorer_binding(
        scorer.descriptor(), {"unicode_version": OTHER_VERSION}
    )
    assert binding.configuration_sha256 != other.configuration_sha256
    registry = ScorerExtensionRegistry(allow_installed=False)
    with pytest.raises(ScorerExtensionError, match="unicode_version"):
        registry.validate_binding(
            build_scorer_binding(scorer.descriptor(), {}),
            task="text_causal",
            input_kinds=("text",),
            output_kind="text",
        )
    request = ScorerReplayRequest(
        binding=other,
        task="text_causal",
        input_kinds=("text",),
        output_kind="text",
        schedule_sha256="a" * 64,
        records=(
            AuthenticatedScorerRecord(
                record_id="one",
                input_sha256="b" * 64,
                facts={
                    "expected_output": "\u1c8a",
                    "output_text": "\u1c89",
                    "output_sha256": hashlib.sha256("\u1c89".encode()).hexdigest(),
                },
            ),
        ),
    )
    with pytest.raises(ScorerExtensionError, match="this runtime provides"):
        registry.replay(request)


@pytest.mark.parametrize("kind", ["normalized_match", "token_f1"])
def test_pipeline_policy_and_embedded_evidence_require_the_version(kind):
    baseline, candidate, policy = example_project("classification")
    policy["metrics"][0]["kind"] = kind
    evidence = create_evidence(baseline, candidate, policy)
    del policy["metrics"][0]["configuration"]["unicode_version"]
    with pytest.raises(PipelineError, match="unicode_version"):
        validate(policy, "policy")
    evidence["policy"] = policy
    with pytest.raises(PipelineError, match="unicode_version"):
        validate(evidence, "evidence")


@pytest.mark.parametrize("kind", ["normalized_match", "token_f1"])
def test_signed_evidence_requires_matching_recipient_unicode_semantics(
    monkeypatch, kind
):
    baseline, candidate, policy = example_project("classification")
    policy["metrics"][0]["kind"] = kind
    key = Ed25519PrivateKey.generate()
    evidence = create_evidence(baseline, candidate, policy, key)
    expected = {
        "public_key": key.public_key(),
        "expected_baseline": digest(baseline),
        "expected_candidate": digest(candidate),
        "policy": policy,
    }
    assert verify_evidence(evidence, **expected)["decision"] == "pass"
    changed = copy.deepcopy(policy)
    changed["metrics"][0]["configuration"]["unicode_version"] = OTHER_VERSION
    assert digest(policy) != digest(changed)
    with pytest.raises(PipelineError, match="policy differs"):
        verify_evidence(evidence, **{**expected, "policy": changed})
    monkeypatch.setattr(metrics.unicodedata, "unidata_version", OTHER_VERSION)
    with pytest.raises(PipelineError, match="this runtime provides"):
        verify_evidence(evidence, **expected)


def test_templates_bind_unicode_only_for_metrics_that_normalize_text():
    for example in ("classification", "extraction", "judge"):
        _, _, policy = example_project(example)
        for metric in policy["metrics"]:
            if metric["kind"] in ("normalized_match", "token_f1"):
                assert metric["configuration"]["unicode_version"] == UNICODE_VERSION
            else:
                assert "unicode_version" not in metric["configuration"]
