"""The same shipped metrics must work at the strict scorer extension boundary."""

import hashlib

import pytest

from invarlock.core.builtin_scorers import BuiltinScorer
from invarlock.core.scorer_extension import (
    AuthenticatedScorerRecord,
    ScorerExtensionError,
    ScorerExtensionRegistry,
    ScorerReplayRequest,
    build_scorer_binding,
)
from invarlock.pipeline.metrics import UNICODE_VERSION


@pytest.mark.parametrize(
    "kind,config,target,output,value",
    [
        ("normalized_match", {"unicode_version": UNICODE_VERSION}, "yes", " YES ", 1),
        ("numeric_tolerance", {"absolute": 0.1}, "10", "10.01", 1),
        ("token_f1", {"unicode_version": UNICODE_VERSION}, "a b", "a", 2 / 3),
        ("json_fields", {"fields": ["/x"]}, '{"x":1}', '{"x":1}', 1),
    ],
)
def test_shipped_scorers_replay_with_third_party_code_disabled(
    kind, config, target, output, value
):
    scorer = BuiltinScorer(kind)
    binding = build_scorer_binding(scorer.descriptor(), config)
    request = ScorerReplayRequest(
        binding=binding,
        task="text_causal",
        input_kinds=("text",),
        output_kind="text",
        schedule_sha256="a" * 64,
        records=(
            AuthenticatedScorerRecord(
                record_id="one",
                input_sha256="b" * 64,
                facts={
                    "expected_output": target,
                    "output_text": output,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                },
            ),
        ),
    )
    result = ScorerExtensionRegistry(allow_installed=False).replay(request)
    assert result.aggregate == value
    assert result.record_results[0].value == value


def test_external_code_cannot_shadow_a_shipped_scorer():
    scorer = BuiltinScorer("normalized_match")
    registry = ScorerExtensionRegistry(allow_installed=False, authorized=(scorer,))
    with pytest.raises(ScorerExtensionError, match="shadowed"):
        registry.validate_binding(
            build_scorer_binding(
                scorer.descriptor(), {"unicode_version": UNICODE_VERSION}
            ),
            task="text_causal",
            input_kinds=("text",),
            output_kind="text",
        )


def test_shipped_scorer_completes_existing_signed_evaluate_verify_report(tmp_path):
    from invarlock.evaluation_transaction import evaluate_request_file
    from invarlock.evidence_reporting import render_evidence
    from invarlock.evidence_verification import verify_evidence
    from tests.cli.test_import_journey import _input_anchors, _key, _materialize_request

    registry = ScorerExtensionRegistry(allow_installed=False)
    binding = build_scorer_binding(
        BuiltinScorer("normalized_match").descriptor(),
        {"unicode_version": UNICODE_VERSION},
    )
    material = _materialize_request(
        tmp_path, scorer_binding=binding, scorer_registry=registry
    )
    signer, fingerprint = _key(tmp_path / "evidence-signer.pem")
    verifier, _ = _key(tmp_path / "reviewer.pem")
    evaluated = evaluate_request_file(
        material["request"], signing_key_path=signer, scorer_registry=registry
    )
    anchors = _input_anchors(evaluated.evidence_path)
    runtimes = material["runtime_digests"]
    verified = verify_evidence(
        evaluated.evidence_path,
        policy_path=material["policy"],
        expected_baseline_artifact=anchors["baseline"],
        expected_subject_artifact=anchors["subject"],
        expected_schedule=anchors["dataset"],
        expected_baseline_runtime=runtimes["baseline"],
        expected_subject_runtime=runtimes["subject"],
        expected_signer=fingerprint,
        receipt_path=tmp_path / "receipt.json",
        verifier_signing_key_path=verifier,
        verifier_identity="independent-reviewer",
        scorer_registry=registry,
    )
    assert verified.payload["ok"] is True
    assert "invarlock.normalized_match" in render_evidence(evaluated.evidence_path).text
