"""Captured scorer intent is closed, preserved, and independently replayed."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import yaml

from invarlock.captured_evaluation import (
    CapturedEvaluationError,
    evaluate_captured_request,
    preflight_captured_request,
)
from invarlock.captured_normalization import (
    captured_request_digest,
    normalize_captured_request,
)
from invarlock.captured_verification import verify_captured_evidence
from invarlock.core.evaluation_request import (
    CapturedEvaluationRequest,
    CapturedJudgeRequest,
    EvaluationRequestError,
    load_evaluation_request,
)
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
)
from invarlock.evaluation_records.adapters import _parse_run_bytes
from tests.cli.test_import_journey import _key
from tests.evaluation_comparison.test_likelihood import policy as nll_policy


@pytest.fixture
def material(tmp_path):
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [
            {
                "name": "exact",
                "kind": "exact_match",
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 1,
                "maximum_regression": 0.5,
                "maximum_interval_width": 1,
            }
        ],
        "slices": [],
    }
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    sources, runs = {}, {}
    for side, marker in (("baseline", "a"), ("subject", "b")):
        source = {
            "path": f"{side}.jsonl",
            "adapter": "jsonl",
            "source": {"name": "existing-evaluator", "version": "1"},
            "run_id": side,
            "artifact_digest": "sha256:" + marker * 64,
            "input_projection": {"kind": "json-pointer", "pointer": "/input/text"},
        }
        raw = json.dumps(
            {
                "id": "one",
                "input": {"text": "question", "alias": "question"},
                "expected": "yes",
                "output": "yes",
            }
        ).encode()
        (tmp_path / source["path"]).write_bytes(raw)
        runs[side] = _parse_run_bytes(
            raw, **{key: value for key, value in source.items() if key != "path"}
        )
        sources[side] = source
    authored = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {**sources, "policy": "policy.json", "metric": "exact_match"},
        "output": {"evidence": "evidence"},
    }
    return {"authored": authored, "policy": policy, **runs}


def _normalize(material):
    return normalize_captured_request(
        material["authored"],
        baseline=material["baseline"],
        subject=material["subject"],
        policy=material["policy"],
    )


def _write(tmp_path, material):
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(material["authored"]))
    return path


def test_selector_and_projection_survive_path_free_normalization_and_detach(material):
    normalized = _normalize(material)
    assert normalized["comparison"]["metric"] == "exact_match"
    for side in ("baseline", "subject"):
        assert normalized["comparison"][side]["input_projection"] == {
            "kind": "json-pointer",
            "pointer": "/input/text",
        }
        assert "path" not in normalized["comparison"][side]
    before = captured_request_digest(normalized)
    material["authored"]["comparison"]["baseline"]["input_projection"]["pointer"] = (
        "/input/alias"
    )
    assert captured_request_digest(normalized) == before


def test_omitted_selector_preserves_existing_multi_metric_policy_intent(material):
    del material["authored"]["comparison"]["metric"]
    second = copy.deepcopy(material["policy"]["metrics"][0])
    second["name"] = "second-exact"
    material["policy"]["metrics"].append(second)
    assert "metric" not in _normalize(material)["comparison"]


@pytest.mark.parametrize("change", ["different-kind", "multiple"])
def test_explicit_selector_must_equal_the_only_policy_metric(material, change):
    if change == "different-kind":
        material["authored"]["comparison"]["metric"] = "normalized_nll_per_utf8_byte"
    else:
        second = copy.deepcopy(material["policy"]["metrics"][0])
        second["name"] = "second-exact"
        material["policy"]["metrics"].append(second)
    with pytest.raises(EvaluationRecordsError, match="match the single policy metric"):
        _normalize(material)


def test_normalization_accepts_explicit_nll_policy_semantics(material):
    material["policy"] = nll_policy()
    material["authored"]["comparison"]["metric"] = "normalized_nll_per_utf8_byte"
    normalized = _normalize(material)
    assert normalized["comparison"]["metric"] == "normalized_nll_per_utf8_byte"
    assert normalized["comparison"]["policy_digest"] == digest(material["policy"])


@pytest.mark.parametrize("side", ["baseline", "subject"])
@pytest.mark.parametrize("retained", ["different", "missing", "nonobject-context"])
def test_authored_projection_must_match_every_retained_row(material, side, retained):
    if retained == "different":
        material["authored"]["comparison"][side]["input_projection"]["pointer"] = (
            "/input/alias"
        )
    else:
        row = material[side]["records"][0]
        row["input"] = {"text": "question", "alias": "question"}
        row["context"] = [] if retained == "nonobject-context" else {}
    with pytest.raises(
        EvaluationRecordsError, match=f"{side} input projection differs"
    ):
        _normalize(material)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_projection", {"kind": "json-pointer", "pointer": "/input/text"}),
        ("source", {"name": "other", "version": "1"}),
        ("run_id", "override"),
        ("artifact_digest", "sha256:" + "a" * 64),
        ("score_provenance", {}),
    ],
)
def test_canonical_normalization_refuses_identity_or_projection_overrides(
    material, field, value
):
    material["authored"]["comparison"]["baseline"] = {
        "path": "baseline.jsonl",
        "adapter": "invarlock",
        field: value,
    }
    with pytest.raises(
        EvaluationRecordsError, match="cannot be overridden|invalid captured request"
    ):
        _normalize(material)


@pytest.mark.parametrize("target", ["comparison", "projection", "source", "judge"])
def test_new_loader_contracts_reject_unknown_fields(tmp_path, material, target):
    comparison = material["authored"]["comparison"]
    if target == "comparison":
        comparison["unrecognized"] = True
    elif target == "projection":
        comparison["baseline"]["input_projection"]["template"] = "{input}"
    elif target == "source":
        comparison["baseline"]["source"]["trust"] = "native"
    else:
        comparison.update(
            metric="judge",
            judge={
                "workspace": "workspace",
                "signer_identity": "signer",
                "provider": "inferred",
            },
        )
    with pytest.raises(EvaluationRequestError):
        load_evaluation_request(_write(tmp_path, material))


@pytest.mark.parametrize(
    "selector", [None, "exact_match", "normalized_nll_per_utf8_byte"]
)
def test_loader_preserves_optional_metric_and_source_projection(
    tmp_path, material, selector
):
    if selector is None:
        del material["authored"]["comparison"]["metric"]
    else:
        material["authored"]["comparison"]["metric"] = selector
    request = load_evaluation_request(_write(tmp_path, material))
    assert isinstance(request, CapturedEvaluationRequest)
    assert request.metric == selector
    assert request.judge is None
    assert request.baseline.input_projection == {
        "kind": "json-pointer",
        "pointer": "/input/text",
    }
    assert request.baseline.path == tmp_path / "baseline.jsonl"


@pytest.mark.parametrize("measurements", [False, True])
def test_loader_resolves_captured_judge_configuration(tmp_path, material, measurements):
    judge = {"workspace": "private/workspace", "signer_identity": "reviewed-signer"}
    if measurements:
        (tmp_path / "measurements.json").write_text("{}")
        judge["measurements"] = "measurements.json"
    material["authored"]["comparison"].update(metric="judge", judge=judge)
    request = load_evaluation_request(_write(tmp_path, material))
    assert request.metric == "judge"
    assert request.judge == CapturedJudgeRequest(
        workspace=tmp_path / "private/workspace",
        signer_identity="reviewed-signer",
        measurements=tmp_path / "measurements.json" if measurements else None,
    )


@pytest.mark.parametrize(
    "selector", [None, "exact_match", "normalized_nll_per_utf8_byte"]
)
@pytest.mark.parametrize(
    "function", [preflight_captured_request, evaluate_captured_request]
)
def test_typed_request_refuses_judge_configuration_for_nonjudge_metric(
    tmp_path, material, selector, function
):
    request = load_evaluation_request(_write(tmp_path, material))
    request = replace(
        request,
        metric=selector,
        judge=CapturedJudgeRequest(
            workspace=tmp_path / "workspace", signer_identity="signer"
        ),
    )
    with pytest.raises(
        CapturedEvaluationError, match="judge configuration requires metric: judge"
    ):
        function(request, unsigned=True)
    assert not request.evidence.exists()
    assert not request.judge.workspace.exists()


@pytest.mark.parametrize(
    "function", [preflight_captured_request, evaluate_captured_request]
)
def test_typed_judge_refuses_bootstrap_override_before_collection(
    tmp_path, material, function
):
    request = load_evaluation_request(_write(tmp_path, material))
    request = replace(
        request,
        metric="judge",
        judge=CapturedJudgeRequest(
            workspace=tmp_path / "workspace", signer_identity="signer"
        ),
    )
    with pytest.raises(
        CapturedEvaluationError, match="bootstrap overrides do not apply"
    ):
        function(request, unsigned=True, max_bootstrap_draws=0)
    assert not request.evidence.exists()


def test_judge_workspace_cannot_contain_the_request_with_explicit_ancestor_root(
    tmp_path, material
):
    material["authored"]["comparison"].update(
        metric="judge", judge={"workspace": "workspace", "signer_identity": "signer"}
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "request.yaml"
    path.write_text(yaml.safe_dump(material["authored"]))
    with pytest.raises(
        EvaluationRequestError, match="request.*workspace|workspace.*request"
    ):
        load_evaluation_request(path, request_root=tmp_path)


@pytest.mark.parametrize(
    "metric,decision",
    [("exact_match", "regression"), ("normalized_nll_per_utf8_byte", "pass")],
)
def test_signed_projected_capture_replays_with_recipient_owned_intent(
    tmp_path, material, metric, decision
):
    if metric == "normalized_nll_per_utf8_byte":
        material["policy"] = nll_policy()
        material["authored"]["comparison"]["metric"] = metric
        (tmp_path / "policy.json").write_text(json.dumps(material["policy"]))
        for side in ("baseline", "subject"):
            source = material["authored"]["comparison"][side]
            path = tmp_path / source["path"]
            row = json.loads(path.read_bytes())
            row["likelihood"] = {
                "basis": "reference_continuation",
                "logprob_sum": -3,
                "token_count": 1,
                "utf8_byte_count": 3,
                "input_digest": digest(row["input"]),
                "reference_digest": digest(row["expected"]),
                "artifact_digest": source["artifact_digest"],
                "configuration_digest": digest("configuration"),
                "tokenizer_digest": digest("tokenizer"),
                "source": source["source"],
            }
            raw = json.dumps(row).encode()
            path.write_bytes(raw)
            material[side] = _parse_run_bytes(
                raw, **{key: value for key, value in source.items() if key != "path"}
            )
    normalized = _normalize(material)
    request = load_evaluation_request(_write(tmp_path, material))
    key, signer = _key(tmp_path / "signer.pem")
    evaluate_captured_request(request, signing_key_path=key)
    result = verify_captured_evidence(
        tmp_path / "evidence",
        policy_path=tmp_path / "policy.json",
        expected_baseline_run=digest(material["baseline"]),
        expected_subject_run=digest(material["subject"]),
        expected_request_digest=captured_request_digest(normalized),
        expected_signer=signer,
        receipt_path=tmp_path / "receipt.json",
        verifier_signing_key_path=key,
        verifier_identity="recipient",
    )
    assert result["integrity_ok"]
    assert result["replay_status"] == "completed"
    assert result["decision"] == decision
    assert (tmp_path / "receipt.json").is_file()


@pytest.mark.parametrize(
    "change",
    [
        "missing-judge",
        "wrong-selector",
        "canonical-projection",
        "invalid-pointer",
        "missing-measurements",
    ],
)
def test_loader_rejects_unsupported_judge_and_projection_intent(
    tmp_path, material, change
):
    comparison = material["authored"]["comparison"]
    if change == "missing-judge":
        comparison["metric"] = "judge"
    elif change == "wrong-selector":
        comparison["judge"] = {"workspace": "workspace", "signer_identity": "signer"}
    elif change == "canonical-projection":
        comparison["baseline"] = {
            "path": "baseline.jsonl",
            "adapter": "invarlock",
            "input_projection": {"kind": "json-pointer", "pointer": "/input"},
        }
    elif change == "invalid-pointer":
        comparison["baseline"]["input_projection"]["pointer"] = "/output"
    else:
        comparison.update(
            metric="judge",
            judge={
                "workspace": "workspace",
                "signer_identity": "signer",
                "measurements": "missing.json",
            },
        )
    with pytest.raises(EvaluationRequestError):
        load_evaluation_request(_write(tmp_path, material))


@pytest.mark.parametrize("target", ["metric", "projection"])
def test_normalized_request_hash_rejects_unknown_semantics(material, target):
    normalized = _normalize(material)
    if target == "metric":
        normalized["comparison"]["metric"] = "invented-scorer"
    else:
        normalized["comparison"]["baseline"]["input_projection"]["execute"] = (
            "arbitrary code"
        )
    with pytest.raises(EvaluationRecordsError, match="invalid captured request"):
        captured_request_digest(normalized)


@pytest.mark.parametrize(
    "function", [preflight_captured_request, evaluate_captured_request]
)
@pytest.mark.parametrize("unsigned", [False, True])
def test_deterministic_signing_intent_fails_before_reading_records(
    tmp_path, material, monkeypatch, function, unsigned
):
    request = load_evaluation_request(_write(tmp_path, material))
    monkeypatch.setattr(
        "invarlock.captured_evaluation._run",
        lambda *args, **kwargs: pytest.fail("invalid signing intent read inputs"),
    )
    with pytest.raises(CapturedEvaluationError, match="signing key"):
        function(
            request,
            unsigned=unsigned,
            signing_key_path=tmp_path / "missing.pem" if unsigned else None,
        )
    assert not request.evidence.exists()
