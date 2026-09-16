"""Preflight and evaluation failures through the callable captured boundary."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from invarlock import captured_evaluation as evaluation
from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_pack_contract import canonical_json_bytes
from tests.core.test_captured_contract_freeze import (
    _digest,
    _raw,
)
from tests.core.test_captured_contract_freeze import (
    _request as frozen_request,
)
from tests.core.test_captured_evaluation import _request


@pytest.mark.parametrize("unsigned", [None, 0, 1, "true"])
def test_callable_preflight_rejects_non_boolean_unsigned(
    tmp_path, monkeypatch, unsigned
):
    request = frozen_request(tmp_path)
    forbidden = Mock(
        side_effect=AssertionError("invalid options reached input acquisition")
    )
    monkeypatch.setattr(evaluation, "_run", forbidden)
    with pytest.raises(
        evaluation.CapturedEvaluationError, match="unsigned must be a boolean"
    ):
        evaluation.preflight_captured_request(request, unsigned=unsigned)
    forbidden.assert_not_called()
    assert not request.evidence.exists()


@pytest.mark.parametrize(
    "raw", [b"[]", b"null", b'{"metrics": [], "metrics": []}', b"{", b'{"value": NaN}']
)
def test_bad_policy_bytes_raise_a_captured_error_before_comparison(
    tmp_path, monkeypatch, raw
):
    request = frozen_request(tmp_path)
    request.policy.write_bytes(raw)
    forbidden = Mock(
        side_effect=AssertionError("invalid policy reached comparison/publication")
    )
    monkeypatch.setattr(evaluation, "compare_runs", forbidden)
    monkeypatch.setattr(evaluation, "publish_captured_evidence", forbidden)
    with pytest.raises(evaluation.CapturedEvaluationError, match="captured policy"):
        evaluation.evaluate_captured_request(request, unsigned=True)
    forbidden.assert_not_called()
    assert not request.evidence.exists()


@pytest.mark.parametrize("side", ["baseline", "subject"])
def test_invalid_canonical_run_is_labeled_with_its_side(tmp_path, side):
    request = frozen_request(tmp_path)
    getattr(request, side).path.write_bytes(b"{}\n")
    with pytest.raises(
        evaluation.CapturedEvaluationError, match=f"{side} run could not be loaded"
    ):
        evaluation.preflight_captured_request(request, unsigned=True)
    assert not request.evidence.exists()


@pytest.mark.parametrize("reference", ["outside", "traversal", "missing", "symlink"])
def test_typed_request_input_references_are_rechecked(tmp_path, reference):
    request = frozen_request(tmp_path)
    path = {
        "outside": tmp_path.parent / "not-under-request.json",
        "traversal": tmp_path / "folder/../baseline.json",
        "missing": tmp_path / "missing.json",
        "symlink": tmp_path / "link.json",
    }[reference]
    if reference == "symlink":
        path.symlink_to(request.baseline.path)
    request = replace(request, baseline=replace(request.baseline, path=path))
    with pytest.raises(
        evaluation.CapturedEvaluationError,
        match="captured input could not be read safely",
    ):
        evaluation.preflight_captured_request(request, unsigned=True)
    assert not request.evidence.exists()


@pytest.mark.parametrize(
    "field", ["ids", "input", "expected", "metadata", "provenance"]
)
def test_preflight_rejects_pair_or_provenance_drift_without_scoring(
    tmp_path, monkeypatch, field
):
    baseline, subject, policy = example_project("judge")
    if field == "ids":
        subject["records"].pop()
        message = "record IDs differ"
    elif field == "provenance":
        next(iter(subject["score_provenance"].values()))["version"] = "unapproved"
        message = "scorer provenance differs"
    else:
        subject["records"][0][field] = (
            {"changed": "yes"} if field == "metadata" else "changed"
        )
        message = f"{field} changed between runs"
    request = _request(tmp_path, baseline, subject, policy)
    forbidden = Mock(
        side_effect=AssertionError("preflight performed scoring or publication")
    )
    monkeypatch.setattr(evaluation, "compare_runs", forbidden)
    monkeypatch.setattr(evaluation, "publish_captured_evidence", forbidden)
    with pytest.raises(evaluation.CapturedEvaluationError, match=message):
        evaluation.preflight_captured_request(request, unsigned=True)
    forbidden.assert_not_called()
    assert not request.evidence.exists()


@pytest.mark.parametrize(
    "key_kind", ["missing", "malformed", "public", "wrong_algorithm"]
)
def test_signed_preflight_refuses_unusable_keys_without_signing(
    tmp_path, monkeypatch, key_kind
):
    request = frozen_request(tmp_path)
    key = tmp_path / "candidate.pem"
    if key_kind == "malformed":
        key.write_bytes(b"not a PEM key")
    elif key_kind in {"public", "wrong_algorithm"}:
        private = ec.generate_private_key(ec.SECP256R1())
        key.write_bytes(
            private.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            if key_kind == "public"
            else private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    forbidden = Mock(
        side_effect=AssertionError("refused key reached signing or publication")
    )
    monkeypatch.setattr(evaluation, "compare_runs", forbidden)
    monkeypatch.setattr(evaluation, "publish_captured_evidence", forbidden)
    monkeypatch.setattr(
        "invarlock.captured_evidence_publication.manifest_signature", forbidden
    )
    with pytest.raises(evaluation.CapturedEvaluationError, match="signing key"):
        evaluation.preflight_captured_request(request, signing_key_path=key)
    forbidden.assert_not_called()
    assert not request.evidence.exists()


@pytest.mark.parametrize("side", ["baseline", "subject"])
def test_signed_preflight_checks_pins_before_key_loading(tmp_path, monkeypatch, side):
    request = frozen_request(tmp_path)
    request = replace(
        request,
        **{
            side: replace(
                getattr(request, side), expected_run_digest="sha256:" + "0" * 64
            )
        },
    )
    forbidden = Mock(side_effect=AssertionError("bad run pin reached signing key"))
    monkeypatch.setattr(
        "invarlock.captured_evidence_publication._private_key", forbidden
    )
    with pytest.raises(
        evaluation.CapturedEvaluationError,
        match=f"{side} run digest differs from expected pin",
    ):
        evaluation.preflight_captured_request(
            request, signing_key_path=tmp_path / "signer.pem"
        )
    forbidden.assert_not_called()
    assert not request.evidence.exists()


def test_real_invalid_scoring_reference_fails_after_structural_preflight(
    tmp_path, monkeypatch
):
    baseline, subject, policy = example_project("classification")
    policy["metrics"] = [policy["metrics"][0]]
    policy["metrics"][0].update(kind="numeric_tolerance", configuration={})
    for run in (baseline, subject):
        run["records"][0]["expected"] = "not a numeric target"
    request = _request(tmp_path, baseline, subject, policy)
    assert (
        evaluation.preflight_captured_request(request, unsigned=True).record_count == 40
    )
    forbidden = Mock(
        side_effect=AssertionError("invalid reference reached publication")
    )
    monkeypatch.setattr(evaluation, "publish_captured_evidence", forbidden)
    with pytest.raises(
        evaluation.CapturedEvaluationError,
        match="captured comparison failed:.*numeric reference",
    ):
        evaluation.evaluate_captured_request(request, unsigned=True)
    forbidden.assert_not_called()
    assert not request.evidence.exists()


def test_output_collision_after_preflight_preserves_foreign_directory(
    tmp_path, monkeypatch
):
    request = frozen_request(tmp_path)
    compare = evaluation.compare_runs

    def concurrent_destination(*args, **kwargs):
        result = compare(*args, **kwargs)
        request.evidence.mkdir()
        (request.evidence / "foreign").write_bytes(b"must survive")
        return result

    monkeypatch.setattr(evaluation, "compare_runs", concurrent_destination)
    with pytest.raises(evaluation.CapturedEvaluationError, match="already exists"):
        evaluation.evaluate_captured_request(request, unsigned=True)
    assert list(request.evidence.iterdir()) == [request.evidence / "foreign"]
    assert (request.evidence / "foreign").read_bytes() == b"must survive"


def test_transaction_json_and_published_comparison_match_frozen_vectors(tmp_path):
    request = frozen_request(tmp_path)
    preflight = evaluation.preflight_captured_request(request, unsigned=True)
    result = evaluation.evaluate_captured_request(request, unsigned=True)
    payload = json.loads(result.as_json())
    assert payload == {
        "format_version": "invarlock/evaluation-result-v2",
        "kind": "captured",
        "ok": True,
        "evidence": str(request.evidence),
        "comparison_id": _digest("identity-pass.json"),
        "baseline_run_digest": _digest("baseline.json"),
        "subject_run_digest": _digest("subject-pass.json"),
        "policy_digest": _digest("policy.json"),
        "authentication": "unsigned_local",
        "policy_verdict": "pass",
        "decision": "pass",
        "pack_manifest_digest": _digest("manifest-pass-unsigned.json"),
        "request_digest": preflight.request_digest,
    }
    assert result.as_json().encode() == canonical_json_bytes(payload)
    assert (request.evidence / "reports/evaluation.report.json").read_bytes() == _raw(
        "comparison-pass.json"
    )
    assert result.metric_summaries
