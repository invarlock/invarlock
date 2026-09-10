"""SDK request identities and shared transactions, without native placeholders."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from invarlock import captured_evaluation, engine
from invarlock.evidence_pack_integrity import public_key_fingerprint


def _bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _digest(value):
    return "sha256:" + hashlib.sha256(_bytes(value)).hexdigest()


def _key(path):
    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return public_key_fingerprint(key.public_key())


def _inputs(root):
    baseline = engine.make_run(
        [
            {"id": str(i), "input": "question", "expected": "yes", "output": "yes"}
            for i in range(4)
        ],
        source={"name": "test", "version": "1"},
        run_id="baseline",
        artifact_digest="sha256:" + "a" * 64,
    )
    subject = {**baseline, "run_id": "subject"}
    policy = {
        "format": "invarlock/comparison-policy-v1",
        "metrics": [
            {
                "name": "accuracy",
                "kind": "exact_match",
                "configuration": {},
                "direction": "higher",
                "unit": "score",
                "aggregation": "mean",
                "minimum_count": 2,
                "maximum_regression": 1,
                "maximum_interval_width": 2,
            }
        ],
        "slices": [],
    }
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {
                "path": "baseline.json",
                "adapter": "invarlock",
                "expected_run_digest": _digest(baseline),
            },
            "subject": {
                "path": "subject.json",
                "adapter": "invarlock",
                "expected_run_digest": _digest(subject),
            },
            "policy": "policy.json",
        },
        "output": {"evidence": "artifacts/evidence"},
    }
    for name, value in (
        ("baseline", baseline),
        ("subject", subject),
        ("policy", policy),
        ("request", request),
    ):
        (root / f"{name}.json").write_bytes(_bytes(value))
    return request, baseline, subject, policy


def _profile(root, request, baseline, subject, policy, signer):
    verifier = _key(root / "verifier.pem")
    normalized = engine.normalize_captured_request(
        request, baseline=baseline, subject=subject, policy=policy
    )
    profile = {
        "format": "invarlock/trust-inputs-v2",
        "kind": "captured",
        "policy": {"path": "policy.json"},
        "anchors": {
            "baseline_run_digest": _digest(baseline),
            "subject_run_digest": _digest(subject),
            "request_digest": _digest(normalized),
            "evidence_signer_fingerprint": signer,
        },
        "verifier": {"identity": "test-verifier", "signing_key_path": "verifier.pem"},
    }
    (root / "trust.json").write_bytes(_bytes(profile))
    return profile, verifier


def test_normalization_matches_independent_path_free_projection(tmp_path):
    request, baseline, subject, policy = _inputs(tmp_path)
    expected = {
        "format": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {
                "adapter": "invarlock",
                "expected_run_digest": _digest(baseline),
                "run_digest": _digest(baseline),
            },
            "subject": {
                "adapter": "invarlock",
                "expected_run_digest": _digest(subject),
                "run_digest": _digest(subject),
            },
            "policy_digest": _digest(policy),
        },
    }
    actual = engine.normalize_captured_request(
        MappingProxyType(request), baseline=baseline, subject=subject, policy=policy
    )
    assert _bytes(actual) == _bytes(expected)
    assert engine.captured_request_digest(actual) == _digest(expected)
    assert engine.comparison_policy_digest(policy) == _digest(policy)
    request["output"]["evidence"] = "elsewhere/out"
    request["comparison"]["baseline"]["path"] = "other/location"
    assert (
        engine.normalize_captured_request(
            request, baseline=baseline, subject=subject, policy=policy
        )
        == expected
    )
    with pytest.raises(engine.EvaluationRecordsError):
        engine.captured_request_digest(request)
    assert not hasattr(engine, "EvaluationRecordError")
    assert len(engine.__all__) == len(set(engine.__all__))


def test_every_declared_source_intent_is_bound_and_checked(tmp_path):
    request, baseline, subject, policy = _inputs(tmp_path)
    spec = request["comparison"]["baseline"]
    spec.update(
        adapter="jsonl",
        source=baseline["source"],
        run_id=baseline["run_id"],
        artifact_digest=baseline["artifact_digest"],
        score_provenance={},
    )
    baseline["source_digest"] = "sha256:" + "b" * 64
    spec["expected_run_digest"] = _digest(baseline)
    actual = engine.normalize_captured_request(
        request, baseline=baseline, subject=subject, policy=policy
    )
    assert actual["comparison"]["baseline"] == {
        **{k: v for k, v in spec.items() if k != "path"},
        "run_digest": _digest(baseline),
    }
    for field, replacement in (
        ("run_id", "other"),
        ("source", {"name": "other", "version": "1"}),
        ("artifact_digest", "sha256:" + "c" * 64),
    ):
        changed = copy.deepcopy(request)
        changed["comparison"]["baseline"][field] = replacement
        with pytest.raises(engine.EvaluationRecordsError, match="source intent"):
            engine.normalize_captured_request(
                changed, baseline=baseline, subject=subject, policy=policy
            )
    actual["comparison"]["baseline"]["source"]["name"] = "mutated"
    assert spec["source"]["name"] == "test"


def test_portable_policy_hash_does_not_require_local_unicode(tmp_path, monkeypatch):
    request, baseline, subject, policy = _inputs(tmp_path)
    policy["metrics"][0].update(
        kind="token_f1", configuration={"unicode_version": "999.0.0"}
    )
    forbidden = Mock(side_effect=AssertionError("portable identity must not score"))
    monkeypatch.setattr("invarlock.core.scoring.score", forbidden)
    assert engine.comparison_policy_digest(policy) == _digest(policy)
    normalized = engine.normalize_captured_request(
        request, baseline=baseline, subject=subject, policy=policy
    )
    assert engine.captured_request_digest(normalized) == _digest(normalized)
    forbidden.assert_not_called()
    policy["metrics"][0]["configuration"]["casefold"] = "yes"
    with pytest.raises(engine.EvaluationRecordsError):
        engine.comparison_policy_digest(policy)


def test_loader_overrides_precede_superseded_existence_checks(tmp_path, monkeypatch):
    root = tmp_path / "request-root"
    root.mkdir()
    request, baseline, subject, policy = _inputs(root)
    (root / "old-output").mkdir()
    request["comparison"]["baseline"]["path"] = "missing-original.json"
    request["output"]["evidence"] = "old-output"
    (root / "request.json").write_bytes(_bytes(request))
    monkeypatch.chdir(tmp_path)
    loaded = engine.load_evaluation_request(
        root / "request.json",
        baseline_run=Path("request-root/baseline.json"),
        output=Path("request-root/new/output"),
    )
    assert loaded.baseline.path == root / "baseline.json"
    assert loaded.evidence == root / "new/output"
    assert loaded.baseline.expected_run_digest == _digest(baseline)
    result = engine.preflight_evaluation_request(
        loaded, signing_key_path=None, unsigned=True
    )
    assert result.output == str(root / "new/output")
    assert not (root / "new").exists()
    request["comparison"]["baseline"]["path"] = "../unsafe.json"
    (root / "request.json").write_bytes(_bytes(request))
    with pytest.raises(engine.EvaluationRequestError):
        engine.load_evaluation_request(
            root / "request.json",
            baseline_run=root / "baseline.json",
            output=root / "new/output",
        )


@pytest.mark.parametrize(
    "override", ["../escape", "/outside", "baseline.json/../subject.json"]
)
def test_loader_overrides_remain_root_confined(tmp_path, monkeypatch, override):
    _inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(engine.EvaluationRequestError):
        engine.load_evaluation_request(tmp_path / "request.json", baseline_run=override)


def test_sdk_preflight_and_signed_publication_share_bindings_without_runtime(
    tmp_path, monkeypatch
):
    request, baseline, subject, policy = _inputs(tmp_path)
    signer = _key(tmp_path / "evidence-signer.pem")
    loaded = engine.load_evaluation_request(tmp_path / "request.json")
    forbidden = Mock(
        side_effect=AssertionError(
            "preflight must not score, sign or discover providers"
        )
    )
    with monkeypatch.context() as context:
        context.setattr("invarlock.evaluation_transaction.CoreRegistry", forbidden)
        context.setattr(captured_evaluation, "compare_runs", forbidden)
        context.setattr(
            "invarlock.captured_evidence_publication.manifest_signature", forbidden
        )
        unsigned = engine.preflight_evaluation_request(
            loaded, signing_key_path=None, unsigned=True
        )
        signed = engine.preflight_evaluation_request(
            tmp_path / "request.json", signing_key_path=tmp_path / "evidence-signer.pem"
        )
    assert signed.request_digest == unsigned.request_digest
    assert signed.baseline_run_digest == _digest(baseline)
    assert signed.max_bootstrap_draws == engine.DEFAULT_MAX_BOOTSTRAP_DRAWS
    assert not (tmp_path / "artifacts").exists()
    assert (
        not {
            "decision",
            "policy_verdict",
            "pack_manifest_digest",
            "receipt",
            "scoring_assurance",
        }
        & json.loads(signed.as_json()).keys()
    )
    published = engine.evaluate_request_file(
        loaded, signing_key_path=tmp_path / "evidence-signer.pem"
    )
    assert isinstance(published, engine.CapturedEvaluationTransactionResult)
    assert published.request_digest == unsigned.request_digest
    assert (
        published.pack_manifest_digest
        == "sha256:"
        + hashlib.sha256(
            (published.evidence_path / "manifest.json").read_bytes()
        ).hexdigest()
    )
    profile, verifier = _profile(tmp_path, request, baseline, subject, policy, signer)
    trust = engine.load_trust_inputs(tmp_path / "trust.json")
    assert isinstance(trust, engine.CapturedTrustInputs)
    assert trust.profile_digest == _digest(profile)
    original_policy = trust.policy_bytes
    (tmp_path / "policy.json").write_text("changed after acquisition")
    (tmp_path / "verifier.pem").write_text("changed after acquisition")
    result = engine.verify_evidence(
        published.evidence_path,
        policy_path=trust.policy_path,
        policy_bytes=trust.policy_bytes,
        expected_baseline_run=trust.expected_run_digests["baseline"],
        expected_subject_run=trust.expected_run_digests["subject"],
        expected_request_digest=trust.expected_request_digest,
        expected_signer=trust.expected_signer_fingerprint,
        verifier_signing_key_path=trust.verifier_signing_key_path,
        verifier_signing_key_bytes=trust.verifier_signing_key_bytes,
        verifier_identity=trust.verifier_identity,
        trust_profile_digest=trust.profile_digest,
        receipt_path=tmp_path / "receipt.json",
    )
    assert isinstance(result, engine.EvidenceVerification)
    assert result.payload["ok"] is True
    (tmp_path / "policy.json").write_bytes(original_policy)
    authenticated = engine.verify_signed_verification_receipt(
        tmp_path / "receipt.json",
        published.evidence_path,
        policy_path=tmp_path / "policy.json",
        expected_run_digests=dict(trust.expected_run_digests),
        expected_request_digest=trust.expected_request_digest,
        expected_pack_signer_fingerprint=signer,
        expected_verifier_identity=trust.verifier_identity,
        expected_verifier_fingerprint=verifier,
        expected_trust_profile_digest=trust.profile_digest,
    )
    assert authenticated.ok, authenticated.errors


@pytest.mark.parametrize(
    "operation", [engine.evaluate_request_file, engine.preflight_evaluation_request]
)
@pytest.mark.parametrize(
    "extra",
    [
        {"registry": None},
        {"scorer_registry": None},
        {"runtime_image_digests": None},
        {"resource_resolver": None},
    ],
)
def test_captured_sdk_rejects_explicit_null_runtime_arguments(
    tmp_path, operation, extra
):
    _inputs(tmp_path)
    loaded = engine.load_evaluation_request(tmp_path / "request.json")
    with pytest.raises(
        (engine.EvaluationPreflightError, engine.EvaluationTransactionError),
        match="runtime/scorer",
    ):
        operation(loaded, signing_key_path=None, unsigned=True, **extra)
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    "unsigned,key", [(False, None), (False, "invalid.pem"), (True, "invalid.pem")]
)
def test_preflight_signing_failures_use_captured_error_contract(
    tmp_path, unsigned, key
):
    _inputs(tmp_path)
    with pytest.raises(engine.EvaluationPreflightError) as failure:
        engine.preflight_evaluation_request(
            tmp_path / "request.json",
            signing_key_path=tmp_path / key if key else None,
            unsigned=unsigned,
        )
    payload = json.loads(failure.value.as_json())
    assert payload["format_version"] == "invarlock/evaluation-preflight-v3"
    assert payload["kind"] == "captured"
    assert payload["ok"] is False
    assert not (tmp_path / "artifacts").exists()


def test_acquired_export_bytes_are_not_reopened_by_parser(tmp_path, monkeypatch):
    request, baseline, subject, policy = _inputs(tmp_path)
    source = {"name": "external", "version": "1"}
    raw = b"\n".join(_bytes(row).rstrip(b"\n") for row in baseline["records"]) + b"\n"
    (tmp_path / "export.jsonl").write_bytes(raw)
    imported = engine.load_run(
        tmp_path / "export.jsonl",
        adapter="jsonl",
        source=source,
        run_id="export",
        artifact_digest=baseline["artifact_digest"],
    )
    request["comparison"]["baseline"] = {
        "path": "export.jsonl",
        "adapter": "jsonl",
        "source": source,
        "run_id": "export",
        "artifact_digest": baseline["artifact_digest"],
        "expected_run_digest": _digest(imported),
    }
    (tmp_path / "request.json").write_bytes(_bytes(request))
    original = captured_evaluation._parse_run_bytes

    def parse(raw, **kwargs):
        (tmp_path / "export.jsonl").write_text("replacement must not be read")
        return original(raw, **kwargs)

    monkeypatch.setattr(captured_evaluation, "_parse_run_bytes", parse)
    result = engine.preflight_evaluation_request(
        tmp_path / "request.json", signing_key_path=None, unsigned=True
    )
    assert result.baseline_run_digest == _digest(imported)
    assert imported["source_digest"] == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_source_symlink_replacement_after_loading_is_rejected(tmp_path):
    _inputs(tmp_path)
    loaded = engine.load_evaluation_request(tmp_path / "request.json")
    (tmp_path / "baseline.json").unlink()
    (tmp_path / "baseline.json").symlink_to(tmp_path / "subject.json")
    with pytest.raises(engine.EvaluationPreflightError):
        engine.preflight_evaluation_request(
            loaded, signing_key_path=None, unsigned=True
        )
    assert not (tmp_path / "artifacts").exists()


def test_captured_verification_rejects_explicit_native_null_anchor(tmp_path):
    _inputs(tmp_path)
    result = engine.evaluate_request_file(
        tmp_path / "request.json", signing_key_path=None, unsigned=True
    )
    with pytest.raises(engine.EvidenceVerificationError, match="native anchors"):
        engine.verify_evidence(
            result.evidence_path,
            policy_path=tmp_path / "policy.json",
            expected_signer=None,
            expected_baseline_artifact=None,
        )


@pytest.mark.parametrize(
    "field,value",
    [("kind", "runtime"), ("allow_installed_scorers", False), ("unexpected", 1)],
)
def test_captured_trust_profile_has_closed_mode_specific_shape(tmp_path, field, value):
    request, baseline, subject, policy = _inputs(tmp_path)
    profile, _ = _profile(
        tmp_path, request, baseline, subject, policy, "sha256:" + "a" * 64
    )
    profile[field] = value
    (tmp_path / "trust.json").write_bytes(_bytes(profile))
    with pytest.raises(engine.TrustInputsError):
        engine.load_trust_inputs(tmp_path / "trust.json")


def test_trust_profile_public_key_override_retains_original_profile_digest(tmp_path):
    request, baseline, subject, policy = _inputs(tmp_path)
    profile, _ = _profile(
        tmp_path, request, baseline, subject, policy, "sha256:" + "a" * 64
    )
    (tmp_path / "verifier.pem").unlink()
    public = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    trust = engine.load_trust_inputs(
        tmp_path / "trust.json", verifier_key_bytes_override=public
    )
    assert trust.profile_digest == _digest(profile)
    assert trust.verifier_signing_key_bytes == public


@pytest.mark.parametrize(
    "field",
    [
        "baseline_run_digest",
        "subject_run_digest",
        "request_digest",
        "evidence_signer_fingerprint",
    ],
)
def test_profile_digest_anchors_reject_newline_suffixes(tmp_path, field):
    request, baseline, subject, policy = _inputs(tmp_path)
    profile, _ = _profile(
        tmp_path, request, baseline, subject, policy, "sha256:" + "a" * 64
    )
    profile["anchors"][field] += "\n"
    (tmp_path / "trust.json").write_bytes(_bytes(profile))
    with pytest.raises(engine.TrustInputsError):
        engine.load_trust_inputs(tmp_path / "trust.json")


def test_native_output_override_does_not_require_superseded_destination(tmp_path):
    import yaml

    from tests.core.test_evaluation_request_contract import _valid_request

    path = _valid_request(tmp_path)
    request = yaml.safe_load(path.read_text())
    (tmp_path / request["output"]["evidence"]).mkdir(parents=True)
    loaded = engine.load_evaluation_request(path, output=tmp_path / "replacement")
    assert isinstance(loaded, engine.EvaluationRequest)
    assert loaded.output.evidence == tmp_path / "replacement"
    with pytest.raises(engine.EvaluationRequestError, match="captured"):
        engine.load_evaluation_request(
            path,
            baseline_run=tmp_path / "models/baseline",
            output=tmp_path / "replacement",
        )


def test_run_io_is_no_clobber_and_refuses_unsafe_paths(tmp_path):
    _, baseline, _, _ = _inputs(tmp_path)
    path = tmp_path / "export.json"
    assert engine.write_run(path, MappingProxyType(baseline)) == path
    assert engine.load_run(path) == baseline
    assert (
        engine.physical_file_digest(path)
        == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(engine.EvaluationRecordsError):
        engine.write_run(path, baseline)
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    for operation, unsafe in (
        (engine.load_run, tmp_path / "link/export.json"),
        (engine.physical_file_digest, tmp_path / "link/export.json"),
    ):
        with pytest.raises(engine.EvaluationRecordsError):
            operation(unsafe)
    with pytest.raises(engine.EvaluationRecordsError):
        engine.write_run(tmp_path / "link/new.json", baseline)
    assert not (tmp_path / "new.json").exists()


def test_planned_case_helpers_accept_readonly_mappings_and_do_not_mutate():
    cases = [
        {"id": "b", "input": {"parts": [2, 1]}, "expected": "yes", "metadata": {}},
        {"id": "a", "input": {"parts": [1, 2]}, "expected": "yes", "metadata": {}},
    ]
    frozen = engine.freeze_case_set(tuple(MappingProxyType(case) for case in cases))
    assert [case["id"] for case in frozen["cases"]] == ["a", "b"]
    assert [case["id"] for case in cases] == ["b", "a"]
    expected = engine.case_set_digest(MappingProxyType(frozen))
    cases[0]["input"]["parts"].reverse()
    assert engine.case_set_digest(frozen) == expected
    assert not any(
        hasattr(engine, name)
        for name in ("digest", "validate", "example_project", "PipelineError")
    )


@pytest.mark.parametrize(
    "failure,exit_code,receipt_exists", [("budget", 2, False), ("policy", 7, True)]
)
def test_verification_distinguishes_incomplete_work_and_completed_policy_failure(
    tmp_path, failure, exit_code, receipt_exists
):
    import unicodedata

    request, baseline, subject, policy = _inputs(tmp_path)
    if failure == "budget":
        policy["metrics"][0].update(
            kind="token_f1",
            configuration={"unicode_version": unicodedata.unidata_version},
        )
    else:
        policy["metrics"][0]["subject_minimum"] = 2
    (tmp_path / "policy.json").write_bytes(_bytes(policy))
    signer = _key(tmp_path / "evidence-signer.pem")
    pack = engine.evaluate_request_file(
        tmp_path / "request.json", signing_key_path=tmp_path / "evidence-signer.pem"
    ).evidence_path
    _profile(tmp_path, request, baseline, subject, policy, signer)
    trust = engine.load_trust_inputs(tmp_path / "trust.json")
    receipt = tmp_path / "result.receipt.json"
    with pytest.raises(engine.EvidenceVerificationError) as caught:
        engine.verify_evidence(
            pack,
            policy_path=trust.policy_path,
            expected_baseline_run=trust.expected_run_digests["baseline"],
            expected_subject_run=trust.expected_run_digests["subject"],
            expected_request_digest=trust.expected_request_digest,
            expected_signer=trust.expected_signer_fingerprint,
            receipt_path=receipt,
            verifier_signing_key_path=trust.verifier_signing_key_path,
            verifier_identity=trust.verifier_identity,
            max_bootstrap_draws=0,
        )
    assert caught.value.exit_code == exit_code
    assert receipt.exists() is receipt_exists
    payload = json.loads(caught.value.as_json())
    assert payload["format_version"] == "invarlock/evidence-pack-verify-v2"
    assert payload["integrity_ok"] is (True if receipt_exists else None)
