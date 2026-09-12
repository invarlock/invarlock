from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives import serialization

from invarlock.captured_contracts import sha
from invarlock.captured_evidence_publication import publish_captured_evidence
from invarlock.captured_normalization import (
    captured_request_digest,
    normalize_captured_request,
)
from invarlock.evaluation_comparison.comparison import compare_runs
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_sets.contracts import (
    SCOPE,
    STATISTICAL_SCOPE,
    write_evidence_set_index,
)
from invarlock.evidence_sets.verification import (
    shared_captured_inputs,
    verify_evidence_set,
    verify_stored_evidence_set_result,
)
from tests.judge_measurements.test_evidence_acceptance import KEY, _publish


def write(path, value):
    path.write_bytes(canonical_json_bytes(value))


def fixture(
    tmp_path,
    *,
    role="required",
    incomplete=False,
    captured_minimum=1,
    captured_change=None,
):
    root = tmp_path / "set"
    root.mkdir()
    staging = tmp_path / "judge-source"
    staging.mkdir()
    publication, judge_policy = _publish(staging, role=role, incomplete=incomplete)
    publication.path.rename(root / "judge")
    trust = tmp_path / "trust"
    trust.mkdir()
    judge_policy.rename(trust / "judge.json")
    baseline = json.loads((root / "judge/baseline_run.json").read_text())
    subject = json.loads((root / "judge/subject_run.json").read_text())
    if captured_change:
        captured_change(baseline, subject)
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
                "minimum_count": captured_minimum,
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
            "baseline": {"path": "baseline.json", "adapter": "invarlock"},
            "subject": {"path": "subject.json", "adapter": "invarlock"},
            "policy": "policy.json",
        },
        "output": {"evidence": "evidence"},
    }
    normalized = normalize_captured_request(
        request, baseline=baseline, subject=subject, policy=policy
    )
    request_digest = captured_request_digest(normalized)
    key = trust / "key.pem"
    key.write_bytes(
        KEY.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key.chmod(0o600)
    publish_captured_evidence(
        root / "deterministic",
        baseline=baseline,
        subject=subject,
        policy=policy,
        comparison=compare_runs(baseline, subject, policy),
        request_digest=request_digest,
        signing_key_path=key,
        unsigned=False,
        normalized_request=normalized,
    )
    shared = shared_captured_inputs({"baseline": baseline, "subject": subject})
    write(trust / "policy.json", policy)
    write(
        trust / "captured.json",
        {
            "format": "invarlock/trust-inputs-v2",
            "kind": "captured",
            "policy": {"path": "policy.json"},
            "anchors": {
                "baseline_run_digest": shared["baseline_run_sha256"],
                "subject_run_digest": shared["subject_run_sha256"],
                "request_digest": request_digest,
                "evidence_signer_fingerprint": public_key_fingerprint(KEY.public_key()),
            },
            "verifier": {"identity": "recipient", "signing_key_path": "key.pem"},
        },
    )
    index = write_evidence_set_index(root, deterministic="deterministic", judge="judge")
    composition = {
        "format": "invarlock/evidence-set-recipient-policy-v1",
        "scope": SCOPE,
        "index_sha256": sha(index.read_bytes()),
        "shared_inputs": shared,
        "members": {
            name: {
                "kind": kind,
                "trust_profile": filename,
                "trust_profile_sha256": sha((trust / filename).read_bytes()),
                "role": "required",
            }
            for name, kind, filename in [
                ("deterministic", "captured", "captured.json"),
                ("judge", "judge", "judge.json"),
            ]
        },
        "decision_rule": "all-required-components-pass",
        "statistical_scope": STATISTICAL_SCOPE,
    }
    write(trust / "composition.json", composition)
    return root, trust / "composition.json"


def test_same_original_runs_verify_and_replay_stored_result(tmp_path):
    root, policy = fixture(tmp_path)
    receipt = tmp_path / "result.json"
    result = verify_evidence_set(root, recipient_policy=policy, receipt=receipt)
    assert result.accepted, result.payload
    assert result.payload["shared_inputs_verified"]
    assert result.payload["statistical_scope"] == STATISTICAL_SCOPE
    assert all(member["receipt"] for member in result.payload["members"].values())
    assert verify_stored_evidence_set_result(
        receipt, root, recipient_policy=policy
    ).accepted
    modified = json.loads(receipt.read_text())
    modified["members"]["judge"]["decision"] = "regression"
    write(receipt, modified)
    assert not verify_stored_evidence_set_result(
        receipt, root, recipient_policy=policy
    ).accepted


@pytest.mark.parametrize("field", ["input", "expected", "output", "metadata"])
def test_same_ids_do_not_allow_different_case_content(tmp_path, field):
    def mutate(baseline, subject):
        for run in (baseline, subject):
            run["records"][0][field] = (
                {"changed": "yes"} if field == "metadata" else "changed"
            )

    root, policy = fixture(tmp_path, captured_change=mutate)
    result = verify_evidence_set(root, recipient_policy=policy)
    assert not result.accepted and not result.payload["verified"]
    assert "shared recipient pin" in str(result.payload["errors"])


@pytest.mark.parametrize(
    "options", [{"role": "advisory"}, {"incomplete": True}, {"captured_minimum": 20}]
)
def test_component_nonacceptance_cannot_be_rescued(tmp_path, options):
    root, policy = fixture(tmp_path, **options)
    result = verify_evidence_set(root, recipient_policy=policy)
    assert not result.accepted
    if "role" not in options:
        assert result.payload["verified"] and result.exit_code == 7
        assert result.payload["decision"] == "insufficient_evidence"


@pytest.mark.parametrize(
    "target",
    [
        "index_sha256",
        "shared_inputs.case_set_sha256",
        "shared_inputs.subject_artifact_sha256",
        "members.judge.trust_profile_sha256",
    ],
)
def test_wrong_recipient_pins_reject(tmp_path, target):
    root, path = fixture(tmp_path)
    policy = json.loads(path.read_text())
    value = policy
    parts = target.split(".")
    for part in parts[:-1]:
        value = value[part]
    value[parts[-1]] = "sha256:" + "f" * 64
    write(path, policy)
    assert not verify_evidence_set(root, recipient_policy=path).accepted


def test_receipt_cannot_be_written_into_any_member(tmp_path):
    root, policy = fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        verify_evidence_set(
            root, recipient_policy=policy, receipt=root / "judge/result.json"
        )
    assert not (root / "judge/result.json").exists()


def test_recipient_profile_inside_set_is_rejected(tmp_path):
    root, path = fixture(tmp_path)
    submitted = root / "policy.json"
    submitted.write_bytes(path.read_bytes())
    result = verify_evidence_set(root, recipient_policy=submitted)
    assert not result.accepted and "outside" in str(result.payload["errors"])


def test_statement_mutation_rejected(tmp_path):
    root, path = fixture(tmp_path)
    envelope = root / "judge/envelope.json"
    envelope.write_bytes(envelope.read_bytes() + b" ")
    assert not verify_evidence_set(root, recipient_policy=path).accepted


def test_component_profile_symlink_rejected(tmp_path):
    root, path = fixture(tmp_path)
    profile = path.parent / "judge.json"
    actual = path.parent / "actual.json"
    profile.rename(actual)
    profile.symlink_to(actual)
    assert not verify_evidence_set(root, recipient_policy=path).accepted


def test_malformed_judge_profile_is_a_structured_rejection(tmp_path):
    root, path = fixture(tmp_path)
    profile = path.parent / "judge.json"
    malformed = json.loads(profile.read_text())
    malformed["bindings"] = []
    write(profile, malformed)
    policy = json.loads(path.read_text())
    policy["members"]["judge"]["trust_profile_sha256"] = sha(profile.read_bytes())
    write(path, policy)
    result = verify_evidence_set(root, recipient_policy=path)
    assert not result.accepted and "invalid" in str(result.payload["errors"])


def test_recorded_policy_cannot_enter_deterministic_component():
    from invarlock.evaluation_records.templates import example_project
    from invarlock.evidence_sets.verification import require_deterministic_policy

    _, _, policy = example_project("judge")
    with pytest.raises(ValueError, match="recomputed"):
        require_deterministic_policy(policy)


def test_case_set_reconstruction_checks_both_sides(tmp_path):
    root, _ = fixture(tmp_path)
    baseline = json.loads((root / "judge/baseline_run.json").read_text())
    subject = json.loads((root / "judge/subject_run.json").read_text())
    subject["records"][0]["expected"] = "changed"
    with pytest.raises(ValueError, match="case sets differ"):
        shared_captured_inputs({"baseline": baseline, "subject": subject})


@pytest.mark.parametrize("maximum", [-1, True, 1.5])
def test_invalid_replay_budget_is_rejected(tmp_path, maximum):
    root, policy = fixture(tmp_path)
    result = verify_evidence_set(
        root, recipient_policy=policy, max_bootstrap_draws=maximum
    )
    assert not result.accepted and result.exit_code == 4


def test_receipt_no_clobber(tmp_path):
    root, policy = fixture(tmp_path)
    receipt = tmp_path / "existing.json"
    receipt.write_text("preserve")
    with pytest.raises((ValueError, OSError)):
        verify_evidence_set(root, recipient_policy=policy, receipt=receipt)
    assert receipt.read_text() == "preserve"


def test_unsigned_judge_is_not_accepted(tmp_path):
    root, path = fixture(tmp_path)
    envelope = root / "judge/envelope.json"
    value = json.loads(envelope.read_text())
    value.update(signer=None, signature=None)
    write(envelope, value)
    index = json.loads((root / "evidence-set.json").read_text())
    index["members"]["judge"]["statement_sha256"] = sha(envelope.read_bytes())
    write(root / "evidence-set.json", index)
    policy = json.loads(path.read_text())
    policy["index_sha256"] = sha((root / "evidence-set.json").read_bytes())
    write(path, policy)
    result = verify_evidence_set(root, recipient_policy=path)
    assert (
        not result.accepted and not result.payload["members"]["judge"]["authenticated"]
    )


@pytest.mark.parametrize(
    "mutation", ["recipient_policy", "index", "profile", "analysis_policy"]
)
def test_files_changed_after_component_verification_reject(
    tmp_path, monkeypatch, mutation
):
    from invarlock.evidence_sets import verification

    root, path = fixture(tmp_path)
    original = verification.verify_judge_evidence

    def mutate(*args, **kwargs):
        receipt = original(*args, **kwargs)
        target = {
            "recipient_policy": path,
            "index": root / "evidence-set.json",
            "profile": path.parent / "captured.json",
            "analysis_policy": root / "judge/analysis_policy.json",
        }[mutation]
        if mutation == "analysis_policy":
            content = json.loads(target.read_text())
            content["metric_name"] = "changed"
            write(target, content)
        else:
            target.write_bytes(target.read_bytes() + b" ")
        return receipt

    monkeypatch.setattr(verification, "verify_judge_evidence", mutate)
    result = verify_evidence_set(root, recipient_policy=path)
    assert not result.accepted and "changed" in str(result.payload["errors"])


def test_wrong_judge_subject_pin_rejected(tmp_path):
    root, path = fixture(tmp_path)
    profile = path.parent / "judge.json"
    value = json.loads(profile.read_text())
    value["intended_subject"] = "sha256:" + "e" * 64
    write(profile, value)
    policy = json.loads(path.read_text())
    policy["members"]["judge"]["trust_profile_sha256"] = sha(profile.read_bytes())
    write(path, policy)
    result = verify_evidence_set(root, recipient_policy=path)
    assert not result.accepted and "intended subject" in str(result.payload["errors"])


def test_captured_signed_rejection_receipt_is_preserved(tmp_path):
    root, path = fixture(tmp_path)
    policy = path.parent / "policy.json"
    value = json.loads(policy.read_text())
    value["metrics"][0]["maximum_interval_width"] = 1
    write(policy, value)
    result = verify_evidence_set(root, recipient_policy=path)
    assert not result.accepted
    member = result.payload["members"]["deterministic"]
    assert member["errors"] and member["receipt"]
    retained = json.loads(member["receipt"])
    assert not retained["statement"]["verdict"]["ok"]


def test_local_captured_replay_failure_does_not_fabricate_receipt(
    tmp_path, monkeypatch
):
    from invarlock.evidence_sets import verification

    root, path = fixture(tmp_path)

    def unavailable(*args, **kwargs):
        raise ValueError("local replay unavailable")

    monkeypatch.setattr(verification, "verify_captured_evidence", unavailable)
    result = verify_evidence_set(root, recipient_policy=path)
    assert not result.accepted
    assert result.payload["members"]["deterministic"]["receipt"] is None
