"""Focused security/contract vectors; all signing material here is test-only."""

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator

from invarlock import captured_contracts as contract
from invarlock import captured_verification as verification
from invarlock.captured_evidence_publication import (
    CapturedEvidenceError,
    publish_captured_evidence,
)
from invarlock.captured_reporting import (
    CapturedReportError,
    is_captured_manifest,
)
from invarlock.evaluation_comparison.comparison import compare_runs
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    digest,
)
from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.evidence_pack_integrity import public_key_fingerprint, verify_signature
from invarlock.evidence_reporting import (
    EvidenceReportError,
    EvidenceReportV2,
    render_evidence,
)
from invarlock.public_contracts import (
    load_evidence_pack_schema,
    load_evidence_pack_v2_schema,
    load_evidence_verification_receipt_v3_schema,
)


def _replace(path, raw):
    path.chmod(0o600)
    path.write_bytes(raw)


@pytest.fixture
def handoff(tmp_path):
    baseline, subject, policy = example_project("classification")
    # RFC 8032 section 7.1 test seed, never a production key.
    key = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
        )
    )
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    key_path = tmp_path / "test-only.pem"
    key_path.write_bytes(pem)
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(canonical_json_bytes(policy))
    normalized = {
        "format": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": {"adapter": "invarlock", "run_digest": digest(baseline)},
            "subject": {"adapter": "invarlock", "run_digest": digest(subject)},
            "policy_digest": digest(policy),
        },
    }
    pack = tmp_path / "evidence"
    publish_args = {
        "baseline": baseline,
        "subject": subject,
        "policy": policy,
        "comparison": compare_runs(baseline, subject, policy),
        "normalized_request": normalized,
        "request_digest": digest(normalized),
        "signing_key_path": key_path,
        "unsigned": False,
    }
    publish_captured_evidence(pack, **publish_args)
    kwargs = {
        "policy_path": policy_path,
        "expected_baseline_run": digest(baseline),
        "expected_subject_run": digest(subject),
        "expected_request_digest": digest(normalized),
        "expected_signer": public_key_fingerprint(key.public_key()),
        "receipt_path": tmp_path / "receipt.json",
        "verifier_signing_key_path": key_path,
        "verifier_identity": "test-verifier",
    }
    return pack, kwargs, publish_args, key, pem


def _read_receipt(handoff, **overrides):
    pack, kwargs, _, key, _ = handoff
    return verification.verify_captured_receipt(
        kwargs["receipt_path"],
        pack,
        **{
            **{
                name: kwargs[name]
                for name in (
                    "policy_path",
                    "expected_baseline_run",
                    "expected_subject_run",
                    "expected_request_digest",
                    "expected_signer",
                )
            },
            "expected_verifier_identity": "test-verifier",
            "expected_verifier_fingerprint": public_key_fingerprint(key.public_key()),
            **overrides,
        },
    )


def _resign_receipt(handoff, mutate):
    _, kwargs, _, key, _ = handoff
    receipt = json.loads(kwargs["receipt_path"].read_bytes())
    mutate(receipt["statement"])
    receipt["signature"]["value"] = base64.b64encode(
        key.sign(canonical_json_bytes(receipt["statement"]))
    ).decode("ascii")
    kwargs["receipt_path"].write_bytes(canonical_json_bytes(receipt))


def _rebind(pack, key):
    manifest = json.loads((pack / "manifest.json").read_bytes())
    payloads = {name: (pack / name).read_bytes() for name in contract.PAYLOADS.values()}
    for role, name in contract.PAYLOADS.items():
        manifest["files"][role]["digest"] = contract.sha(payloads[name])
    manifest["request_digest"] = manifest["files"]["request"]["digest"]
    manifest["comparison_id"] = digest(
        {
            "kind": "captured",
            "request_digest": manifest["request_digest"],
            "baseline_run_digest": manifest["files"]["baseline"]["digest"],
            "subject_run_digest": manifest["files"]["subject"]["digest"],
            "policy_digest": manifest["files"]["policy"]["digest"],
        }
    )
    ledger = contract.checksums(payloads)
    manifest["checksums_sha256_digest"] = hashlib.sha256(ledger).hexdigest()
    raw = canonical_json_bytes(manifest)
    _replace(pack / "checksums.sha256", ledger)
    _replace(pack / "manifest.json", raw)
    _replace(
        pack / "manifest.signature.json",
        canonical_json_bytes(contract.manifest_signature(raw, key)),
    )


def test_manifest_wire_and_existing_signature_convention(handoff):
    pack, kwargs, _, key, _ = handoff
    manifest_raw = (pack / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    Draft202012Validator(load_evidence_pack_v2_schema()).validate(manifest)
    assert not Draft202012Validator(load_evidence_pack_schema()).is_valid(manifest)
    expected_ledger = b"".join(
        hashlib.sha256((pack / name).read_bytes()).hexdigest().encode("ascii")
        + b"  "
        + name.encode("ascii")
        + b"\n"
        for name in (
            "inputs/policy.json",
            "records/baseline.json",
            "records/subject.json",
            "reports/evaluation.report.json",
            "request.json",
        )
    )
    assert (pack / "checksums.sha256").read_bytes() == expected_ledger
    assert (
        manifest["checksums_sha256_digest"]
        == hashlib.sha256(expected_ledger).hexdigest()
    )
    errors, warnings, signer = verify_signature(
        pack, strict=True, expected_fingerprints={kwargs["expected_signer"]}
    )
    assert (errors, warnings, signer) == ([], [], kwargs["expected_signer"])
    signature = json.loads((pack / "manifest.signature.json").read_bytes())
    key.public_key().verify(
        base64.b64decode(signature["signature"]["value"]), manifest_raw
    )
    # Independent canonical encoding, not the publisher's JSON helper.
    assert (
        manifest_raw
        == (
            json.dumps(
                manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            + "\n"
        ).encode()
    )


def test_rfc8032_signature_algorithm_vector():
    key = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
        )
    )
    bundle = contract.manifest_signature(b"", key)
    assert base64.b64decode(bundle["signature"]["value"]).hex() == (
        "e5564300c360ac729086e2cc806e828a"
        "84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd"
        "25bf5f0595bbe24655141438e7a100b"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra", True),
        ("kind", "native"),
        ("signature", "elsewhere.json"),
        ("checksums_sha256_digest", "0" * 63),
        ("signing_key_fingerprint", None),
    ],
)
def test_manifest_is_closed(handoff, field, value):
    pack = handoff[0]
    manifest = json.loads((pack / "manifest.json").read_bytes())
    manifest[field] = value
    assert not Draft202012Validator(load_evidence_pack_v2_schema()).is_valid(manifest)


def test_success_receipt_compact_assurance_and_byte_trust_inputs(handoff):
    pack, kwargs, _, _, pem = handoff
    result = verification.verify_captured_evidence(
        pack,
        **{
            **kwargs,
            "policy_path": Path("not-opened-policy"),
            "verifier_signing_key_path": None,
            "policy_bytes": kwargs["policy_path"].read_bytes(),
            "verifier_signing_key_bytes": pem,
        },
    )
    assert result["ok"] is True
    receipt = json.loads(kwargs["receipt_path"].read_bytes())
    Draft202012Validator(load_evidence_verification_receipt_v3_schema()).validate(
        receipt
    )
    statement = receipt["statement"]
    assert statement["verdict"]["verification_status"] == 0
    assert statement["verification_scope"] == "captured_comparison"
    assert statement["subject"] == {
        "kind": "captured_run",
        "run_digest": kwargs["expected_subject_run"],
    }
    assert all(
        set(metric) == {"name", "slice", "kind", "scoring_assurance"}
        for metric in statement["scoring_assurance"]
    )
    assert _read_receipt(handoff).ok
    assert kwargs["receipt_path"].stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "anchor",
    [
        "expected_request_digest",
        "expected_baseline_run",
        "expected_subject_run",
        "expected_signer",
        "policy_path",
    ],
)
def test_rejection_authenticates_expected_a_not_examined_b(handoff, anchor):
    pack, kwargs, args, _, _ = handoff
    if anchor == "policy_path":
        changed_policy = copy.deepcopy(args["policy"])
        changed_policy["metrics"][0]["maximum_regression"] = 0.123
        kwargs[anchor].write_bytes(canonical_json_bytes(changed_policy))
    else:
        kwargs[anchor] = "sha256:" + "0" * 64
    with pytest.raises(verification.CapturedVerificationError):
        verification.verify_captured_evidence(pack, **kwargs)
    # Receipt authentication must not load even one rejected payload.
    (pack / "request.json").unlink()
    authenticated = _read_receipt(handoff)
    assert authenticated.ok
    assert authenticated.statement["verdict"] == {
        "ok": False,
        "integrity_ok": False,
        "decision": None,
        "policy_verdict": None,
        "verification_status": 6,
    }
    assert authenticated.statement["scoring_assurance"] is None
    assert not _read_receipt(handoff, expected_verifier_identity="unauthorized").ok
    assert not _read_receipt(handoff, expected_profile_digest="sha256:" + "1" * 64).ok


def test_completed_mismatch_is_authentic_rejection_with_null_assurance(handoff):
    pack, kwargs, _, key, _ = handoff
    report_path = pack / "reports/evaluation.report.json"
    report = json.loads(report_path.read_bytes())
    report["decision"] = "regression"
    _replace(report_path, canonical_json_bytes(report))
    _rebind(pack, key)
    with pytest.raises(verification.CapturedVerificationError, match="disagrees"):
        verification.verify_captured_evidence(pack, **kwargs)
    authenticated = _read_receipt(handoff)
    assert authenticated.ok
    assert authenticated.statement["replay_status"] == "completed"
    assert authenticated.statement["scoring_assurance"] is None
    assert authenticated.statement["verdict"]["decision"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(extra=True),
        lambda s: s.update(verification_scope="model_execution"),
        lambda s: s.update(replay_status="not_started"),
        lambda s: s.update(scoring_assurance=[]),
        lambda s: s["scoring_assurance"].pop(),
        lambda s: s["scoring_assurance"][0].update(scoring_assurance="recorded"),
        lambda s: s["verdict"].update(ok=False),
        lambda s: s["verdict"].update(integrity_ok=False),
        lambda s: s["verdict"].update(verification_status=7),
        lambda s: s["verdict"].update(policy_verdict="fail"),
        lambda s: s["subject"].update(run_digest="sha256:" + "0" * 64),
    ],
)
def test_authorized_resigning_cannot_certify_contradictory_statement(handoff, mutate):
    verification.verify_captured_evidence(handoff[0], **handoff[1])
    _resign_receipt(handoff, mutate)
    assert not _read_receipt(handoff).ok


@pytest.mark.parametrize("decision", ["regression", "insufficient_evidence"])
def test_completed_policy_rejection_retains_assurance(handoff, decision):
    pack, kwargs, args, _, _ = handoff
    changed = copy.deepcopy(args)
    if decision == "regression":
        changed["policy"]["metrics"][0]["subject_minimum"] = 1
        for row in changed["subject"]["records"]:
            row["output"] = "wrong"
    else:
        changed["policy"]["metrics"][0]["minimum_count"] = 50000
    changed["comparison"] = compare_runs(
        changed["baseline"], changed["subject"], changed["policy"]
    )
    changed["normalized_request"]["comparison"]["policy_digest"] = digest(
        changed["policy"]
    )
    changed["normalized_request"]["comparison"]["subject"]["run_digest"] = digest(
        changed["subject"]
    )
    changed["request_digest"] = digest(changed["normalized_request"])
    kwargs.update(
        expected_subject_run=digest(changed["subject"]),
        expected_request_digest=changed["request_digest"],
    )
    kwargs["policy_path"].write_bytes(canonical_json_bytes(changed["policy"]))
    new_pack = pack.parent / "adverse"
    publish_captured_evidence(new_pack, **changed)
    result = verification.verify_captured_evidence(new_pack, **kwargs)
    assert result["ok"] is False and result["decision"] == decision
    authenticated = _read_receipt((new_pack, *handoff[1:]))
    assert authenticated.ok
    assert authenticated.statement["verdict"]["verification_status"] == 7
    assert authenticated.statement["scoring_assurance"]


@pytest.mark.parametrize(
    "mutation", ["append", "duplicate", "reverse", "missing", "wrong_digest"]
)
def test_exact_ledger_binding_is_required(handoff, mutation):
    pack, kwargs, _, _, _ = handoff
    ledger = pack / "checksums.sha256"
    raw = ledger.read_bytes()
    lines = raw.splitlines(keepends=True)
    changed = {
        "append": raw + b"\n",
        "duplicate": raw + lines[0],
        "reverse": b"".join(reversed(lines)),
        "missing": b"".join(lines[1:]),
        "wrong_digest": b"0" * 64 + raw[64:],
    }[mutation]
    _replace(ledger, changed)
    with pytest.raises(verification.CapturedVerificationError, match="ledger"):
        verification.verify_captured_evidence(pack, **kwargs)
    assert _read_receipt(handoff).ok
    with pytest.raises(EvidenceReportError, match="ledger"):
        render_evidence(pack)


def test_tampered_native_shaped_signature_rejects_in_verify_and_report(handoff):
    pack, kwargs, _, _, _ = handoff
    path = pack / "manifest.signature.json"
    signature = json.loads(path.read_bytes())
    value = bytearray(base64.b64decode(signature["signature"]["value"]))
    value[0] ^= 1
    signature["signature"]["value"] = base64.b64encode(value).decode("ascii")
    _replace(path, canonical_json_bytes(signature))
    with pytest.raises(
        verification.CapturedVerificationError, match="signature is invalid"
    ):
        verification.verify_captured_evidence(pack, **kwargs)
    assert _read_receipt(handoff).ok
    with pytest.raises(EvidenceReportError, match="signature is invalid"):
        render_evidence(pack)


@pytest.mark.parametrize("length", [64 * 1024 + 1, 256 * 1024, 256 * 1024 + 1])
def test_manifest_detector_and_control_caps_precede_all_payload_reads(
    handoff, monkeypatch, length
):
    pack = handoff[0]
    path = pack / "manifest.json"
    raw = path.read_bytes()
    _replace(path, raw + b" " * (length - len(raw)))
    read_at = Mock(wraps=contract._read_at)
    monkeypatch.setattr(contract, "_read_at", read_at)
    with pytest.raises(contract.CapturedContractError):
        with contract.captured_snapshot(pack):
            pytest.fail("oversized manifest accepted")
    assert [call.args[1] for call in read_at.call_args_list] == ["manifest.json"]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"format":"invarlock/evidence-pack-v2","format":"invarlock/evidence-pack-v1","kind":"captured"}',
        b'{"format":"invarlock/evidence-pack-v2","kind":"native"}',
        b'{"format":"unknown","kind":"captured"}',
    ],
)
def test_discriminator_rejection_never_falls_back(handoff, raw):
    _replace(handoff[0] / "manifest.json", raw)
    with pytest.raises(CapturedReportError):
        is_captured_manifest(handoff[0])


def test_native_detector_keeps_256k_ceiling(tmp_path):
    manifest = tmp_path / "manifest.json"
    raw = b'{"format":"invarlock/evidence-pack-v1"}'
    manifest.write_bytes(raw + b" " * (100 * 1024 - len(raw)))
    assert is_captured_manifest(tmp_path) is False


@pytest.mark.parametrize("entry", ["unknown.json", "empty", "tree", "symlink", "fifo"])
def test_fixed_inventory_rejects_unsafe_or_unexpected_entries(
    handoff, monkeypatch, entry
):
    pack = handoff[0]
    if entry == "unknown.json":
        (pack / entry).write_bytes(b"{}")
    elif entry in {"empty", "tree"}:
        (pack / entry).mkdir()
        if entry == "tree":
            for i in range(1000):
                (pack / entry / str(i)).touch()
    elif entry == "symlink":
        (pack / "records/baseline.json").unlink()
        (pack / "records/baseline.json").symlink_to(pack.parent / "policy.json")
    else:
        (pack / "records/baseline.json").unlink()
        os.mkfifo(pack / "records/baseline.json")
    monkeypatch.setattr(
        Path, "rglob", Mock(side_effect=AssertionError("unbounded scan"))
    )
    read_at = Mock(wraps=contract._read_at)
    monkeypatch.setattr(contract, "_read_at", read_at)
    with pytest.raises(contract.CapturedContractError):
        with contract.captured_snapshot(pack):
            pytest.fail("invalid inventory accepted")
    assert {call.args[1] for call in read_at.call_args_list} == {"manifest.json"}


def test_late_inventory_flood_suppresses_positive_receipt(handoff, monkeypatch):
    pack, kwargs, _, _, _ = handoff
    original = verification.compare_runs

    def flood(*args, **kwargs):
        result = original(*args, **kwargs)
        (pack / "late").mkdir()
        for i in range(1000):
            (pack / "late" / str(i)).touch()
        return result

    monkeypatch.setattr(verification, "compare_runs", flood)
    monkeypatch.setattr(
        Path, "rglob", Mock(side_effect=AssertionError("unbounded scan"))
    )
    with pytest.raises(
        verification.CapturedVerificationError, match="inventory changed"
    ):
        verification.verify_captured_evidence(pack, **kwargs)
    authenticated = _read_receipt(handoff)
    assert authenticated.ok
    assert authenticated.statement["scoring_assurance"] is None
    assert authenticated.statement["verdict"]["ok"] is False


def test_manifest_replacement_after_detector_has_no_receipt(handoff, monkeypatch):
    pack, kwargs, _, _, _ = handoff
    original = contract._inventory

    def replace(root, signed):
        path = pack / "manifest.json"
        replacement = pack.parent / "replacement.json"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        return original(root, signed)

    monkeypatch.setattr(contract, "_inventory", replace)
    with pytest.raises(
        verification.CapturedVerificationError, match="changed after detection"
    ):
        verification.verify_captured_evidence(pack, **kwargs)
    assert not kwargs["receipt_path"].exists()


@pytest.mark.parametrize(
    "error,reason",
    [
        (OSError("disk failed"), "verification_io_error"),
        (RuntimeError("bug"), "verification_internal_error"),
        (ValueError("unexpected bug"), "verification_internal_error"),
    ],
)
def test_replay_operational_errors_never_issue_rejection_receipts(
    handoff, monkeypatch, error, reason
):
    monkeypatch.setattr(verification, "compare_runs", Mock(side_effect=error))
    with pytest.raises(verification.CapturedVerificationIncomplete, match=reason):
        verification.verify_captured_evidence(handoff[0], **handoff[1])
    assert not handoff[1]["receipt_path"].exists()


def test_deterministic_replay_failure_issues_failed_rejection(handoff, monkeypatch):
    monkeypatch.setattr(
        verification,
        "compare_runs",
        Mock(side_effect=EvaluationRecordsError("reference is invalid")),
    )
    with pytest.raises(verification.CapturedVerificationError, match="replay failed"):
        verification.verify_captured_evidence(handoff[0], **handoff[1])
    authenticated = _read_receipt(handoff)
    assert authenticated.ok
    assert authenticated.statement["replay_status"] == "failed"
    assert authenticated.statement["scoring_assurance"] is None


@pytest.mark.parametrize("budget", [0, -1, True, "100", 1.5])
def test_work_refusal_precedes_scoring_and_never_issues_receipt(
    handoff, monkeypatch, budget
):
    forbidden = Mock(side_effect=AssertionError("scoring on refused work"))
    monkeypatch.setattr(verification, "compare_runs", forbidden)
    with pytest.raises(verification.CapturedVerificationIncomplete):
        verification.verify_captured_evidence(
            handoff[0], **{**handoff[1], "max_bootstrap_draws": budget}
        )
    forbidden.assert_not_called()
    assert not handoff[1]["receipt_path"].exists()


def test_snapshot_bytes_are_immutable_and_detached(handoff):
    with contract.captured_snapshot(handoff[0]) as snapshot:
        with pytest.raises(TypeError):
            snapshot.files["manifest.json"] = b"changed"
        parsed = contract.json_object(snapshot.manifest_bytes, "manifest")
        parsed.clear()
        assert (
            contract.json_object(snapshot.manifest_bytes, "manifest")["kind"]
            == "captured"
        )


def test_exact_aggregate_bound_includes_all_controls_and_request(handoff):
    pack = handoff[0]
    for relative in ("records/baseline.json", "records/subject.json"):
        path = pack / relative
        path.chmod(0o600)
        with path.open("r+b") as handle:
            handle.truncate(contract.PAYLOAD_LIMIT)
    other_bytes = sum(
        (pack / name).stat().st_size
        for name in (
            "manifest.json",
            "manifest.signature.json",
            "checksums.sha256",
            "request.json",
            "inputs/policy.json",
            "records/baseline.json",
            "records/subject.json",
        )
    )
    report = pack / "reports/evaluation.report.json"
    report.chmod(0o600)
    with report.open("r+b") as handle:
        handle.truncate(contract.TOTAL_LIMIT - other_bytes)
    with contract.secure_directory(pack) as root:
        assert len(contract._inventory(root, True)) == 11
        with report.open("r+b") as handle:
            handle.truncate(contract.TOTAL_LIMIT - other_bytes + 1)
        with pytest.raises(contract.CapturedContractError, match="total byte limit"):
            contract._inventory(root, True)


def test_request_expansion_cannot_publish(handoff):
    pack, _, args, _, _ = handoff
    changed = copy.deepcopy(args)
    changed["normalized_request"]["padding"] = "x" * contract.REQUEST_LIMIT
    changed["request_digest"] = digest(changed["normalized_request"])
    destination = pack.parent / "unpublished" / "evidence"
    with pytest.raises(CapturedEvidenceError, match="byte limit"):
        publish_captured_evidence(destination, **changed)
    assert not destination.parent.exists()


@pytest.mark.parametrize("target", ["inside", "symlink", "exists"])
def test_receipt_destination_rejects_unsafe_or_existing_target(handoff, target):
    pack, kwargs, _, _, _ = handoff
    if target == "inside":
        kwargs["receipt_path"] = pack / "receipt.json"
    elif target == "symlink":
        alias = pack.parent / "alias"
        alias.symlink_to(pack, target_is_directory=True)
        kwargs["receipt_path"] = alias / "receipt.json"
    else:
        kwargs["receipt_path"].write_bytes(b"keep")
    with pytest.raises(verification.CapturedVerificationIncomplete):
        verification.verify_captured_evidence(pack, **kwargs)
    assert not (pack / "receipt.json").exists()
    if target == "exists":
        assert kwargs["receipt_path"].read_bytes() == b"keep"


def test_atomic_receipt_failure_leaves_no_partial_or_positive_file(
    handoff, monkeypatch
):
    original = os.fsync

    def fail_directory(fd):
        import stat

        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        original(fd)

    monkeypatch.setattr(contract.os, "fsync", fail_directory)
    with pytest.raises(
        verification.CapturedVerificationIncomplete, match="publication_failed"
    ):
        verification.verify_captured_evidence(handoff[0], **handoff[1])
    assert not handoff[1]["receipt_path"].exists()
    assert not list(handoff[0].parent.glob(".captured-*"))


def test_captured_payload_above_native_64m_limit_round_trips(handoff):
    pack, kwargs, args, _, _ = handoff
    changed = copy.deepcopy(args)
    changed["baseline"]["records"][0]["context"] = "x" * (64 * 1024 * 1024 + 1)
    changed["normalized_request"]["comparison"]["baseline"]["run_digest"] = digest(
        changed["baseline"]
    )
    changed["request_digest"] = digest(changed["normalized_request"])
    changed["comparison"] = compare_runs(
        changed["baseline"], changed["subject"], changed["policy"]
    )
    destination = pack.parent / "large"
    publish_captured_evidence(destination, **changed)
    assert (destination / "records/baseline.json").stat().st_size > 64 * 1024 * 1024
    kwargs.update(
        expected_baseline_run=digest(changed["baseline"]),
        expected_request_digest=changed["request_digest"],
    )
    assert verification.verify_captured_evidence(destination, **kwargs)["ok"]
    rendered = render_evidence(destination)
    assert isinstance(rendered, EvidenceReportV2)
    assert "Not performed by report" in rendered.text


@pytest.mark.parametrize(
    "relative,limit",
    [
        ("request.json", 1024 * 1024),
        ("manifest.signature.json", 64 * 1024),
        ("checksums.sha256", 64 * 1024),
        ("records/baseline.json", 128 * 1024 * 1024),
        ("records/subject.json", 128 * 1024 * 1024),
        ("inputs/policy.json", 128 * 1024 * 1024),
        ("reports/evaluation.report.json", 128 * 1024 * 1024),
    ],
)
def test_physical_per_file_limits_are_inclusive_and_precede_copy(
    handoff, monkeypatch, relative, limit
):
    pack = handoff[0]
    path = pack / relative
    path.chmod(0o600)
    with path.open("r+b") as handle:
        handle.truncate(limit)
    with contract.secure_directory(pack) as root:
        assert len(contract._inventory(root, True)) == 11
    with path.open("r+b") as handle:
        handle.truncate(limit + 1)
    read_at = Mock(wraps=contract._read_at)
    monkeypatch.setattr(contract, "_read_at", read_at)
    with pytest.raises(contract.CapturedContractError, match="byte limit"):
        with contract.captured_snapshot(pack):
            pytest.fail("oversized file accepted")
    assert {call.args[1] for call in read_at.call_args_list} == {"manifest.json"}


def test_stable_schema_rejection_has_status_four_receipt(handoff):
    pack, kwargs, _, _, _ = handoff
    path = pack / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["extra"] = True
    _replace(path, canonical_json_bytes(manifest))
    with pytest.raises(verification.CapturedVerificationError) as error:
        verification.verify_captured_evidence(pack, **kwargs)
    assert error.value.exit_code == 4
    authenticated = _read_receipt(handoff)
    assert authenticated.ok
    assert authenticated.statement["verdict"]["verification_status"] == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("algorithm", "other"),
        ("format", "other"),
        ("extra", True),
        ("public_key", {"encoding": "other", "value": "anything"}),
    ],
)
def test_receipt_signature_container_is_closed(handoff, field, value):
    verification.verify_captured_evidence(handoff[0], **handoff[1])
    path = handoff[1]["receipt_path"]
    receipt = json.loads(path.read_bytes())
    receipt["signature"][field] = value
    path.write_bytes(canonical_json_bytes(receipt))
    assert not _read_receipt(handoff).ok


def test_signed_rejection_cannot_claim_positive_ok_or_assurance(handoff):
    pack, kwargs, _, _, _ = handoff
    kwargs["expected_request_digest"] = "sha256:" + "0" * 64
    with pytest.raises(verification.CapturedVerificationError):
        verification.verify_captured_evidence(pack, **kwargs)
    _resign_receipt(handoff, lambda s: s["verdict"].update(ok=True))
    assert not _read_receipt(handoff).ok


def test_unsigned_manifest_forbids_signature_and_never_passes_verification(handoff):
    pack, kwargs, args, _, _ = handoff
    destination = pack.parent / "unsigned"
    publish_captured_evidence(
        destination, **{**args, "unsigned": True, "signing_key_path": None}
    )
    manifest = json.loads((destination / "manifest.json").read_bytes())
    assert manifest["signing_key_fingerprint"] is None
    assert "signature" not in manifest
    with pytest.raises(verification.CapturedVerificationError, match="requires signed"):
        verification.verify_captured_evidence(destination, **kwargs)
    assert _read_receipt((destination, *handoff[1:])).ok
    (destination / "manifest.signature.json").write_bytes(b"{}\n")
    with pytest.raises(EvidenceReportError, match="inventory"):
        render_evidence(destination)


def test_unsupported_unicode_environment_is_incomplete_not_corruption(
    handoff, monkeypatch
):
    pack, kwargs, _, _, _ = handoff
    monkeypatch.setattr(verification.unicodedata, "unidata_version", "0.0.0")
    scorer = Mock(side_effect=AssertionError("unsupported environment scored"))
    monkeypatch.setattr(verification, "compare_runs", scorer)
    with pytest.raises(
        verification.CapturedVerificationIncomplete,
        match="unsupported_scoring_environment",
    ):
        verification.verify_captured_evidence(pack, **kwargs)
    scorer.assert_not_called()
    assert not kwargs["receipt_path"].exists()
    rendered = render_evidence(pack)
    assert isinstance(rendered, EvidenceReportV2)
    assert "Not performed by report" in rendered.text


def test_read_io_failure_has_no_integrity_finding(handoff, monkeypatch):
    original = contract._read_at

    def fail(parent, name, limit):
        if name == "baseline.json":
            raise OSError("read failed")
        return original(parent, name, limit)

    monkeypatch.setattr(contract, "_read_at", fail)
    with pytest.raises(
        verification.CapturedVerificationIncomplete, match="verification_io_error"
    ):
        verification.verify_captured_evidence(handoff[0], **handoff[1])
    assert not handoff[1]["receipt_path"].exists()


def test_source_file_replacement_during_read_is_detected(handoff, monkeypatch):
    pack = handoff[0]
    path = pack / "records/baseline.json"
    identity = (path.stat().st_dev, path.stat().st_ino)
    original = os.fstat
    swapped = False

    def replace(fd):
        nonlocal swapped
        value = original(fd)
        if not swapped and (value.st_dev, value.st_ino) == identity:
            replacement = pack.parent / "new-baseline.json"
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            swapped = True
        return value

    monkeypatch.setattr(contract.os, "fstat", replace)
    with pytest.raises(
        verification.CapturedVerificationError, match="changed while reading"
    ):
        verification.verify_captured_evidence(pack, **handoff[1])
    assert swapped
    assert _read_receipt(handoff).ok


def test_publication_and_readers_reject_symlinked_ancestors(handoff):
    pack, kwargs, args, _, _ = handoff
    alias = pack.parent / "alias"
    alias.symlink_to(pack.parent, target_is_directory=True)
    with pytest.raises(CapturedEvidenceError, match="non-symlink"):
        publish_captured_evidence(alias / "new-evidence", **args)
    with pytest.raises(verification.CapturedVerificationError):
        verification.verify_captured_evidence(alias / "evidence", **kwargs)
    assert not kwargs["receipt_path"].exists()


def test_atomic_receipt_parent_replacement_does_not_risk_foreign_output(
    handoff, monkeypatch
):
    pack, kwargs, _, _, _ = handoff
    output = pack.parent / "output"
    output.mkdir()
    kwargs["receipt_path"] = output / "receipt.json"
    original = os.link
    renamed = pack.parent / "moved-output"

    def replace(*args, **kwargs):
        original(*args, **kwargs)
        output.rename(renamed)
        output.mkdir()

    monkeypatch.setattr(contract.os, "link", replace)
    with pytest.raises(
        verification.CapturedVerificationIncomplete, match="publication_failed"
    ):
        verification.verify_captured_evidence(pack, **kwargs)
    assert (renamed / "receipt.json").is_file()
    assert not kwargs["receipt_path"].exists()
    assert not list(renamed.glob(".captured-*"))


def test_resigned_ledger_digest_cannot_authorize_duplicate_inventory(handoff):
    pack, kwargs, _, key, _ = handoff
    ledger = (pack / "checksums.sha256").read_bytes()
    ledger += ledger.splitlines(keepends=True)[0]
    manifest = json.loads((pack / "manifest.json").read_bytes())
    manifest["checksums_sha256_digest"] = hashlib.sha256(ledger).hexdigest()
    manifest_raw = canonical_json_bytes(manifest)
    _replace(pack / "checksums.sha256", ledger)
    _replace(pack / "manifest.json", manifest_raw)
    _replace(
        pack / "manifest.signature.json",
        canonical_json_bytes(contract.manifest_signature(manifest_raw, key)),
    )
    with pytest.raises(
        verification.CapturedVerificationError, match="ledger inventory"
    ):
        verification.verify_captured_evidence(pack, **kwargs)
    assert _read_receipt(handoff).ok


def test_positive_receipt_must_match_manifest_reference_expectations(handoff):
    pack, kwargs, _, _, _ = handoff
    verification.verify_captured_evidence(pack, **kwargs)
    manifest = json.loads((pack / "manifest.json").read_bytes())
    manifest["files"]["request"]["digest"] = "sha256:" + "0" * 64
    manifest["request_digest"] = manifest["files"]["request"]["digest"]
    raw = canonical_json_bytes(manifest)
    _replace(pack / "manifest.json", raw)
    _resign_receipt(handoff, lambda s: s.update(pack_manifest_digest=contract.sha(raw)))
    assert not _read_receipt(handoff).ok
