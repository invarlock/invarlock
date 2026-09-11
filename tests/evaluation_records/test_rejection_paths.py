"""Untrusted exports and inconsistent policies must never yield a passing gate."""

import copy
import hashlib
import json
import os
from decimal import localcontext

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock.cli import app
from invarlock.core.builtin_scorers import BuiltinScorer
from invarlock.core.scoring import UNICODE_VERSION, MetricError, score
from invarlock.engine import (
    EvaluationRecordsError,
    EvaluationTransactionError,
    EvidenceVerificationError,
    evaluate_request_file,
    physical_file_digest,
    run_digest,
    verify_evidence,
    write_run,
)
from tests._evaluation_support import (
    build_pack,
    captured_request_digest,
    compare_runs,
    digest,
    example_project,
    load_run,
    make_run,
    materialize_captured_request,
    read_json,
    write_directory,
    write_new,
    write_snapshot,
)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_metric",
        "duplicate_slice",
        "reserved_slice",
        "no_key",
        "configuration",
        "metric_unit",
        "no_rubric",
        "recomputed_provenance",
        "direction",
        "scorer_config",
        "absolute_bounds",
    ],
)
def test_policy_contradictions_are_integration_errors(change):
    base, candidate, policy = example_project("judge")
    metric = policy["metrics"][0]
    if change == "duplicate_metric":
        policy["metrics"].append(copy.deepcopy(metric))
    elif change == "duplicate_slice":
        policy["slices"].append(copy.deepcopy(policy["slices"][0]))
    elif change == "reserved_slice":
        policy["slices"][0]["name"] = "overall"
    elif change == "no_key":
        del metric["score_key"]
    elif change == "configuration":
        metric["configuration"] = {"casefold": True}
    elif change == "metric_unit":
        metric["unit"] = "percent"
    elif change == "no_rubric":
        metric["accepted_provenance"]["rubric_digest"] = None
    elif change == "recomputed_provenance":
        metric["kind"] = "normalized_match"
        metric["configuration"] = {"unicode_version": UNICODE_VERSION}
    elif change in ("direction", "scorer_config"):
        metric.update(kind="normalized_match")
        metric["configuration"] = {"unicode_version": UNICODE_VERSION}
        del metric["accepted_provenance"], metric["score_key"]
        if change == "direction":
            metric["direction"] = "lower"
        else:
            metric["configuration"] = {"absolute": 0.1}
    else:
        metric.update(subject_minimum=1, subject_maximum=0)
    with pytest.raises(EvaluationRecordsError):
        compare_runs(base, candidate, policy)


@pytest.mark.parametrize(
    "kind,expected,output,configuration",
    [
        ("unknown", "x", "x", {}),
        ("json_fields", {}, {}, {}),
        ("json_fields", {}, {}, {"fields": ["/a", "/a"]}),
        ("numeric_tolerance", 1, 1, {"absolute": True}),
        ("normalized_match", "x", "x", {"absolute": 0}),
    ],
)
def test_scorer_configurations_do_not_guess(kind, expected, output, configuration):
    with pytest.raises(MetricError):
        score(kind, expected, output, configuration)


def test_structured_array_pointers_and_bad_answers():
    assert (
        score("json_fields", {"a": [1, 2]}, {"a": [1, 3]}, {"fields": ["/a/0", "/a/1"]})
        == 0.5
    )
    assert score("json_fields", {"a": [1]}, {"a": 3}, {"fields": ["/a/0"]}) == 0
    assert (
        score("normalized_match", "answer", {}, {"unicode_version": UNICODE_VERSION})
        == 0
    )
    with pytest.raises(ValueError):
        BuiltinScorer("unknown")


def test_numeric_tolerance_does_not_round_distinct_large_integers_together():
    reference = 9007199254740992
    assert score("numeric_tolerance", reference, reference + 1, {}) == 0
    assert (
        score("numeric_tolerance", reference, str(reference + 1), {"absolute": 1}) == 1
    )
    assert (
        score(
            "numeric_tolerance",
            "0.123456789012345678901",
            "0.123456789012345678902",
            {},
        )
        == 0
    )
    assert score("numeric_tolerance", 1, "1e999999999", {}) == 0
    assert score("numeric_tolerance", 1, "not a number", {}) == 0
    with localcontext() as context:
        context.prec = 2
        context.Emax = 5
        assert (
            score("numeric_tolerance", reference, str(reference + 1), {"absolute": 1})
            == 1
        )


@pytest.mark.parametrize(
    "native,adapter",
    [
        ([], "jsonl"),
        ({"version": True}, "inspect-json"),
        ({"version": 2, "status": "unknown"}, "inspect-json"),
        ({"version": 2, "status": "cancelled"}, "inspect-json"),
        ({"version": 2, "status": "success", "samples": []}, "inspect-json"),
        ([{"doc_id": 0}], "lm-eval-samples"),
        ([{"testIdx": 0}], "promptfoo-jsonl"),
    ],
)
def test_unsupported_native_shapes_fail(tmp_path, native, adapter):
    path = tmp_path / "export"
    path.write_text(
        json.dumps(native)
        if isinstance(native, dict)
        else "\n".join(json.dumps(r) for r in native)
    )
    with pytest.raises(EvaluationRecordsError):
        load_run(path, adapter=adapter)


@pytest.mark.parametrize(
    "change", ["choices", "nontext", "targets", "epoch", "id", "score", "metadata"]
)
def test_ambiguous_inspect_records_fail(tmp_path, change):
    row = {
        "id": 1,
        "input": "q",
        "target": "yes",
        "output": {"choices": [{"message": {"content": "yes"}}]},
    }
    if change == "choices":
        row["output"]["choices"] = []
    elif change == "nontext":
        row["output"]["choices"][0]["message"]["content"] = [{"type": "image"}]
    elif change == "targets":
        row["target"] = ["yes", "no"]
    elif change == "epoch":
        row["epoch"] = 2
    elif change == "id":
        row["id"] = None
    elif change == "score":
        row["scores"] = {"judgment": {"value": "maybe"}}
    else:
        row["metadata"] = "not tags"
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"version": 2, "status": "success", "samples": [row]}))
    with pytest.raises(EvaluationRecordsError):
        load_run(path, adapter="inspect-json")


def test_inspect_text_parts_single_target_and_error_preservation(tmp_path):
    row = {
        "id": 1,
        "input": "q",
        "target": ["yes"],
        "output": {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "y"},
                            {"type": "text", "text": "es"},
                        ]
                    }
                }
            ]
        },
    }
    path = tmp_path / "export.json"
    options = {
        "adapter": "inspect-json",
        "source": {"name": "inspect", "version": "0.3.254"},
        "run_id": "one",
        "artifact_digest": "sha256:" + "a" * 64,
    }
    path.write_text(json.dumps({"version": 2, "status": "success", "samples": [row]}))
    run = load_run(path, **options)
    assert run["records"][0]["output"] == "yes"
    row["error"] = {"message": "failed"}
    row["output"]["choices"] = []
    path.write_text(json.dumps({"version": 2, "status": "success", "samples": [row]}))
    assert load_run(path, **options)["records"][0]["error"] == "upstream_error"


def test_invalid_file_and_identity_operations(tmp_path):
    base, candidate, policy = example_project("classification")
    with pytest.raises(EvaluationRecordsError):
        make_run(
            [{"unknown": 1}],
            source=base["source"],
            run_id="r",
            artifact_digest=base["artifact_digest"],
        )
    with pytest.raises(EvaluationRecordsError):
        load_run(tmp_path / "absent", adapter="guess")
    with pytest.raises(EvaluationRecordsError):
        read_json(tmp_path / "absent")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(base))
    with pytest.raises(EvaluationRecordsError):
        load_run(path, run_id="override")
    with pytest.raises(EvaluationRecordsError):
        write_new(path, b"changed")
    assert json.loads(path.read_text()) == base
    with pytest.raises(EvaluationRecordsError):
        write_directory(tmp_path / "result", {"../escape": b"bad"})
    request = materialize_captured_request(
        tmp_path / "request", base, candidate, policy
    )
    invalid_key = tmp_path / "invalid.pem"
    invalid_key.write_bytes(b"not a signing key")
    with pytest.raises(EvaluationTransactionError, match="signing key"):
        evaluate_request_file(request, signing_key_path=invalid_key)
    pack = tmp_path / "pack"
    write_snapshot(pack, build_pack(base, candidate, policy))
    with pytest.raises(EvidenceVerificationError, match="verifier key"):
        verify_evidence(
            pack,
            policy_path=request.parent / "policy.json",
            expected_signer="sha256:" + "a" * 64,
            expected_baseline_run=digest(base),
            expected_subject_run=digest(candidate),
            expected_request_digest=captured_request_digest(base, candidate, policy),
            receipt_path=tmp_path / "receipt.json",
            verifier_signing_key_bytes=object(),
            verifier_identity="rejection-test",
        )


def test_cli_evaluate_verify_report_reject_invalid_inputs(tmp_path):
    runner = CliRunner()
    invalid_case_set = tmp_path / "invalid-case-set.json"
    invalid_case_set.write_text("{}")
    assert (
        runner.invoke(
            app,
            ["evaluate", "--freeze-cases", str(invalid_case_set)],
        ).exit_code
        == 2
    )
    assert (
        runner.invoke(
            app,
            ["evaluate", "--init", str(tmp_path / "bad"), "--example", "unknown"],
        ).exit_code
        == 2
    )
    project = tmp_path / "project"
    assert runner.invoke(app, ["evaluate", "--init", str(project)]).exit_code == 0
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = tmp_path / "rsa.pem"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public = tmp_path / "rsa.pub"
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    assert (
        runner.invoke(
            app,
            [
                "evaluate",
                str(project / "request.yaml"),
                "--signing-key",
                str(private),
                "--json",
            ],
        ).exit_code
        == 2
    )
    assert (
        runner.invoke(
            app,
            [
                "verify",
                str(project),
                "--json",
            ],
        ).exit_code
        == 2
    )
    assert runner.invoke(app, ["report", str(project), "--json"]).exit_code == 2


def test_cli_verify_rejects_symlinked_evidence_directory(tmp_path):
    target = tmp_path / "evidence"
    target.mkdir()
    (target / "manifest.json").write_text("{}")
    link = tmp_path / "evidence-link"
    link.symlink_to(target, target_is_directory=True)
    result = CliRunner().invoke(app, ["verify", str(link), "--json"])
    assert result.exit_code == 2


def test_sdk_import_and_distinct_file_run_digests(tmp_path):
    from invarlock.engine import load_run as sdk_load_run

    export = tmp_path / "records.jsonl"
    export.write_text('{"id":"1","input":"q","expected":"yes","output":"yes"}\n')
    run = sdk_load_run(
        export,
        adapter="jsonl",
        source={"name": "jsonl", "version": "1"},
        run_id="one",
        artifact_digest="sha256:" + "a" * 64,
    )
    target = tmp_path / "run.json"
    write_run(target, run)
    original = target.read_bytes()
    with pytest.raises(EvaluationRecordsError):
        write_run(target, run)
    assert target.read_bytes() == original
    assert run_digest(sdk_load_run(target)) == digest(run)
    assert (
        physical_file_digest(target) == "sha256:" + hashlib.sha256(original).hexdigest()
    )

    pretty = tmp_path / "pretty-run.json"
    pretty.write_text(json.dumps(run, indent=2))
    assert run_digest(sdk_load_run(pretty)) == digest(run)
    assert physical_file_digest(pretty) != digest(run)

    link = tmp_path / "run-link.json"
    link.symlink_to(target)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for unsafe in (link, fifo):
        with pytest.raises(EvaluationRecordsError):
            physical_file_digest(unsafe)
        with pytest.raises(EvaluationRecordsError):
            sdk_load_run(unsafe)


@pytest.mark.parametrize("replace", [False, True])
def test_physical_digest_rejects_artifact_changed_while_reading(
    tmp_path, monkeypatch, replace
):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"original artifact")
    inode = artifact.stat().st_ino
    original_fstat = os.fstat
    changed = False

    def mutate_after_initial_stat(fd):
        nonlocal changed
        snapshot = original_fstat(fd)
        if snapshot.st_ino == inode and not changed:
            changed = True
            if replace:
                replacement = tmp_path / "replacement.bin"
                replacement.write_bytes(b"different artifact bytes")
                replacement.replace(artifact)
            else:
                artifact.write_bytes(b"different artifact bytes")
        return snapshot

    monkeypatch.setattr(os, "fstat", mutate_after_initial_stat)
    with pytest.raises(EvaluationRecordsError, match="artifact changed during hashing"):
        physical_file_digest(artifact)
    assert changed


def test_sdk_verification_rejects_rsa_key(tmp_path):
    base, subject, policy = example_project("classification")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    request = materialize_captured_request(tmp_path / "request", base, subject, policy)
    pack = tmp_path / "pack"
    signer = Ed25519PrivateKey.generate()
    write_snapshot(pack, build_pack(base, subject, policy, signer))
    with pytest.raises(EvidenceVerificationError, match="verifier key must be Ed25519"):
        verify_evidence(
            pack,
            policy_path=request.parent / "policy.json",
            expected_signer="sha256:" + "a" * 64,
            expected_baseline_run=digest(base),
            expected_subject_run=digest(subject),
            expected_request_digest=captured_request_digest(base, subject, policy),
            receipt_path=tmp_path / "receipt.json",
            verifier_signing_key_bytes=key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
            verifier_identity="rejection-test",
        )
