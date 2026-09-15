"""Reconstruct and verify retained HTTP observations without contacting a service."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import socket
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.engine import case_set_digest, freeze_case_set
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_receipt import verify_signed_verification_receipt
from tests.cli.test_import_journey import _key

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/hosted-service/references/mistral-7b-http"
MODELS = {
    "baseline": "mistralai/Mistral-7B-v0.1",
    "subject": "mistralai/Mistral-7B-Instruct-v0.1",
}
FILES = {
    "README.md",
    "capture/protocol.json",
    "capture/baseline.json",
    "capture/subject.json",
    "capture/collector-source.txt",
    "capture/journey-source.txt",
    "capture/service-source.txt",
    "capture/checkpoint-loader-source.txt",
    "capture/checkpoint-loading-protocol.json",
    "capture/runtime-environment.json",
    "earlier-attempt/protocol.json",
    "earlier-attempt/baseline.json",
    "earlier-attempt/collector-source.txt",
    "earlier-attempt/journey-source.txt",
    "evidence/checksums.sha256",
    "evidence/inputs/policy.json",
    "evidence/manifest.json",
    "evidence/manifest.signature.json",
    "evidence/records/baseline.json",
    "evidence/records/subject.json",
    "evidence/reports/evaluation.report.json",
    "evidence/request.json",
    "signer.public.pem",
    "verifier.public.pem",
    "verification.receipt.json",
    "report.html",
    "report.md",
    "report.xml",
    "current/report.html",
    "current/report.md",
    "current/report.xml",
    "current/report.json",
    "current/verification.json",
    "current/verification.receipt.json",
}


def _read(path):
    return json.loads(path.read_bytes())


def _digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject_connection(*args, **kwargs):
        pytest.fail("retained reference verification must not contact a service")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)


@pytest.fixture
def reference():
    return _read(REFERENCE / "reference.json")


@pytest.fixture
def protocol():
    return _read(REFERENCE / "capture/protocol.json")


@pytest.fixture
def journey(tmp_path):
    # Execute only the retained offline helpers. Never import the model service.
    source = tmp_path / "retained-helpers"
    source.mkdir()
    for retained, name in (
        ("collector-source.txt", "capture.py"),
        ("journey-source.txt", "journey.py"),
    ):
        shutil.copyfile(REFERENCE / "capture" / retained, source / name)
    spec = importlib.util.spec_from_file_location(
        "retained_http_journey", source / "journey.py"
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _inputs(reference):
    return {
        name: value
        for role in ("protocol", "baseline", "subject")
        for name, value in (
            (role, REFERENCE / f"capture/{role}.json"),
            (f"{role}_sha256", reference["files"][f"capture/{role}.json"]["sha256"]),
        )
    }


@pytest.fixture
def prepared(reference, journey, tmp_path):
    destination = tmp_path / "independent-inputs"
    assert (
        journey.prepare(**_inputs(reference), output=destination)
        == reference["anchors"]
    )
    return destination


def _receipt_options(reference, prepared):
    return {
        "policy_path": prepared / "policy.json",
        "expected_run_digests": {
            role: reference["anchors"][f"{role}_run_digest"] for role in MODELS
        },
        "expected_request_digest": reference["anchors"]["request_digest"],
        "expected_pack_signer_fingerprint": reference["signer_fingerprint"],
        "expected_verifier_identity": "hosted-service-recipient",
        "expected_verifier_fingerprint": reference["verifier_fingerprint"],
        "expected_trust_profile_digest": (
            "sha256:852eea14de4efccb4f746cdd07458f4e72d8719bcbe26e0d4f8a7e2d739b714a"
        ),
    }


def test_complete_inventory_and_physical_source_pins(reference, protocol):
    assert len(FILES | {"reference.json"}) == 35
    assert set(reference["files"]) == FILES
    assert {
        path.relative_to(REFERENCE).as_posix()
        for path in REFERENCE.rglob("*")
        if path.is_file()
    } == FILES | {"reference.json"}
    assert all(not path.is_symlink() for path in REFERENCE.rglob("*"))
    for name, fact in reference["files"].items():
        raw = (REFERENCE / name).read_bytes()
        assert len(raw) == fact["byte_size"], name
        assert _digest(raw) == fact["sha256"], name
    for field, source in (
        ("collector_source_digest", "collector-source.txt"),
        ("journey_source_digest", "journey-source.txt"),
    ):
        assert protocol[field] == _digest((REFERENCE / "capture" / source).read_bytes())
    for field, source in (
        ("service_source_digest", "service-source.txt"),
        ("checkpoint_loading_source_digest", "checkpoint-loader-source.txt"),
    ):
        assert protocol["environment"][field] == _digest(
            (REFERENCE / "capture" / source).read_bytes()
        )
    for role in ("signer", "verifier"):
        key = serialization.load_pem_public_key(
            (REFERENCE / f"{role}.public.pem").read_bytes()
        )
        assert public_key_fingerprint(key) == reference[f"{role}_fingerprint"]
    assert reference["signer_fingerprint"] != reference["verifier_fingerprint"]


def test_cases_reproduce_source_bytes_and_declared_projection(protocol):
    source = (
        ROOT
        / "examples/integrations/evaluator_transaction/lambada_qwen35_deployment_400.jsonl"
    ).read_bytes()
    assert _digest(source) == protocol["environment"]["source_dataset_digest"]
    cases = [
        {"id": row["id"], "input": row["prompt"], "expected": row["expected"].strip()}
        for row in map(json.loads, source.splitlines())
    ]
    assert len(cases) == len({row["id"] for row in cases}) == 400
    assert cases == protocol["cases"]
    assert (
        case_set_digest(freeze_case_set([{**case, "metadata": {}} for case in cases]))
        == protocol["policy"]["expected_case_set_digest"]
    )
    loading = _read(REFERENCE / "capture/checkpoint-loading-protocol.json")
    for role, model in MODELS.items():
        declared = protocol["environment"]["local_service_deployments"][role]
        assert declared["id"] == model
        assert declared == {key: loading["models"][role][key] for key in declared}
    assert (
        loading["models"]["baseline"]["artifact_digest"]
        != loading["models"]["subject"]["artifact_digest"]
    )
    assert loading["policy"] != protocol["policy"]


def test_reconstruction_preserves_all_http_results_and_captured_identity(
    reference, protocol, prepared
):
    for role, model in MODELS.items():
        assert (prepared / f"{role}.json").read_bytes() == (
            REFERENCE / f"evidence/records/{role}.json"
        ).read_bytes()
        run = _read(prepared / f"{role}.json")
        capture = _read(REFERENCE / f"capture/{role}.json")
        assert len(run["records"]) == len(capture["observations"]) == 400
        assert run["artifact_digest"] is None
        identity = run["service_identity"]
        assert identity["kind"] == "hosted_service"
        assert identity["observed_model"] == model
        assert identity["requested_model"] == "mistral-requalification"
        assert identity["exposed_revision"] is None
        assert identity["observation_window"] == capture["observation_window"]
        for record, observation, case in zip(
            run["records"], capture["observations"], protocol["cases"], strict=True
        ):
            assert record["context"]["http_observation"] == observation
            assert record["id"] == observation["id"] == case["id"]
            assert observation["status"] == 200
            assert observation["error"] is record["error"] is None
            body = json.loads(
                base64.b64decode(observation["body_base64"], validate=True)
            )
            assert body["model"] == model
            assert "revision" not in body
            observed = body["service_observation"]
            words = observed["generated_text"].split()
            assert record["output"] == (words[0] if words else "")
            assert record["output"] == body["choices"][0]["message"]["content"]
            assert observed["output_projection"] == "first-whitespace-word-v1"
            assert 0 < len(observed["generated_token_ids"]) <= 8
            assert (
                len(observed["generated_token_ids"])
                == body["usage"]["completion_tokens"]
            )
            assert observation["request"]["messages"] == [
                {"role": "system", "content": ""},
                {"role": "user", "content": case["input"]},
            ]
        matches = sum(row["output"] == row["expected"] for row in run["records"])
        assert matches == {"baseline": 20, "subject": 24}[role]
    assert (prepared / "policy.json").read_bytes() == (
        REFERENCE / "evidence/inputs/policy.json"
    ).read_bytes()
    assert _read(prepared / "request.json")["execution"] == {"mode": "captured"}
    assert not (prepared / "evidence").exists()
    assert reference["external_provider_qualification"] is False


def test_comparative_pass_retains_low_absolute_scores_and_no_quality_floor(protocol):
    policy = protocol["policy"]["metrics"][0]
    assert policy == {
        "name": "final_word_accuracy",
        "kind": "exact_match",
        "configuration": {},
        "direction": "higher",
        "aggregation": "mean",
        "unit": "score",
        "maximum_regression": 0.02,
        "maximum_interval_width": 0.1,
        "minimum_count": 400,
    }
    report = _read(REFERENCE / "evidence/reports/evaluation.report.json")
    assert report["decision"] == "pass"
    metric = report["metrics"][0]
    assert metric["baseline_mean"] == 0.05
    assert metric["subject_mean"] == 0.06
    assert metric["delta"] == pytest.approx(0.01)
    assert metric["count"] == 400 and metric["missing_ids"] == []
    assert metric["interval"]["lower"] == pytest.approx(-0.014437971652410744)
    assert metric["interval"]["upper"] == pytest.approx(0.03525820495719842)
    assert metric["scoring_assurance"] == "recomputed"


def test_earlier_baseline_remains_visible_without_a_comparison(reference, protocol):
    earlier = _read(REFERENCE / "earlier-attempt/protocol.json")
    assert earlier["cases"] == protocol["cases"]
    assert earlier["policy"] == protocol["policy"]
    assert earlier["configuration"] == protocol["configuration"]
    for role in MODELS:
        before_service = earlier["services"][role]
        after_service = protocol["services"][role]
        assert {**before_service, "exposed_revision": None} == after_service
        assert (
            before_service["exposed_revision"]
            == (protocol["environment"]["local_service_deployments"][role]["revision"])
        )
    old = _read(REFERENCE / "earlier-attempt/baseline.json")["observations"]
    corrected = _read(REFERENCE / "capture/baseline.json")["observations"]
    assert len(old) == len(corrected) == 400
    for before, after in zip(old, corrected, strict=True):
        assert before["id"] == after["id"]
        assert before["request"] == after["request"]
        before_body = json.loads(base64.b64decode(before["body_base64"], validate=True))
        after_body = json.loads(base64.b64decode(after["body_base64"], validate=True))
        assert (
            before_body["service_observation"]["generated_token_ids"]
            == (after_body["service_observation"]["generated_token_ids"])
        )
    assert reference["earlier_attempt"]["status"] == "retained_without_comparison"
    assert all(
        attempt["admitted_calls"] == 0 for attempt in reference["startup_attempts"]
    )


@pytest.mark.parametrize("role", MODELS)
def test_changed_capture_is_rejected_before_export(reference, journey, tmp_path, role):
    inputs = _inputs(reference)
    changed = _read(inputs[role])
    changed["observations"][0]["body_base64"] = base64.b64encode(b"{}").decode()
    inputs[role] = tmp_path / "changed.json"
    inputs[role].write_text(json.dumps(changed))
    destination = tmp_path / "refused"
    with pytest.raises(ValueError, match="capture digest mismatch"):
        journey.prepare(**inputs, output=destination)
    assert not destination.exists()


@pytest.mark.parametrize("prefix", ["", "current/"])
def test_original_receipts_authenticate_captured_comparison(
    reference, prepared, prefix
):
    result = verify_signed_verification_receipt(
        REFERENCE / f"{prefix}verification.receipt.json",
        REFERENCE / "evidence",
        **_receipt_options(reference, prepared),
    )
    assert result.ok and result.signed, result.errors
    assert result.statement["verification_scope"] == "captured_comparison"
    assert result.statement["replay_status"] == "completed"
    assert result.statement["verdict"]["decision"] == "pass"
    assert result.statement["subject"] == {
        "kind": "captured_run",
        "run_digest": reference["anchors"]["subject_run_digest"],
    }


@pytest.mark.parametrize(
    "pin",
    [
        "expected_request_digest",
        "expected_pack_signer_fingerprint",
        "expected_verifier_fingerprint",
        "expected_trust_profile_digest",
    ],
)
def test_receipt_rejects_unapproved_anchor(reference, prepared, pin):
    options = _receipt_options(reference, prepared)
    options[pin] = "sha256:" + "0" * 64
    result = verify_signed_verification_receipt(
        REFERENCE / "verification.receipt.json", REFERENCE / "evidence", **options
    )
    assert not result.ok and result.errors


def test_fresh_verification_and_reporting_preserve_original_pack(
    reference, prepared, tmp_path
):
    before = {name: (REFERENCE / name).read_bytes() for name in FILES}
    key, fingerprint = _key(tmp_path / "fresh-verifier.pem")
    receipt = tmp_path / "fresh.receipt.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "verify",
            str(REFERENCE / "evidence"),
            "--policy",
            str(prepared / "policy.json"),
            "--expected-baseline-run",
            reference["anchors"]["baseline_run_digest"],
            "--expected-subject-run",
            reference["anchors"]["subject_run_digest"],
            "--expected-request-digest",
            reference["anchors"]["request_digest"],
            "--expected-signer",
            reference["signer_fingerprint"],
            "--receipt",
            str(receipt),
            "--verifier-signing-key",
            str(key),
            "--verifier-identity",
            "fresh-http-recipient",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    verified = json.loads(result.stdout)
    assert verified["kind"] == "captured"
    assert verified["integrity_ok"] is True
    assert verified["decision"] == "pass"
    assert verified["replay_status"] == "completed"
    options = _receipt_options(reference, prepared)
    options.update(
        expected_verifier_identity="fresh-http-recipient",
        expected_verifier_fingerprint=fingerprint,
        expected_trust_profile_digest=None,
    )
    fresh = verify_signed_verification_receipt(
        receipt, REFERENCE / "evidence", **options
    )
    assert fresh.ok and fresh.signed, fresh.errors
    result = runner.invoke(
        app,
        [
            "report",
            str(REFERENCE / "evidence"),
            "--html",
            str(tmp_path / "report.html"),
            "--markdown",
            str(tmp_path / "report.md"),
            "--junit",
            str(tmp_path / "report.xml"),
        ],
    )
    assert result.exit_code == 0, result.output
    for suffix in ("html", "md"):
        report = (tmp_path / f"report.{suffix}").read_text()
        assert all(model in report for model in MODELS.values())
        assert "5%" in report and "6%" in report
        assert "Model weights are not identified" in report
        assert "Not exposed" in report
    assert (tmp_path / "report.xml").read_bytes()
    assert before == {name: (REFERENCE / name).read_bytes() for name in FILES}
