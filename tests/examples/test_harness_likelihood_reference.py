"""Replay the retained real Harness capture using independently recorded anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_receipt import verify_signed_verification_receipt
from tests.cli.test_import_journey import _key
from tests.examples.test_harness_likelihood_handoff import helper as helper

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/captured-results/references/harness-likelihood"
FILES = {
    "capture/manifest.json",
    "capture/raw-results.json",
    "evidence/checksums.sha256",
    "evidence/inputs/policy.json",
    "evidence/manifest.json",
    "evidence/manifest.signature.json",
    "evidence/records/baseline.json",
    "evidence/records/subject.json",
    "evidence/reports/evaluation.report.json",
    "evidence/request.json",
    "signer.public.pem",
    "verification.receipt.json",
    "verifier.public.pem",
}


@pytest.fixture
def reference():
    return json.loads((REFERENCE / "reference.json").read_bytes())


@pytest.fixture
def prepared(helper, reference, tmp_path):
    destination = tmp_path / "independent-inputs"
    anchors = helper.prepare(
        REFERENCE / "capture",
        destination,
        manifest_sha256=reference["files"]["capture/manifest.json"]["sha256"],
        results_sha256=reference["files"]["capture/raw-results.json"]["sha256"],
    )
    assert anchors == reference["anchors"]
    return destination


def _receipt_options(reference, prepared):
    return {
        "policy_path": prepared / "policy.json",
        "expected_run_digests": {
            side: reference["anchors"][f"{side}_run_digest"]
            for side in ("baseline", "subject")
        },
        "expected_request_digest": reference["anchors"]["request_digest"],
        "expected_pack_signer_fingerprint": reference["signer_fingerprint"],
        "expected_verifier_identity": reference["retained_verifier"]["identity"],
        "expected_verifier_fingerprint": reference["retained_verifier"][
            "signing_key_fingerprint"
        ],
        "expected_trust_profile_digest": reference["retained_verifier"][
            "trust_profile_digest"
        ],
    }


def test_exact_reference_inventory_and_physical_hashes(reference):
    assert reference["format"] == "invarlock/harness-likelihood-reference-v1"
    assert reference["scope"] == "same-model-cpu-likelihood-compatibility"
    assert reference["case_count"] == 6
    assert set(reference["files"]) == FILES
    assert {
        path.relative_to(REFERENCE).as_posix()
        for path in REFERENCE.rglob("*")
        if path.is_file()
    } == FILES | {"reference.json", "README.md"}
    assert all(not path.is_symlink() for path in REFERENCE.rglob("*"))
    for name, fact in reference["files"].items():
        raw = (REFERENCE / name).read_bytes()
        assert len(raw) == fact["byte_size"], name
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == fact["sha256"], name
    for name, expected in (
        ("signer.public.pem", reference["signer_fingerprint"]),
        (
            "verifier.public.pem",
            reference["retained_verifier"]["signing_key_fingerprint"],
        ),
    ):
        key = serialization.load_pem_public_key((REFERENCE / name).read_bytes())
        assert public_key_fingerprint(key) == expected


def test_real_capture_is_bound_to_current_capture_script_and_original_cases(reference):
    manifest = json.loads((REFERENCE / "capture/manifest.json").read_bytes())
    fact = manifest["capture_script"]
    assert fact["path"] == "examples/captured-results/harness_likelihood_rehearsal.py"
    source = (ROOT / fact["path"]).read_bytes()
    assert len(source) == fact["byte_size"]
    assert "sha256:" + hashlib.sha256(source).hexdigest() == fact["sha256"]
    assert manifest["source"] == {"name": "lm-eval", "version": "0.4.12"}
    assert len(manifest["cases"]) == reference["case_count"]
    assert (
        next(case for case in manifest["cases"] if case["id"] == "unicode")["expected"]
        == " café"
    )


def test_regenerated_inputs_and_report_match_external_reference(reference, prepared):
    for side in ("baseline", "subject"):
        assert (prepared / f"{side}.json").read_bytes() == (
            REFERENCE / f"evidence/records/{side}.json"
        ).read_bytes()
    assert (prepared / "policy.json").read_bytes() == (
        REFERENCE / "evidence/inputs/policy.json"
    ).read_bytes()
    assert json.loads((prepared / "anchors.json").read_bytes()) == reference["anchors"]
    assert not (prepared / "evidence").exists()
    report = json.loads(
        (REFERENCE / "evidence/reports/evaluation.report.json").read_bytes()
    )
    assert report["decision"] == reference["expected_decision"] == "pass"
    metric = report["metrics"][0]
    assert metric["ratio"] == reference["expected_ratio"] == 1.0
    assert metric["interval"] == reference["expected_interval"]
    assert (
        metric["baseline_mean"]
        == metric["subject_mean"]
        == reference["expected_mean_nll"]
    )
    assert metric["count"] == reference["case_count"]
    assert metric["missing_ids"] == []


def test_retained_receipt_verifies_only_with_independent_recipient_pins(
    reference, prepared
):
    options = _receipt_options(reference, prepared)
    result = verify_signed_verification_receipt(
        REFERENCE / "verification.receipt.json", REFERENCE / "evidence", **options
    )
    assert result.ok and result.signed, result.errors
    assert result.verifier_fingerprint == options["expected_verifier_fingerprint"]
    assert result.statement["replay_status"] == "completed"
    assert result.statement["verdict"]["decision"] == reference["expected_decision"]


@pytest.mark.parametrize(
    "pin",
    [
        "expected_pack_signer_fingerprint",
        "expected_verifier_fingerprint",
        "expected_trust_profile_digest",
    ],
)
def test_retained_receipt_cannot_supply_its_own_trust_pins(reference, prepared, pin):
    options = _receipt_options(reference, prepared)
    options[pin] = "sha256:" + "0" * 64
    result = verify_signed_verification_receipt(
        REFERENCE / "verification.receipt.json", REFERENCE / "evidence", **options
    )
    assert not result.ok
    assert result.errors


def test_fresh_cli_verifier_replays_retained_evidence_and_renders_model_identity(
    reference, prepared, tmp_path
):
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
            "fresh-harness-recipient",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    verified = json.loads(result.stdout)
    assert verified["ok"] and verified["replay_status"] == "completed"
    options = _receipt_options(reference, prepared)
    options.update(
        expected_verifier_identity="fresh-harness-recipient",
        expected_verifier_fingerprint=fingerprint,
        expected_trust_profile_digest=None,
    )
    fresh = verify_signed_verification_receipt(
        receipt, REFERENCE / "evidence", **options
    )
    assert fresh.ok and fresh.signed, fresh.errors
    assert (
        fresh.verifier_fingerprint
        != reference["retained_verifier"]["signing_key_fingerprint"]
    )
    report_path = tmp_path / "report.html"
    result = runner.invoke(
        app, ["report", str(REFERENCE / "evidence"), "--html", str(report_path)]
    )
    assert result.exit_code == 0, result.output
    model = json.loads((REFERENCE / "capture/manifest.json").read_bytes())[
        "model_identity"
    ]
    html = report_path.read_text()
    assert model["id"] in html
    assert model["revision"] in html
