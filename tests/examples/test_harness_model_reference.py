"""Replay retained distinct-checkpoint likelihood evidence without model execution."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.engine import case_set_digest, freeze_case_set
from invarlock.evidence_pack_integrity import public_key_fingerprint
from invarlock.evidence_receipt import verify_signed_verification_receipt
from tests.cli.test_import_journey import _key
from tests.examples.test_harness_model_handoff import module

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/captured-results/references/mistral-7b-likelihood"
FILES = {
    f"capture/{role}/{name}.json"
    for role in ("baseline", "subject")
    for name in ("protocol", "manifest", "tokenizations", "raw-results")
} | {
    "initial-load-failure.json",
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
}
MODELS = {
    "baseline": (
        "mistralai/Mistral-7B-v0.1",
        "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    ),
    "subject": (
        "mistralai/Mistral-7B-Instruct-v0.1",
        "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca",
    ),
}


def _digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


@pytest.fixture
def reference():
    return json.loads((REFERENCE / "reference.json").read_bytes())


@pytest.fixture
def protocol():
    return json.loads((REFERENCE / "capture/baseline/protocol.json").read_bytes())


@pytest.fixture
def prepared(reference, tmp_path):
    destination = tmp_path / "independent-inputs"
    anchors = module().project_capture(
        REFERENCE / "capture", destination, **reference["capture_pins"]
    )
    assert {key: anchors[key] for key in reference["anchors"]} == reference["anchors"]
    assert anchors["source_assurance"] == "captured_inputs"
    return destination


def _receipt_options(reference, prepared):
    return {
        "policy_path": prepared / "policy.json",
        "expected_run_digests": {
            side: reference["anchors"][f"{side}_run_digest"] for side in MODELS
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
    assert reference["format"] == "invarlock/harness-model-reference-v1"
    assert reference["scope"] == "distinct-checkpoint-narrative-likelihood"
    assert reference["case_count"] == 400
    assert set(reference["anchors"]) == {
        "baseline_run_digest",
        "subject_run_digest",
        "request_digest",
    }
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
        assert _digest(raw) == fact["sha256"], name
    for name, expected in (
        ("signer.public.pem", reference["signer_fingerprint"]),
        (
            "verifier.public.pem",
            reference["retained_verifier"]["signing_key_fingerprint"],
        ),
    ):
        key = serialization.load_pem_public_key((REFERENCE / name).read_bytes())
        assert public_key_fingerprint(key) == expected


def test_real_capture_binds_distinct_models_source_and_complete_schedule(
    reference, protocol
):
    assert protocol["source"] == {"name": "lm-eval", "version": "0.4.12"}
    assert protocol["capture_script_sha256"] == _digest(
        (ROOT / "examples/captured-results/harness_model_comparison.py").read_bytes()
    )
    assert len(protocol["cases"]) == 400
    for role, (model_id, revision) in MODELS.items():
        model = protocol["models"][role]
        assert (model["id"], model["revision"]) == (model_id, revision)
        assert reference["models"][role] == {
            field: model[field]
            for field in ("id", "revision", "artifact_digest", "tokenizer_digest")
        }
        raw_protocol = (REFERENCE / f"capture/{role}/protocol.json").read_bytes()
        assert _digest(raw_protocol) == reference["capture_pins"]["protocol_sha256"]
        raw_bytes = (REFERENCE / f"capture/{role}/raw-results.json").read_bytes()
        assert _digest(raw_bytes) == reference["capture_pins"][f"{role}_sha256"]
        raw = json.loads(raw_bytes)
        assert raw["metadata"]["status"] == "complete"
        assert raw["metadata"]["harness_loglikelihood_call_count"] == 400
        assert raw["metadata"]["case_count"] == len(raw["records"]) == 400
        assert [row["id"] for row in raw["records"]] == [
            case["id"] for case in protocol["cases"]
        ]
    assert (
        reference["models"]["baseline"]["artifact_digest"]
        != reference["models"]["subject"]["artifact_digest"]
    )
    assert (
        reference["anchors"]["baseline_run_digest"]
        != reference["anchors"]["subject_run_digest"]
    )


def test_projection_reproduces_original_source_bytes_ids_and_metadata(protocol):
    dataset = protocol["dataset"]
    source_path = "examples/integrations/evaluator_transaction/lambada_qwen35_deployment_400.jsonl"
    manifest_path = "examples/integrations/evaluator_transaction/deployment_corpus.json"
    assert dataset["source"]["path"] == source_path
    assert dataset["source_manifest"]["path"] == manifest_path
    for field, path in (("source", source_path), ("source_manifest", manifest_path)):
        raw = (ROOT / path).read_bytes()
        assert len(raw) == dataset[field]["byte_size"]
        assert _digest(raw) == dataset[field]["sha256"]
    manifest = json.loads((ROOT / manifest_path).read_bytes())
    assert dataset["original_source"] == manifest["source"]
    assert dataset["original_selection"]["profile_id"] == manifest["profile_id"]
    for field in (
        "algorithm",
        "seed",
        "indices_sha256",
        "selected_source_lines_sha256",
        "criteria",
    ):
        assert dataset["original_selection"][field] == manifest["selection"][field]
    projection = dataset["projection"]
    assert projection["algorithm"] == "utf8-two-thirds-whitespace-boundary-v1"
    assert projection["candidate_pattern"] == r"(?<!\s)\s+(?=\S)"
    assert projection["minimum_context_utf8_bytes"] == 64
    assert projection["minimum_reference_utf8_bytes"] == 64
    original = [
        json.loads(line) for line in (ROOT / source_path).read_bytes().splitlines()
    ]
    reconstructed = []
    for row_index, row in enumerate(original):
        text = row["prompt"] + row["expected"]
        candidates = [
            match.start()
            for match in re.finditer(r"(?<!\s)\s+(?=\S)", text)
            if len(text[: match.start()].encode()) >= 64
            and len(text[match.start() :].encode()) >= 64
        ]
        split = min(
            candidates,
            key=lambda i: (abs(3 * len(text[:i].encode()) - 2 * len(text.encode())), i),
        )
        source_index = manifest["selection"]["indices"][row_index]
        assert row["id"] == f"lambada-openai-{source_index:04}"
        reconstructed.append(
            {
                "id": row["id"],
                "input": text[:split],
                "expected": text[split:],
                "metadata": {
                    "retained_row_index": str(row_index),
                    "source_record_id": row["id"],
                    "source_row_index": str(source_index),
                    "source_text_sha256": _digest(text.encode()),
                    "split_character_index": str(split),
                    "split_utf8_byte_index": str(len(text[:split].encode())),
                },
            }
        )
    assert len(reconstructed) == dataset["record_count"] == 400
    assert reconstructed == protocol["cases"]
    raw_cases = _canonical(reconstructed)
    assert len(raw_cases) == dataset["cases"]["byte_size"]
    assert _digest(raw_cases) == dataset["cases"]["sha256"]
    assert (
        case_set_digest(freeze_case_set(reconstructed))
        == protocol["policy"]["expected_case_set_digest"]
    )


def test_initial_loading_failure_is_disclosed_without_repeated_measurements(
    reference, protocol
):
    failure = json.loads((REFERENCE / "initial-load-failure.json").read_bytes())
    assert failure["stage"] == "checkpoint-loading"
    assert failure["role"] == "baseline"
    assert failure["completed_cases"] == failure["measured_results_repeated"] == 0
    assert failure["subject_started"] is False
    assert failure["exit_signal"] == 11
    assert failure["original_model_identities"] == reference["models"]
    assert failure["dataset_sha256"] == protocol["dataset"]["cases"]["sha256"]
    assert failure["protocol_sha256"] != reference["capture_pins"]["protocol_sha256"]
    assert failure["capture_script_sha256"] != protocol["capture_script_sha256"]
    for pin in (
        failure["protocol_sha256"],
        failure["capture_script_sha256"],
        failure["execution_log"]["sha256"],
    ):
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", pin)
    assert failure["execution_log"]["byte_size"] > 0
    assert failure["upstream_issue"] == (
        "https://github.com/huggingface/transformers/issues/48029"
    )
    assert protocol["configuration"]["async_weight_loading"] is False


def test_regenerated_inputs_and_full_metric_match_frozen_reference(reference, prepared):
    for side in MODELS:
        assert (prepared / f"{side}.json").read_bytes() == (
            REFERENCE / f"evidence/records/{side}.json"
        ).read_bytes()
    assert (prepared / "policy.json").read_bytes() == (
        REFERENCE / "evidence/inputs/policy.json"
    ).read_bytes()
    report = json.loads(
        (REFERENCE / "evidence/reports/evaluation.report.json").read_bytes()
    )
    assert report["decision"] == reference["expected_decision"]
    assert report["metrics"] == [reference["expected_metric"]]
    metric = report["metrics"][0]
    assert metric["count"] == 400
    assert metric["missing_ids"] == []
    assert metric["kind"] == "normalized_nll_per_utf8_byte"
    assert metric["interval_unit"] == "ratio"
    assert not (prepared / "evidence").exists()


def test_retained_receipt_authenticates_frozen_decision(reference, prepared):
    result = verify_signed_verification_receipt(
        REFERENCE / "verification.receipt.json",
        REFERENCE / "evidence",
        **_receipt_options(reference, prepared),
    )
    assert result.ok and result.signed, result.errors
    assert result.statement["replay_status"] == "completed"
    assert result.statement["verdict"]["integrity_ok"] is True
    assert result.statement["verdict"]["decision"] == reference["expected_decision"]
    assert result.statement["verdict"]["ok"] is (
        reference["expected_decision"] == "pass"
    )


@pytest.mark.parametrize(
    "pin",
    [
        "expected_pack_signer_fingerprint",
        "expected_verifier_fingerprint",
        "expected_trust_profile_digest",
        "expected_request_digest",
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


@pytest.mark.parametrize("role", MODELS)
def test_changed_raw_measurement_is_rejected_under_original_pin(
    reference, tmp_path, role
):
    capture = tmp_path / "changed-capture"
    shutil.copytree(REFERENCE / "capture", capture)
    path = capture / role / "raw-results.json"
    raw = json.loads(path.read_bytes())
    raw["records"][0]["likelihood"]["logprob_sum"] -= 1.0
    path.write_bytes(_canonical(raw))
    destination = tmp_path / "refused"
    with pytest.raises(ValueError, match="raw results differ from independent pin"):
        module().project_capture(capture, destination, **reference["capture_pins"])
    assert not destination.exists()


def test_fresh_cli_replay_and_report_preserve_pack_and_show_both_models(
    reference, prepared, tmp_path
):
    before = {
        name: (REFERENCE / name).read_bytes()
        for name in FILES
        if name.startswith("evidence/")
    }
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
            "fresh-model-recipient",
            "--json",
        ],
    )
    passed = reference["expected_decision"] == "pass"
    assert result.exit_code == (0 if passed else 7), result.output
    verified = json.loads(result.stdout)
    assert verified["ok"] is passed
    assert verified["integrity_ok"] is True
    assert verified["replay_status"] == "completed"
    options = _receipt_options(reference, prepared)
    options.update(
        expected_verifier_identity="fresh-model-recipient",
        expected_verifier_fingerprint=fingerprint,
        expected_trust_profile_digest=None,
    )
    fresh = verify_signed_verification_receipt(
        receipt, REFERENCE / "evidence", **options
    )
    assert fresh.ok and fresh.signed, fresh.errors
    assert fresh.statement["verdict"]["decision"] == reference["expected_decision"]
    assert (
        fresh.verifier_fingerprint
        != reference["retained_verifier"]["signing_key_fingerprint"]
    )
    report_path = tmp_path / "report.html"
    result = runner.invoke(
        app, ["report", str(REFERENCE / "evidence"), "--html", str(report_path)]
    )
    assert result.exit_code == 0, result.output
    html = report_path.read_text()
    for model_id, revision in MODELS.values():
        assert model_id in html
        assert revision in html
    assert before == {name: (REFERENCE / name).read_bytes() for name in before}
