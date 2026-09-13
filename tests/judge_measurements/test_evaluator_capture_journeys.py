"""Signed offline scorer journeys preserve native exports and evaluator identities."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.evaluation_record_contracts.contracts import digest
from invarlock.evaluation_records.adapters import load_run
from invarlock.evaluation_records.io import run_digest
from invarlock.evaluator_capture import capture_evaluator_run
from invarlock.judge_measurements import native_workflow
from invarlock.judge_measurements.analysis import (
    analyze_measurements,
    decode_analysis_policy,
)
from invarlock.judge_measurements.captured_workflow import prepare_evaluator_judge
from invarlock.judge_measurements.evidence import DECISION_SCOPE, object_sha256
from tests.cli.test_import_journey import _key
from tests.core.test_native_judge_transaction import _completed_measurements, _recipe

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "tests/evaluation_records/fixtures"
PROFILES = json.loads(
    (ROOT / "examples/evaluator-qualification/matrix.json").read_bytes()
)["profiles"]
ECOSYSTEMS = [profile["profile_id"] for profile in PROFILES]
SIGNER = "reviewed-capture-signer"


@pytest.fixture(autouse=True)
def no_collection(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("retained measurement import attempted judge collection")

    monkeypatch.setattr(native_workflow, "collection_preflight", forbidden)
    monkeypatch.setattr(native_workflow, "collect_frozen", forbidden)


@pytest.fixture
def capture_helper():
    path = ROOT / "examples/evaluator-qualification/maintained/capture.py"
    spec = importlib.util.spec_from_file_location("capture_journey_helper", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_request(tmp_path, sources, runs):
    recipe = _recipe([row["id"] for row in runs[0]["records"]])
    recipe["analysis"]["allowed_degradation"] = "1"
    plan, policy = prepare_evaluator_judge(recipe, *runs)
    measurements = _completed_measurements(
        plan=plan, baseline_run=runs[0], subject_run=runs[1]
    )
    analysis = analyze_measurements(
        plan,
        measurements,
        decode_analysis_policy(policy, plan=plan),
        baseline_run=runs[0],
        subject_run=runs[1],
    ).to_dict()
    assert analysis["decision"] == "pass"
    (tmp_path / "policy.json").write_text(json.dumps(recipe))
    (tmp_path / "measurements.json").write_text(json.dumps(measurements))
    value = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": sources[0],
            "subject": sources[1],
            "metric": "judge",
            "policy": "policy.json",
            "judge": {
                "workspace": "judge-work",
                "signer_identity": SIGNER,
                "measurements": "measurements.json",
            },
        },
        "output": {"evidence": "evidence"},
    }
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump(value))
    key, fingerprint = _key(tmp_path / "signer.pem")
    recipient = {
        "format": "invarlock/judge-measurement-recipient-policy-v1",
        "decision_scope": DECISION_SCOPE,
        "intended_subject": runs[1]["artifact_digest"],
        "required_metric_name": policy["metric_name"],
        "trusted_signer": {"identity": SIGNER, "public_key_sha256": fingerprint},
        "bindings": {
            "baseline_run_sha256": run_digest(runs[0]),
            "subject_run_sha256": run_digest(runs[1]),
            "case_set_sha256": plan["case_set_sha256"],
            "plan_sha256": object_sha256(plan),
            "measurements_sha256": object_sha256(measurements),
            "analysis_policy_sha256": object_sha256(policy),
            "analysis_result_sha256": object_sha256(analysis),
        },
        "required_decision": "pass",
    }
    recipient_path = tmp_path / "recipient.json"
    recipient_path.write_text(json.dumps(recipient))
    return request, key, recipient_path


def _journey(tmp_path, request, key, recipient, runs):
    runner = CliRunner()
    evaluation = runner.invoke(
        app, ["evaluate", str(request), "--signing-key", str(key), "--json"]
    )
    assert evaluation.exit_code == 0, evaluation.output
    result = json.loads(evaluation.stdout)
    assert result["kind"] == "judge"
    assert result["authentication"] == "signed"
    assert result["source_assurance"] == "captured_inputs"
    assert result["decision"] == "pass"
    evidence = tmp_path / "evidence"
    assert not (evidence / "native_capture.json").exists()
    assert not (tmp_path / "judge-work").exists()
    for side, expected in zip(("baseline", "subject"), runs, strict=True):
        assert json.loads((evidence / f"{side}_run.json").read_bytes()) == expected
    verification = runner.invoke(
        app, ["verify", str(evidence), "--trust-profile", str(recipient), "--json"]
    )
    assert verification.exit_code == 0, verification.output
    verified = json.loads(verification.stdout)
    assert verified["kind"] == "judge"
    assert verified["authenticated"] and verified["replayed"] and verified["accepted"]
    report = runner.invoke(app, ["report", str(evidence), "--json"])
    assert report.exit_code == 0, report.output
    rendered = json.loads(report.stdout)
    assert rendered["analysis"]["decision"] == "pass"
    assert (
        rendered["assurance"]["authentication"]
        == "signature_present_not_recipient_authorized"
    )
    html_path = tmp_path / "report.html"
    html = runner.invoke(app, ["report", str(evidence), "--html", str(html_path)])
    assert html.exit_code == 0, html.output
    assert runs[0]["source"]["name"] in html_path.read_text()


@pytest.mark.parametrize(
    "filename,adapter,pointer,version",
    [
        ("inspect-0.3.254.json", "inspect-json", "/input", "0.3.254"),
        ("lm-eval-0.4.12.jsonl", "lm-eval-samples", "/context/arguments/0/0", "0.4.12"),
        ("promptfoo-0.121.19.jsonl", "promptfoo-jsonl", "/context/prompt", "0.121.19"),
    ],
)
def test_retained_exports_project_evaluate_verify_and_report(
    tmp_path, filename, adapter, pointer, version
):
    sources, runs = [], []
    raw = (EXPORTS / filename).read_bytes()
    for side, marker in (("baseline", "a"), ("subject", "b")):
        path = tmp_path / f"{side}-{filename}"
        path.write_bytes(raw)
        source = {
            "adapter": adapter,
            "source": {"name": adapter, "version": version},
            "run_id": side,
            "artifact_digest": "sha256:" + marker * 64,
            "input_projection": {"kind": "json-pointer", "pointer": pointer},
        }
        runs.append(load_run(path, **source))
        sources.append({"path": path.name, **source})
    request, key, recipient = _write_request(tmp_path, sources, runs)
    _journey(tmp_path, request, key, recipient, runs)


@pytest.mark.parametrize("ecosystem", ECOSYSTEMS)
def test_all_maintained_evaluator_identities_use_captured_judge_journey(
    tmp_path, capture_helper, ecosystem
):
    records = [
        {
            "id": "case-1",
            "input": "What is the capital of France?",
            "expected": "Paris",
            "output": "Paris",
        },
        {
            "id": "case-2",
            "input": "Return the release label.",
            "expected": "approved",
            "output": "candidate",
        },
    ]
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records))
    sources, runs = [], []
    for side, marker in (("baseline", "a"), ("subject", "b")):
        run = capture_helper.capture_records(
            ecosystem=ecosystem,
            records_path=records_path,
            run_id=side,
            artifact_digest="sha256:" + marker * 64,
            source_version="1.2.3",
        )
        assert (
            run["source"]["name"]
            == capture_helper.profiles()[ecosystem]["upstream"]["name"]
        )
        assert run["source"]["version"] == "1.2.3"
        path = tmp_path / f"{side}.json"
        path.write_text(json.dumps(run))
        sources.append({"path": path.name, "adapter": "invarlock"})
        runs.append(run)
    request, key, recipient = _write_request(tmp_path, sources, runs)
    _journey(tmp_path, request, key, recipient, runs)


def _structured_request(tmp_path):
    row = {
        "id": "one",
        "input": {"prompt": "question", "case_fact": "original"},
        "expected": "yes",
        "output": "yes",
    }
    runs, sources = [], []
    for side, marker in (("baseline", "a"), ("subject", "b")):
        run = capture_evaluator_run(
            [row],
            source={"name": "case-evaluator", "version": "1"},
            run_id=side,
            artifact_digest="sha256:" + marker * 64,
            input_projection={"kind": "json-pointer", "pointer": "/input/prompt"},
        )
        path = tmp_path / f"{side}.json"
        path.write_text(json.dumps(run))
        runs.append(run)
        sources.append({"path": path.name, "adapter": "invarlock"})
    return _write_request(tmp_path, sources, runs)


@pytest.mark.parametrize(
    "mutation", ["projected-text", "unselected-source", "measurements", "aggregate"]
)
def test_captured_judge_rejects_modified_or_aggregate_inputs_before_publication(
    tmp_path, mutation
):
    request, key, _ = _structured_request(tmp_path)
    path = tmp_path / (
        "measurements.json" if mutation == "measurements" else "subject.json"
    )
    value = json.loads(path.read_bytes())
    if mutation == "projected-text":
        value["records"][0]["input"] = "changed prompt"
    elif mutation == "unselected-source":
        binding = value["records"][0]["context"]["input_projection"]
        binding["source"]["input"]["case_fact"] = "changed fact"
        binding["source_digest"] = digest(binding["source"])
    elif mutation == "measurements":
        value["trials"][0]["attempts"][0]["request"]["text"] = "changed request"
    else:
        value = {"accuracy": 1, "count": 1}
    path.write_text(json.dumps(value))
    result = CliRunner().invoke(
        app, ["evaluate", str(request), "--signing-key", str(key), "--json"]
    )
    assert result.exit_code != 0, result.output
    assert not (tmp_path / "evidence").exists()
    assert not (tmp_path / "judge-work").exists()


@pytest.mark.parametrize(
    "pointer", ["/input", "/context/missing", "/context/arguments/0", "/output"]
)
def test_ambiguous_or_missing_native_projection_fails_before_judging(tmp_path, pointer):
    request, key, _ = _structured_request(tmp_path)
    export = tmp_path / "lm.jsonl"
    export.write_bytes((EXPORTS / "lm-eval-0.4.12.jsonl").read_bytes())
    value = yaml.safe_load(request.read_text())
    for side in ("baseline", "subject"):
        value["comparison"][side] = {
            "path": export.name,
            "adapter": "lm-eval-samples",
            "source": {"name": "lm_eval", "version": "0.4.12"},
            "run_id": side,
            "artifact_digest": "sha256:" + "a" * 64,
            "input_projection": {"kind": "json-pointer", "pointer": pointer},
        }
    request.write_text(yaml.safe_dump(value))
    result = CliRunner().invoke(
        app, ["evaluate", str(request), "--signing-key", str(key), "--json"]
    )
    assert result.exit_code != 0, result.output
    assert "projection" in result.output or "pointer" in result.output
    assert not (tmp_path / "evidence").exists()
