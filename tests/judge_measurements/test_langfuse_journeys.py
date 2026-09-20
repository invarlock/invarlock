"""Signed scorer journeys from retained Langfuse SDK exports, without SDK imports."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.engine import captured_request_digest, normalize_captured_request
from invarlock.evaluation_records.adapters import load_run
from invarlock.evaluation_records.io import run_digest
from tests.cli.test_import_journey import _key
from tests.judge_measurements.test_evaluator_capture_journeys import (
    _journey,
    _write_request,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "examples/captured-results/references/langfuse"


def retained_sources(tmp_path, metric):
    reference = json.loads((REFERENCE / "reference.json").read_bytes())
    sources, runs = [], []
    for side in ("baseline", "subject"):
        name = f"{metric}-{side}.json"
        raw = (REFERENCE / name).read_bytes()
        binding = reference["exports"][name]
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == binding["sha256"]
        origin = reference["sources"][name]
        original_bytes = (ROOT / origin["path"]).read_bytes()
        assert (
            "sha256:" + hashlib.sha256(original_bytes).hexdigest() == origin["sha256"]
        )
        original_rows = {
            row["id"]: row for row in json.loads(original_bytes)["records"]
        }
        native_rows = json.loads(raw)["result"]["item_results"]
        assert len(native_rows) == binding["record_count"] == len(original_rows) == 400
        assert {row["item"]["metadata"]["invarlock_id"] for row in native_rows} == set(
            original_rows
        )
        for native in native_rows:
            metadata = native["item"]["metadata"]
            original = original_rows[metadata["invarlock_id"]]
            assert metadata["invarlock_retained_source_sha256"] == origin["sha256"]
            assert native["item"]["input"] == original["input"]
            assert native["item"]["expected_output"] == original["expected"]
            assert native["output"] == original["output"]
            assert {
                key: value
                for key, value in metadata.items()
                if not key.startswith("invarlock_")
            } == original["metadata"]
            if metric == "normalized_nll":
                assert (
                    metadata["invarlock_original_likelihood"] == original["likelihood"]
                )
                assert metadata["invarlock_likelihood"] == {
                    **original["likelihood"],
                    "source": {"name": "langfuse", "version": "4.14.1"},
                }
        path = tmp_path / name
        path.write_bytes(raw)
        source = {
            "adapter": "langfuse-json",
            "source": {"name": "langfuse", "version": "4.14.1"},
            "run_id": binding["run_id"],
            "artifact_digest": binding["artifact_digest"],
        }
        if "service_identity" in binding:
            source["service_identity"] = binding["service_identity"]
        runs.append(load_run(path, **source))
        sources.append({"path": name, **source})
    return sources, runs


def comparison_request(tmp_path, metric):
    sources, runs = retained_sources(tmp_path, metric)
    origin = (
        ROOT / "examples/hosted-service/references/mistral-7b-http"
        if metric == "exact_match"
        else ROOT / "examples/captured-results/references/mistral-7b-likelihood"
    )
    policy = json.loads((origin / "evidence/inputs/policy.json").read_bytes())
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    request = {
        "format_version": "invarlock/evaluation-request-v2",
        "execution": {"mode": "captured"},
        "comparison": {
            "baseline": sources[0],
            "subject": sources[1],
            "policy": "policy.json",
        },
        "output": {"evidence": "evidence"},
    }
    request_path = tmp_path / "request.yaml"
    request_path.write_text(yaml.safe_dump(request))
    key, fingerprint = _key(tmp_path / "signer.pem")
    verifier, _ = _key(tmp_path / "verifier.pem")
    trust = {
        "format": "invarlock/trust-inputs-v2",
        "kind": "captured",
        "policy": {"path": "policy.json"},
        "anchors": {
            "baseline_run_digest": run_digest(runs[0]),
            "subject_run_digest": run_digest(runs[1]),
            "request_digest": captured_request_digest(
                normalize_captured_request(
                    request,
                    baseline=runs[0],
                    subject=runs[1],
                    policy=policy,
                )
            ),
            "evidence_signer_fingerprint": fingerprint,
        },
        "verifier": {
            "identity": "langfuse-recipient",
            "signing_key_path": verifier.name,
        },
    }
    trust_path = tmp_path / "trust.json"
    trust_path.write_text(json.dumps(trust))
    return request_path, key, trust_path, runs


@pytest.mark.parametrize(
    "metric,decision,status",
    [
        ("exact_match", "pass", 0),
        ("normalized_nll", "regression", 7),
    ],
)
def test_retained_real_sdk_exports_evaluate_verify_report(
    tmp_path, metric, decision, status
):
    request, key, trust, runs = comparison_request(tmp_path, metric)
    runner = CliRunner()
    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(key),
            "--fail-on-policy",
            "--json",
        ],
    )
    assert evaluated.exit_code == status, evaluated.output
    result = json.loads(evaluated.stdout)
    assert result["decision"] == decision
    assert result["authentication"] == "signed"
    evidence = tmp_path / "evidence"
    verified = runner.invoke(
        app,
        [
            "verify",
            str(evidence),
            "--trust-profile",
            str(trust),
            "--receipt",
            str(tmp_path / "verification.receipt.json"),
            "--json",
        ],
    )
    assert verified.exit_code == status, verified.output
    verification = json.loads(verified.stdout)
    assert verification["integrity_ok"] and verification["replay_status"] == "completed"
    assert verification["decision"] == decision
    report = runner.invoke(app, ["report", str(evidence), "--json"])
    assert report.exit_code == 0, report.output
    report_document = json.loads(report.stdout)
    assert report_document
    comparison = json.loads((evidence / "reports/evaluation.report.json").read_bytes())
    assert comparison["metrics"][0]["count"] == 400
    if metric == "normalized_nll":
        assert comparison["metrics"][0]["baseline_mean"] == pytest.approx(
            0.5446367517489973
        )
        assert comparison["metrics"][0]["subject_mean"] == pytest.approx(
            0.5942868193385209
        )
    html = tmp_path / "report.html"
    rendered = runner.invoke(app, ["report", str(evidence), "--html", str(html)])
    assert rendered.exit_code == 0, rendered.output
    assert "langfuse" in html.read_text()
    for side, run in zip(("baseline", "subject"), runs, strict=True):
        assert json.loads((evidence / f"records/{side}.json").read_bytes()) == run


def synthetic_judge_request(tmp_path):
    """Synthetic full judge measurements test plumbing, not actual model judging."""
    raw = json.loads((REFERENCE / "exact_match-baseline.json").read_bytes())
    sources, runs = [], []
    for side, marker in (("baseline", "a"), ("subject", "b")):
        exported = copy.deepcopy(raw)
        exported["result"]["run_name"] = f"synthetic-judge-{side}"
        exported["result"]["item_results"] = [
            {
                "item": {
                    "input": "Capital of France?",
                    "expected_output": "Paris",
                    "metadata": {"invarlock_id": "case-1"},
                },
                "output": "Paris",
                "evaluations": [{"name": "judge", "value": 1.0}],
                "trace_id": None,
                "dataset_run_id": None,
            }
        ]
        path = tmp_path / f"{side}.json"
        path.write_text(json.dumps(exported))
        source = {
            "adapter": "langfuse-json",
            "source": {"name": "langfuse", "version": "4.14.1"},
            "run_id": exported["result"]["run_name"],
            "artifact_digest": "sha256:" + marker * 64,
        }
        runs.append(load_run(path, **source))
        sources.append({"path": path.name, **source})
    return (*_write_request(tmp_path, sources, runs), runs)


def test_langfuse_judge_replays_full_synthetic_measurements(tmp_path):
    request, key, recipient, runs = synthetic_judge_request(tmp_path)
    _journey(tmp_path, request, key, recipient, runs)


def test_langfuse_scalar_judge_scores_do_not_replace_measurements(tmp_path):
    request, key, _, _ = synthetic_judge_request(tmp_path)
    (tmp_path / "measurements.json").write_text(json.dumps({"score": 1.0}))
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(request),
            "--signing-key",
            str(key),
            "--fail-on-policy",
            "--json",
        ],
    )
    assert result.exit_code != 0
    assert not (tmp_path / "evidence").exists()
