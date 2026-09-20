"""Signed captured passes cannot contradict their displayed policy requirements."""

import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from typer.testing import CliRunner

from invarlock import captured_reporting, record_reporting
from invarlock.captured_contracts import load_payloads
from invarlock.cli.app import app
from invarlock.evaluation_record_contracts.contracts import (
    EvaluationRecordsError,
    validate,
)
from invarlock.evaluation_records.templates import example_project
from invarlock.evidence_reporting import EvidenceReportError, render_evidence
from tests._evaluation_support import build_pack, pack_json, rebind_pack, write_snapshot
from tests.evaluation_comparison.test_likelihood import policy as nll_policy
from tests.evaluation_comparison.test_likelihood import row, run


@pytest.mark.parametrize("contradiction", ["bound", "count", "width", "minimum", "nll"])
def test_signed_pass_with_false_requirement_is_refused_before_output(
    tmp_path, contradiction
):
    if contradiction == "nll":
        baseline = run([row("a"), row("b")])
        subject = run([row("a"), row("b")])
        policy = nll_policy()
    else:
        baseline, subject, policy = example_project("classification")
        policy["metrics"] = policy["metrics"][:1]
        policy["slices"] = []
    key = Ed25519PrivateKey.generate()
    original = build_pack(baseline, subject, policy, key)
    comparison = pack_json(original, "report")
    metric = comparison["metrics"][0]
    assert comparison["decision"] == metric["decision"] == "pass"
    if contradiction == "bound":
        metric["interval"].update(lower=-0.3, upper=0.09)
    elif contradiction == "count":
        metric["count"] = policy["metrics"][0]["minimum_count"] - 1
    elif contradiction == "width":
        metric["interval"].update(lower=-0.19, upper=0.3)
    elif contradiction == "minimum":
        metric.update(baseline_mean=0.7, subject_mean=0.7)
    else:
        metric["interval"].update(lower=1.0, upper=1.2)
    validate(comparison, "comparison")
    validate(policy, "policy")
    snapshot = rebind_pack(original, key, report=comparison)
    # Authentication and bindings succeed; this is a semantic contradiction,
    # not a damaged checksum or a stale signature.
    manifest, payloads, signer = load_payloads(snapshot)
    assert manifest["authentication"] == "signed"
    assert payloads["report"] == comparison
    pack = tmp_path / "evidence"
    pack.mkdir()
    write_snapshot(pack, snapshot)
    before = {
        p.relative_to(pack): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in pack.rglob("*")
        if p.is_file()
    }
    diagnostic = "recorded pass contradicts"
    with pytest.raises(EvaluationRecordsError, match=diagnostic):
        record_reporting.render_reports(comparison, evidence=snapshot)
    with pytest.raises(ValueError, match=diagnostic):
        captured_reporting._view(manifest, payloads, signer)
    html = tmp_path / "report.html"
    markdown = tmp_path / "report.md"
    junit = tmp_path / "report.xml"
    with pytest.raises(EvidenceReportError, match=diagnostic) as error:
        render_evidence(pack, html_path=html, markdown_path=markdown, junit_path=junit)
    assert error.value.payload["written_outputs"] == {}
    cli = CliRunner().invoke(app, ["report", str(pack), "--html", str(html)])
    assert cli.exit_code != 0
    assert diagnostic in cli.output
    assert not any(p.exists() for p in (html, markdown, junit))
    assert before == {
        p.relative_to(pack): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in pack.rglob("*")
        if p.is_file()
    }
    assert snapshot.files == {str(p): (pack / p).read_bytes() for p in before}
    # Without the policy these particular requirements cannot be checked. Keep
    # that bounded projection available, with its original decision and limits.
    bare = record_reporting._view(comparison, None)
    assert bare.decision == "pass"
    assert "policy thresholds are unavailable" in bare.summary
    assert "has not been independently replayed" in bare.summary
    assert "other configured checks passed" not in bare.summary
    assert not any(check.passed is False for check in bare.metrics[0].checks)
