import json
from xml.etree import ElementTree

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.evidence_reporting import EvidenceReportError, render_evidence
from tests.evidence_sets.test_verification import fixture


def test_combined_report_keeps_methods_and_original_counts(tmp_path):
    root, _ = fixture(tmp_path)
    html, markdown, junit = (
        tmp_path / name for name in ("report.html", "report.md", "report.xml")
    )
    report = render_evidence(
        root, html_path=html, markdown_path=markdown, junit_path=junit
    )
    text = html.read_text()
    assert "Deterministic · exact" in text and "Judge ·" in text
    assert "no joint confidence guarantee" in text
    assert "metric-navigation" in text
    assert "16 cases" in text
    assert "Not performed by report." in text
    assert json.loads(report.as_json())["recipient_acceptance"] == "not_performed"
    assert len(ElementTree.parse(junit).getroot().findall("testcase")) == 2
    assert (root / "evidence-set.json").is_file()


def test_report_cannot_overwrite_or_write_inside_evidence(tmp_path):
    root, _ = fixture(tmp_path)
    with pytest.raises(EvidenceReportError, match="outside"):
        render_evidence(root, html_path=root / "report.html")
    output = tmp_path / "report.html"
    output.write_text("keep")
    with pytest.raises(EvidenceReportError, match="already exists"):
        render_evidence(root, html_path=output)
    assert output.read_text() == "keep"


def test_verify_and_report_cli_dispatch(tmp_path):
    root, policy = fixture(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "verify",
            str(root),
            "--trust-profile",
            str(policy),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["accepted"]
    terminal = runner.invoke(app, ["verify", str(root), "--trust-profile", str(policy)])
    assert terminal.exit_code == 0, terminal.output
    assert "Evidence set recipient verification complete" in terminal.output
    report = runner.invoke(app, ["report", str(root), "--json"])
    assert report.exit_code == 0, report.output
    assert json.loads(report.output)["kind"] == "evidence_set"


def test_cli_missing_profile_and_legacy_override_are_rejected(tmp_path):
    root, policy = fixture(tmp_path)
    runner = CliRunner()
    missing = runner.invoke(app, ["verify", str(root), "--json"])
    assert (
        missing.exit_code == 2 and json.loads(missing.output)["kind"] == "evidence_set"
    )
    conflict = runner.invoke(
        app,
        [
            "verify",
            str(root),
            "--trust-profile",
            str(policy),
            "--expected-signer",
            "sha256:" + "0" * 64,
        ],
    )
    assert conflict.exit_code == 2 and "belong in recipient profiles" in conflict.output


def test_cli_nonpassing_required_component_returns_seven(tmp_path):
    root, policy = fixture(tmp_path, incomplete=True)
    result = CliRunner().invoke(
        app, ["verify", str(root), "--trust-profile", str(policy), "--json"]
    )
    assert result.exit_code == 7, result.output
    assert not json.loads(result.output)["accepted"]


def test_report_rejects_different_original_answers(tmp_path):
    def mutate(baseline, subject):
        subject["records"][0]["output"] = "different"

    root, _ = fixture(tmp_path, captured_change=mutate)
    with pytest.raises(EvidenceReportError, match="runs or case sets differ"):
        render_evidence(root)


def test_report_rejects_advisory_component(tmp_path):
    root, _ = fixture(tmp_path, role="advisory")
    with pytest.raises(EvidenceReportError, match="required judge policy"):
        render_evidence(root)


def test_report_output_collisions_and_symlink_parents(tmp_path):
    root, _ = fixture(tmp_path)
    with pytest.raises(EvidenceReportError, match="collide"):
        render_evidence(
            root, html_path=tmp_path / "same", markdown_path=tmp_path / "same"
        )
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(EvidenceReportError, match="real directory"):
        render_evidence(root, html_path=link / "report.html")


def test_incomplete_judge_junit_is_error(tmp_path):
    root, _ = fixture(tmp_path, incomplete=True)
    output = tmp_path / "result.xml"
    render_evidence(root, junit_path=output)
    assert len(ElementTree.parse(output).getroot().findall("testcase/error")) == 1


def test_cli_receipt_write_failure_stays_structured(tmp_path):
    root, policy = fixture(tmp_path)
    output = tmp_path / "occupied"
    output.write_text("preserve")
    result = CliRunner().invoke(
        app,
        [
            "verify",
            str(root),
            "--trust-profile",
            str(policy),
            "--receipt",
            str(output),
            "--json",
        ],
    )
    assert result.exit_code == 2 and json.loads(result.output)["kind"] == "evidence_set"
    assert output.read_text() == "preserve"
