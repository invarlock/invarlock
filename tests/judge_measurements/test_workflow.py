from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path
from xml.etree import ElementTree

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from typer.testing import CliRunner

from invarlock.cli.app import app
from invarlock.core.evaluation_request import evaluation_request_mode
from invarlock.judge_measurements.workflow import (
    JudgeWorkflowError,
    load_judge_request,
    preflight_judge_request,
)
from tests.judge_measurements.test_evidence_acceptance import _publish

RUNNER = CliRunner()
FIXTURES = Path(__file__).parents[1] / "fixtures" / "judge_measurements"


@pytest.fixture
def staged(tmp_path):
    for name in (
        "plan",
        "measurements",
        "baseline_run",
        "subject_run",
        "analysis_policy",
    ):
        shutil.copyfile(FIXTURES / f"{name}.json", tmp_path / f"{name}.json")
    value = {
        "format_version": "invarlock/evaluation-request-v3",
        "execution": {"mode": "judge_import", "collection": None},
        "comparison": {
            "plan": "plan.json",
            "measurements": "measurements.json",
            "baseline_run": "baseline_run.json",
            "subject_run": "subject_run.json",
            "policy": "analysis_policy.json",
        },
        "output": {
            "evidence": "evidence",
            "signer_identity": "example-producer",
        },
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value))
    return path, value


def test_judge_request_rejects_unknown_execution_settings(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "format_version": "invarlock/evaluation-request-v3",
                "execution": {"mode": "judge_import", "network": True},
            }
        )
    )
    with pytest.raises(JudgeWorkflowError):
        load_judge_request(path)


@pytest.mark.parametrize(
    "reference",
    [
        "../plan.json",
        "/tmp/plan.json",
        "https://example.com/plan",
        "nested/../plan.json",
    ],
)
def test_request_paths_cannot_escape_approved_root(staged, reference):
    path, value = staged
    value["comparison"]["plan"] = reference
    path.write_text(json.dumps(value))
    with pytest.raises(JudgeWorkflowError):
        load_judge_request(path)


def test_preflight_has_counts_and_no_network_or_publication(staged, monkeypatch):
    path, _ = staged
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_a, **_k: pytest.fail("preflight network call"),
    )
    assert evaluation_request_mode(path) == "judge_import"
    result = preflight_judge_request(load_judge_request(path)).payload
    assert result["ready"]
    assert (
        result["cases"],
        result["independent_units"],
        result["planned_trials"],
        result["maximum_attempts"],
    ) == (1, 1, 2, 2)
    assert result["network_calls"] == 0
    assert result["judge"]["requested_model"] == "example-judge"
    assert not (path.parent / "evidence").exists()
    cli = RUNNER.invoke(app, ["evaluate", str(path), "--preflight", "--json"])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.stdout)["planned_trials"] == 2


def test_preflight_names_missing_inputs(staged):
    path, _ = staged
    (path.parent / "measurements.json").unlink()
    result = RUNNER.invoke(app, ["evaluate", str(path), "--preflight", "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["missing_inputs"] == ["measurements"]
    assert payload["cases"] == 1


def test_collect_preflight_shows_explicit_budgets_without_claiming_a_runner(staged):
    path, value = staged
    value["execution"] = {
        "mode": "judge_collect",
        "collection": {
            "integration": "inspect-judge",
            "configuration": "collection.json",
        },
    }
    value["comparison"]["measurements"] = None
    path.write_text(json.dumps(value))
    budget = {
        "grader": "example-judge",
        "inspect_version": "0.3.254",
        "profile": "inspect-text-frozen-answer-v1",
        "epochs": 1,
        "log_model_api": True,
        "log_samples": True,
        "sdk_max_retries": 0,
        "tools": False,
        "max_calls": 2,
        "max_input_tokens": 2000,
        "max_output_tokens": 256,
        "max_cost_microusd": 1000000,
        "input_tokens_per_call": 1000,
        "cost_microusd_per_call": 500000,
        "concurrency": 1,
        "requests_per_minute": 10,
        "request_timeout_seconds": 30,
    }
    (path.parent / "collection.json").write_text(json.dumps(budget))
    result = RUNNER.invoke(app, ["evaluate", str(path), "--preflight", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["budgets"] == {
        key: value
        for key, value in budget.items()
        if type(value) is int and key not in {"epochs", "sdk_max_retries"}
    }
    assert payload["budget_capacity"] == {
        "maximum_admitted_calls": 2,
        "planned_calls": 2,
        "full_plan_reserved": True,
    }
    assert payload["collection_available"] is False
    assert payload["ready"] and not payload["errors"]
    assert "inspect-judge collect API" in payload["next_action"]
    result = RUNNER.invoke(app, ["evaluate", str(path), "--unsigned", "--json"])
    assert result.exit_code == 2
    assert "does not execute provider calls" in result.stdout
    assert not (path.parent / "evidence").exists()

    budget["Authorization"] = "secret"
    (path.parent / "collection.json").write_text(json.dumps(budget))
    result = RUNNER.invoke(app, ["evaluate", str(path), "--preflight", "--json"])
    assert result.exit_code == 2
    assert "exactly the supported" in result.stdout


@pytest.mark.parametrize(
    "option", ["--allow-installed-scorers", "--max-bootstrap-draws"]
)
def test_judge_rejects_native_or_bootstrap_flags(staged, option):
    args = ["evaluate", str(staged[0]), "--preflight", option, "--json"]
    if option == "--max-bootstrap-draws":
        args.insert(-1, "100")
    result = RUNNER.invoke(app, args)
    assert result.exit_code == 2
    assert "do not apply" in result.stdout


def test_unsigned_public_workflow_reports_inconclusive_honestly(staged, tmp_path):
    path, _ = staged
    result = RUNNER.invoke(
        app, ["evaluate", str(path), "--unsigned", "--fail-on-policy", "--json"]
    )
    assert result.exit_code == 7, result.output
    payload = json.loads(result.stdout)
    assert payload["decision"] == "insufficient_evidence"
    assert payload["independent_verification"] == "not_performed"
    outputs = {
        name: tmp_path / f"report.{extension}"
        for name, extension in (("html", "html"), ("markdown", "md"), ("junit", "xml"))
    }
    args = ["report", str(tmp_path / "evidence"), "--json", "--explain"]
    for name, destination in outputs.items():
        args.extend([f"--{name}", str(destination)])
    report = RUNNER.invoke(app, args)
    assert report.exit_code == 0, report.output
    facts = json.loads(report.stdout)
    assert facts["assurance"]["authentication"] == "unsigned"
    assert facts["assurance"]["recipient_acceptance"] == "not_performed"
    assert facts["analysis"]["counts"]["scheduled_units"] == 1
    assert facts["comparison"]["baseline"]["artifact_digest"].startswith("sha256:")
    assert facts["comparison"]["subject"]["artifact_digest"].startswith("sha256:")
    assert facts["prompt"]["template_excerpt"].startswith("Grade the input")
    assert facts["assurance"]["signer_identity"] is None
    assert "example-judge" in outputs["html"].read_text()
    suite = ElementTree.fromstring(outputs["junit"].read_bytes())
    assert suite.get("errors") == "1" and suite.get("failures") == "0"


def test_publish_requires_explicit_authentication_choice(staged):
    result = RUNNER.invoke(app, ["evaluate", str(staged[0]), "--json"])
    assert result.exit_code == 2
    assert "--signing-key" in result.stdout and "--unsigned" in result.stdout


def test_signed_workflow_uses_request_declared_signer_identity(staged):
    path, _ = staged
    key = Ed25519PrivateKey.generate()
    key_path = path.parent / "producer-private.pem"
    key_path.write_bytes(
        key.private_bytes(
            Encoding.PEM,
            PrivateFormat.PKCS8,
            NoEncryption(),
        )
    )
    result = RUNNER.invoke(
        app, ["evaluate", str(path), "--signing-key", str(key_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["signer_identity"] == "example-producer"
    envelope = json.loads((path.parent / "evidence" / "envelope.json").read_text())
    assert envelope["signer"]["identity"] == "example-producer"


def test_request_rejects_symlinked_input_before_publication(staged, tmp_path):
    (tmp_path / "plan.json").unlink()
    (tmp_path / "plan.json").symlink_to(FIXTURES / "plan.json")
    result = RUNNER.invoke(app, ["evaluate", str(staged[0]), "--preflight", "--json"])
    assert result.exit_code == 2
    assert "symlink" in result.stdout


def test_signed_recipient_cli_replay_and_receipt_remain_offline(tmp_path, monkeypatch):
    publication, policy = _publish(tmp_path)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_a, **_k: pytest.fail("verification network call"),
    )
    receipt_path = tmp_path / "recipient-receipt.json"
    result = RUNNER.invoke(
        app,
        [
            "verify",
            str(publication.path),
            "--trust-profile",
            str(policy),
            "--receipt",
            str(receipt_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["authenticated"] and payload["replayed"] and payload["accepted"]
    assert (
        json.loads(receipt_path.read_text())["format"]
        == "invarlock/judge-measurement-verification-receipt-v1"
    )


@pytest.mark.parametrize(
    "args", [[], ["--policy", "old.json"], ["--verifier-identity", "native-signer"]]
)
def test_judge_verification_rejects_missing_or_legacy_trust(tmp_path, args):
    publication, policy = _publish(tmp_path)
    arguments = ["verify", str(publication.path), "--json"]
    if args:
        arguments.extend(["--trust-profile", str(policy)])
    result = RUNNER.invoke(app, arguments + args)
    assert result.exit_code == 2


def test_adverse_and_incomplete_verification_preserve_distinct_decisions(tmp_path):
    for name, kwargs, expected in (
        ("adverse", {"baseline": 1, "subject": 0}, "regression"),
        ("incomplete", {"incomplete": True}, "insufficient_evidence"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        publication, policy = _publish(directory, **kwargs)
        result = RUNNER.invoke(
            app,
            ["verify", str(publication.path), "--trust-profile", str(policy), "--json"],
        )
        assert result.exit_code == 7, result.output
        payload = json.loads(result.stdout)
        assert payload["decision"] == expected
        assert (
            payload["authenticated"] and payload["replayed"] and not payload["accepted"]
        )


def test_report_escapes_metric_markup_and_rejects_in_pack_output(tmp_path):
    metric = '<script>alert("x")</script>'
    publication, _ = _publish(tmp_path, policy_changes={"metric_name": metric})
    output = tmp_path / "report.html"
    result = RUNNER.invoke(
        app, ["report", str(publication.path), "--html", str(output), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert metric not in output.read_text()
    assert "&lt;script&gt;" in output.read_text()
    blocked = RUNNER.invoke(
        app,
        [
            "report",
            str(publication.path),
            "--html",
            str(publication.path / "report.html"),
            "--json",
        ],
    )
    assert blocked.exit_code == 2
    assert not (publication.path / "report.html").exists()


def test_committed_judge_example_preflights_and_renders(tmp_path):
    example = Path(__file__).parents[2] / "examples" / "judge-measurements"
    shutil.copytree(example, tmp_path / "example")
    request = tmp_path / "example" / "request.yaml"
    result = RUNNER.invoke(app, ["evaluate", str(request), "--preflight", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["planned_trials"] == 2
    result = RUNNER.invoke(app, ["evaluate", str(request), "--unsigned", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["decision"] == "insufficient_evidence"
    result = RUNNER.invoke(app, ["report", str(request.parent / "evidence"), "--json"])
    assert result.exit_code == 0, result.output
    assert (
        json.loads(result.stdout)["assurance"]["recipient_acceptance"]
        == "not_performed"
    )
