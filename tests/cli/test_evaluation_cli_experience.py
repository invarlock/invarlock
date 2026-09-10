"""Captured evaluation CLI journeys use neutral evaluate, verify, and report commands."""

import hashlib
import json
import shlex
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from invarlock import (
    captured_contracts,
    captured_evaluation,
    captured_reporting,
    captured_verification,
    evaluation_transaction,
    evidence_reporting,
)
from invarlock.cli import app
from invarlock.core import scoring
from invarlock.core.registry import CoreRegistry
from invarlock.evaluation_comparison import comparison
from invarlock.evidence_pack_contract import canonical_json_bytes
from tests._evaluation_support import captured_request_digest, digest


def _keygen(runner: CliRunner, directory: Path) -> tuple[Path, str]:
    result = runner.invoke(app, ["evaluate", "--keygen", str(directory), "--json"])
    assert result.exit_code == 0, result.output
    details = json.loads(result.stdout)["details"]
    return directory / "private.pem", details["public_key_fingerprint"]


def _inputs(root: Path) -> tuple[dict, dict, dict]:
    return tuple(
        json.loads((root / path).read_text())
        for path in ("inputs/baseline.json", "inputs/subject.json", "policy.json")
    )


def _inventory(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _forbid_report_work(monkeypatch):
    forbidden = Mock(
        side_effect=AssertionError("report must not score, execute or verify")
    )
    for module, name in (
        (captured_evaluation, "evaluate_captured_request"),
        (captured_evaluation, "compare_runs"),
        (captured_verification, "verify_captured_evidence"),
        (captured_verification, "verify_captured_receipt"),
        (captured_verification, "compare_runs"),
        (comparison, "compare_runs"),
        (comparison, "score"),
        (comparison, "_interval"),
        (scoring, "score"),
        (evaluation_transaction, "evaluate_request_file"),
        (CoreRegistry, "get_runtime_provider"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    return forbidden


def _forbid_presentation(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("invalid input reached presentation"))
    for module, names in (
        (captured_reporting, ("_view",)),
        (captured_contracts, ("atomic_write",)),
        (
            evidence_reporting,
            (
                "_report_view",
                "render_report_html",
                "render_report_markdown",
                "_write_html_no_clobber",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    return forbidden


@pytest.fixture
def project(tmp_path: Path) -> tuple[CliRunner, Path]:
    runner = CliRunner()
    root = tmp_path / "project"
    initialized = runner.invoke(
        app, ["evaluate", "--init", str(root), "--example", "classification"]
    )
    assert initialized.exit_code == 0, initialized.output
    return runner, root


def evaluate(project: tuple[CliRunner, Path], name: str, *options: str):
    runner, root = project
    request_path = root / "request.yaml"
    request = json.loads(request_path.read_text())
    request["output"]["evidence"] = name
    request_path.chmod(0o644)
    request_path.write_bytes(canonical_json_bytes(request))
    return runner.invoke(app, ["evaluate", str(request_path), *options])


def _verify_args(
    root: Path,
    evidence: str,
    signer: str,
    verifier: Path,
    baseline: dict,
    subject: dict,
    policy: dict,
    receipt: Path,
) -> list[str]:
    return [
        "verify",
        str(root / evidence),
        "--policy",
        str(root / "policy.json"),
        "--expected-baseline-run",
        digest(baseline),
        "--expected-subject-run",
        digest(subject),
        "--expected-request-digest",
        captured_request_digest(baseline, subject, policy),
        "--expected-signer",
        signer,
        "--receipt",
        str(receipt),
        "--verifier-signing-key",
        str(verifier),
        "--verifier-identity",
        "cli-test",
        "--json",
    ]


def test_human_evaluate_preserves_json_result_and_evidence_bytes(project):
    first = evaluate(project, "json", "--unsigned", "--json")
    second = evaluate(project, "human", "--unsigned")
    assert first.exit_code == second.exit_code == 0
    payload = json.loads(first.stdout)
    assert set(payload) == {
        "format_version",
        "kind",
        "ok",
        "evidence",
        "comparison_id",
        "baseline_run_digest",
        "subject_run_digest",
        "policy_digest",
        "authentication",
        "policy_verdict",
        "decision",
        "pack_manifest_digest",
        "request_digest",
    }
    assert payload["format_version"] == "invarlock/evaluation-result-v2"
    assert payload["kind"] == "captured"
    assert payload["decision"] == payload["policy_verdict"] == "pass"
    assert payload["request_digest"] == captured_request_digest(*_inputs(project[1]))
    assert "Captured evidence created" in second.stdout
    assert "Recorded policy result: pass" in second.stdout
    assert "Signing: Unsigned local evidence" in second.stdout
    assert "Independent verification: not performed" in second.stdout
    assert "Recipient verification" not in second.stdout
    assert json.loads(first.stdout)["authentication"] == "unsigned_local"
    assert first.stdout.endswith("\n")
    inventory = _inventory(project[1] / "json")
    assert payload["pack_manifest_digest"] == (
        "sha256:" + hashlib.sha256(inventory["manifest.json"]).hexdigest()
    )
    assert set(inventory) == {
        "manifest.json",
        "checksums.sha256",
        "request.json",
        "inputs/policy.json",
        "records/baseline.json",
        "records/subject.json",
        "reports/evaluation.report.json",
    }
    assert inventory == _inventory(project[1] / "human")


@pytest.mark.parametrize(
    "missing,expected", [(False, "regression"), (True, "insufficient_evidence")]
)
def test_human_adverse_decisions_preserve_recorded_verdict(project, missing, expected):
    _, root = project
    path = root / "inputs/subject.json"
    value = json.loads(path.read_text())
    for row in value["records"]:
        row["output"] = "wrong"
    if missing:
        value["records"][0]["output"] = None
        value["records"][0]["error"] = "capture incomplete"
    path.chmod(0o644)
    path.write_bytes(canonical_json_bytes(value))
    human = evaluate(project, "human", "--unsigned")
    machine = evaluate(project, "json", "--unsigned", "--json")
    assert human.exit_code == machine.exit_code == 0, human.output + machine.output
    assert json.loads(machine.stdout)["policy_verdict"] == expected
    assert f"Recorded policy result: {expected}" in human.stdout
    assert f"{39 if missing else 40} usable pairs" in human.stdout
    assert f"{1 if missing else 0} missing results" in human.stdout
    assert (
        "missing results"
        if missing
        else "lower interval bound exceeds allowed regression"
    ) in human.stdout
    assert "Signing: Unsigned local evidence" in human.stdout
    assert "Independent verification: not performed" in human.stdout
    assert _inventory(root / "human") == _inventory(root / "json")


@pytest.mark.parametrize(
    "missing", [True, False], ids=["missing-file", "malformed-yaml"]
)
def test_evaluate_bad_request_is_a_machine_readable_failure(
    tmp_path, monkeypatch, missing
):
    runner = CliRunner()
    path = tmp_path / "absent[red].yaml"
    if not missing:
        path.write_text("not: [valid")
    original = _inventory(tmp_path)
    forbidden = _forbid_report_work(monkeypatch)
    human = runner.invoke(app, ["evaluate", str(path), "--unsigned"])
    machine = runner.invoke(app, ["evaluate", str(path), "--unsigned", "--json"])
    assert human.exit_code == machine.exit_code == 2
    forbidden.assert_not_called()
    assert "FAIL" in human.output
    assert json.loads(machine.stdout)["errors"]
    if missing:
        assert "absent[red].yaml" in human.output
        assert not path.exists()
    assert _inventory(tmp_path) == original


def test_invalid_evaluate_option_is_usage_error_before_publication(project):
    result = evaluate(project, "result", "--unsigned", "--output-format", "yaml")
    assert result.exit_code == 2
    assert "No such option" in result.output
    assert not (project[1] / "result").exists()


def test_human_evaluate_error_does_not_interpret_brackets(project, monkeypatch):
    def rejected(*args, **kwargs):
        raise captured_evaluation.CapturedEvaluationError(
            "expected signer [red]required-key[/red]"
        )

    monkeypatch.setattr(captured_evaluation, "evaluate_captured_request", rejected)
    result = evaluate(project, "output", "--unsigned")
    assert result.exit_code == 2
    assert "[red]required-key[/red]" in result.stdout
    assert not (project[1] / "output").exists()


def test_human_verify_uses_recipient_inputs_and_rejects_wrong_signer(project, tmp_path):
    runner, root = project
    signer_path, signer = _keygen(runner, tmp_path / "signer")
    assert evaluate(project, "signed", "--signing-key", str(signer_path)).exit_code == 0
    verifier_path, _ = _keygen(runner, tmp_path / "verifier")
    baseline, subject, policy = _inputs(root)
    args = _verify_args(
        root,
        "signed",
        signer,
        verifier_path,
        baseline,
        subject,
        policy,
        tmp_path / "receipt.json",
    )
    machine = runner.invoke(app, args)
    human_args = list(args)
    human_args[human_args.index("--receipt") + 1] = str(tmp_path / "human-receipt.json")
    human = runner.invoke(
        app,
        [arg for arg in human_args if arg != "--json"],
    )
    assert machine.exit_code == 0, machine.output
    assert json.loads(machine.stdout)["ok"] is True
    assert human.exit_code == 0, human.output
    assert "PASS Independent captured verification complete" in human.stdout
    _, wrong_signer = _keygen(runner, tmp_path / "wrong-signer")
    wrong = list(args)
    wrong[wrong.index("--expected-signer") + 1] = wrong_signer
    wrong[wrong.index("--receipt") + 1] = str(tmp_path / "wrong-receipt.json")
    rejected = runner.invoke(app, wrong)
    assert rejected.exit_code == 6
    assert "signer" in rejected.stdout.lower()


@pytest.mark.parametrize("signed", [False, True], ids=["unsigned", "signed"])
def test_report_regenerates_without_scoring_or_inferred_verification(
    project, tmp_path, monkeypatch, signed
):
    runner, root = project
    options = ["--unsigned"]
    if signed:
        key, _ = _keygen(runner, tmp_path / "signer")
        options = ["--signing-key", str(key)]
    assert evaluate(project, "original", *options).exit_code == 0
    source = root / "original"
    original = _inventory(source)
    forbidden = _forbid_report_work(monkeypatch)
    result = runner.invoke(
        app,
        [
            "report",
            str(source),
            "--html",
            str(tmp_path / "view.html"),
            "--markdown",
            str(tmp_path / "view.md"),
            "--junit",
            str(tmp_path / "view.xml"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "format_version": "invarlock/evidence-report-v2",
        "kind": "captured",
        "ok": True,
        "pack_manifest_digest": "sha256:"
        + hashlib.sha256(original["manifest.json"]).hexdigest(),
        "requested_outputs": {
            "html": str(tmp_path / "view.html"),
            "markdown": str(tmp_path / "view.md"),
            "junit": str(tmp_path / "view.xml"),
        },
        "written_outputs": {
            "html": str(tmp_path / "view.html"),
            "markdown": str(tmp_path / "view.md"),
            "junit": str(tmp_path / "view.xml"),
        },
        "failed_output": None,
        "errors": [],
    }
    assert _inventory(source) == original
    assert (tmp_path / "view.html").is_file()
    assert (tmp_path / "view.md").is_file()
    assert (tmp_path / "view.xml").is_file()
    for path in (tmp_path / "view.html", tmp_path / "view.md"):
        text = path.read_text()
        assert (
            "Signed manifest verified." in text
            if signed
            else "Unsigned local evidence" in text
        )
        assert "Independent acceptance" in text
        assert "Not performed by report." in text
        assert "Scoring and replay were not performed by report." in text
    published = _inventory(tmp_path)
    again = runner.invoke(
        app, ["report", str(source), "--html", str(tmp_path / "view.html")]
    )
    assert again.exit_code == 2
    assert _inventory(tmp_path) == published
    assert _inventory(source) == original
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "mutation", ["binding", "schema", "independently_verified", "oversized"]
)
def test_report_rejects_invalid_evidence_before_rendering(
    project, tmp_path, monkeypatch, mutation
):
    runner, root = project
    assert evaluate(project, "original", "--unsigned").exit_code == 0
    source = root / "original"
    if mutation == "binding":
        policy = json.loads((source / "inputs/policy.json").read_text())
        policy["metrics"][0]["maximum_regression"] += 0.01
        policy_path = source / "inputs/policy.json"
        policy_path.chmod(0o644)
        policy_path.write_bytes(canonical_json_bytes(policy))
    elif mutation in {"schema", "independently_verified"}:
        manifest = json.loads((source / "manifest.json").read_text())
        if mutation == "schema":
            manifest["kind"] = "not-captured"
        else:
            manifest["independently_verified"] = True
        manifest_path = source / "manifest.json"
        manifest_path.chmod(0o644)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
    else:
        monkeypatch.setattr(captured_contracts, "CONTROL_LIMIT", 32)

    forbidden = _forbid_presentation(monkeypatch)
    original = _inventory(source)
    destination = tmp_path / "unpublished"
    result = runner.invoke(
        app,
        [
            "report",
            str(source),
            "--html",
            str(destination / "view.html"),
            "--markdown",
            str(destination / "view.md"),
            "--junit",
            str(destination / "view.xml"),
            "--json",
        ],
    )
    assert result.exit_code == 2, result.output
    forbidden.assert_not_called()
    assert json.loads(result.stdout)["ok"] is False
    assert json.loads(result.stdout)["errors"]
    assert not destination.exists()
    assert _inventory(source) == original


def test_signed_evaluate_keeps_metadata_and_entire_evidence_unchanged(
    project, tmp_path
):
    runner, root = project
    signer_path, _ = _keygen(runner, tmp_path / "signer")
    evaluations = []
    for name, options in (("default", []), ("json", ["--json"])):
        result = evaluate(project, name, "--signing-key", str(signer_path), *options)
        assert result.exit_code == 0, result.output
        evaluations.append(result.stdout)
    assert "Signing: Signed evidence" in evaluations[0]
    assert "Independent verification: not performed" in evaluations[0]
    evaluation = json.loads(evaluations[1])
    assert evaluation["authentication"] == "signed"
    original = _inventory(root / "default")
    assert "manifest.signature.json" in original
    assert original == _inventory(root / "json")
    manifest = json.loads(original["manifest.json"])
    assert manifest["comparison_id"] == evaluation["comparison_id"]
    assert manifest["files"]["baseline"]["digest"] == evaluation["baseline_run_digest"]
    assert manifest["files"]["subject"]["digest"] == evaluation["subject_run_digest"]
    assert manifest["files"]["policy"]["digest"] == evaluation["policy_digest"]


@pytest.mark.parametrize("signed", [False, True], ids=["unsigned", "signed"])
def test_explain_keeps_json_metadata_and_entire_evidence_unchanged(
    project, tmp_path, monkeypatch, signed
):
    runner, root = project
    options = ["--unsigned"]
    if signed:
        signer_path, _ = _keygen(runner, tmp_path / "signer")
        options = ["--signing-key", str(signer_path)]
    evaluated = evaluate(project, "original", *options, "--json")
    assert evaluated.exit_code == 0, evaluated.output
    original = _inventory(root / "original")
    manifest = json.loads(original["manifest.json"])
    forbidden = _forbid_report_work(monkeypatch)
    metadata = []
    for explain in (False, True):
        for machine in (False, True):
            options = (["--explain"] if explain else []) + (
                ["--json"] if machine else []
            )
            rendered = runner.invoke(app, ["report", str(root / "original"), *options])
            assert rendered.exit_code == 0, rendered.output
            if machine:
                metadata.append(json.loads(rendered.stdout))
            else:
                text = " ".join(rendered.stdout.split())
                assert "Policy satisfied" in text
                if signed:
                    assert "Signed manifest verified." in text
                    assert manifest["signing_key_fingerprint"] in text
                else:
                    assert "Unsigned local evidence; no signer authentication." in text
                assert "Independent acceptance: Not performed by report." in text
                assert "Replay and scoring: Not performed by report." in text
                assert manifest["comparison_id"] in text
                assert ("Exact comparison data" in text) is explain
            assert _inventory(root / "original") == original
    assert (
        metadata[0]
        == metadata[1]
        == {
            "format_version": "invarlock/evidence-report-v2",
            "kind": "captured",
            "ok": True,
            "pack_manifest_digest": "sha256:"
            + hashlib.sha256(original["manifest.json"]).hexdigest(),
            "requested_outputs": {},
            "written_outputs": {},
            "failed_output": None,
            "errors": [],
        }
    )
    forbidden.assert_not_called()


@pytest.mark.parametrize("explain", [False, True], ids=["default", "explain"])
@pytest.mark.parametrize(
    "missing", [False, True], ids=["regression", "missing-results"]
)
def test_report_signed_manifest_presents_recorded_adverse_result_without_replay(
    project, tmp_path, monkeypatch, explain, missing
):
    runner, root = project
    subject_path = root / "inputs/subject.json"
    subject = json.loads(subject_path.read_text())
    for row in subject["records"]:
        row["output"] = "wrong"
    if missing:
        subject["records"][0]["output"] = None
        subject["records"][0]["error"] = "capture incomplete"
    subject_path.chmod(0o644)
    subject_path.write_bytes(canonical_json_bytes(subject))
    signer_path, _ = _keygen(runner, tmp_path / "signer")
    evaluated = evaluate(
        project, "signed-regression", "--signing-key", str(signer_path), "--json"
    )
    assert evaluated.exit_code == 0
    source = root / "signed-regression"
    original = _inventory(source)
    forbidden = _forbid_report_work(monkeypatch)
    report = runner.invoke(app, ["report", str(root / "signed-regression"), "--json"])
    assert report.exit_code == 0, report.output
    assert json.loads(report.stdout)["ok"] is True
    human = runner.invoke(
        app, ["report", str(source), *(["--explain"] if explain else [])]
    )
    assert human.exit_code == 0, human.output
    text = " ".join(human.stdout.split())
    assert ("More evidence needed" if missing else "Policy not met") in text
    if missing:
        assert "Unavailable" in text
    assert "Signed manifest verified." in text
    assert "Independent acceptance: Not performed by report." in text
    assert "Replay and scoring: Not performed by report." in text
    assert (
        "Recorded reasons: missing results"
        if missing
        else "lower interval bound exceeds allowed regression"
    ) in text
    assert f"{39 if missing else 40} usable pairs" in text
    assert f"{1 if missing else 0} missing results" in text
    assert "Independent verification: passed" not in text
    assert _inventory(source) == original
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "missing,expected", [(False, "regression"), (True, "insufficient_evidence")]
)
def test_authenticated_adverse_result_keeps_policy_verification_status(
    project, tmp_path, missing, expected
):
    runner, root = project
    path = root / "inputs/subject.json"
    subject = json.loads(path.read_text())
    for row in subject["records"]:
        row["output"] = "wrong"
    if missing:
        subject["records"][0]["output"] = None
        subject["records"][0]["error"] = "capture incomplete"
    path.chmod(0o644)
    path.write_bytes(canonical_json_bytes(subject))
    signer_path, signer = _keygen(runner, tmp_path / "signer")
    evaluated = evaluate(project, "signed", "--signing-key", str(signer_path), "--json")
    assert evaluated.exit_code == 0
    baseline, subject, policy = _inputs(root)
    verifier_path, _ = _keygen(runner, tmp_path / "verifier")
    args = _verify_args(
        root,
        "signed",
        signer,
        verifier_path,
        baseline,
        subject,
        policy,
        tmp_path / "receipt.json",
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 7
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    assert payload["decision"] == expected
    human_args = [arg for arg in args if arg != "--json"]
    human_args[human_args.index("--receipt") + 1] = str(tmp_path / "human-receipt.json")
    human = runner.invoke(app, human_args)
    assert human.exit_code == 7, human.output
    assert human.stdout.startswith(
        "FAIL captured evidence does not satisfy the approved policy\n"
    )
    assert "Evidence integrity: verified" in human.stdout
    assert "PASS Independent captured verification complete" not in human.stdout
    assert "Policy result: fail" in human.stdout
    assert f"Recorded decision: {expected}" in human.stdout
    assert f"{39 if missing else 40} usable pairs" in human.stdout


@pytest.mark.parametrize("value", [None, [], {}, {"comparison": None}])
def test_report_missing_structure_is_integration_error_before_presentation(
    tmp_path, monkeypatch, value
):
    source = tmp_path / "malformed"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(value))
    original = _inventory(source)
    forbidden = _forbid_presentation(monkeypatch)
    destination = tmp_path / "unpublished"
    result = CliRunner().invoke(
        app,
        [
            "report",
            str(source),
            "--html",
            str(destination / "report.html"),
            "--markdown",
            str(destination / "summary.md"),
            "--junit",
            str(destination / "junit.xml"),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["ok"] is False
    assert json.loads(result.stdout)["errors"]
    forbidden.assert_not_called()
    assert not destination.exists()
    assert _inventory(source) == original


def test_both_cli_publication_paths_build_one_evidence_view(
    project, tmp_path, monkeypatch
):
    runner, root = project
    original = captured_reporting._view
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(captured_reporting, "_view", counted)
    assert evaluate(project, "first", "--unsigned").exit_code == 0
    first = runner.invoke(
        app, ["report", str(root / "first"), "--html", str(tmp_path / "first.html")]
    )
    assert first.exit_code == 0, first.output
    assert calls == [1]
    calls.clear()
    second = runner.invoke(
        app, ["report", str(root / "first"), "--html", str(tmp_path / "second.html")]
    )
    assert second.exit_code == 0, second.output
    assert calls == [1]


def test_generated_starter_directory_runs_neutral_commands(tmp_path, monkeypatch):
    runner = CliRunner()
    project = tmp_path / "starter"
    initialized = runner.invoke(app, ["evaluate", "--init", str(project)])
    assert initialized.exit_code == 0
    readme = (project / "README.txt").read_text()
    assert "captured evaluation" in readme
    for expected in (
        "--trust-profile",
        "invarlock/trust-inputs-v2",
        "--receipt",
        "--fail-on-policy",
        "requested_outputs",
        "written_outputs",
        "failed_output",
        "Unsigned reports are local",
    ):
        assert expected in readme
    commands = [
        shlex.split(line)[1:]
        for line in readme.splitlines()
        if line.startswith("invarlock ")
    ]
    assert len(commands) == 2, (
        "generated README must include runnable evaluate and report commands"
    )
    assert [command[0] for command in commands] == ["evaluate", "report"]
    monkeypatch.chdir(project)
    evaluate_result = runner.invoke(app, commands[0])
    assert evaluate_result.exit_code == 0, evaluate_result.output
    assert json.loads(evaluate_result.stdout)["policy_verdict"] == "pass"
    original = _inventory(project / "artifacts/evidence")
    assert original
    forbidden = _forbid_report_work(monkeypatch)
    rendered = runner.invoke(app, commands[1])
    assert rendered.exit_code == 0, rendered.output
    text = " ".join(rendered.stdout.split())
    assert "Unsigned local evidence" in text
    assert "Independent acceptance: Not performed by report." in text
    assert "Replay and scoring: Not performed by report." in text
    assert (project / "artifacts/report.html").is_file()
    assert (project / "artifacts/summary.md").is_file()
    assert (project / "artifacts/junit.xml").is_file()
    assert _inventory(project / "artifacts/evidence") == original
    forbidden.assert_not_called()
