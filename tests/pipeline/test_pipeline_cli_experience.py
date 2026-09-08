"""Human views preserve pipeline decisions and never confer recipient authority."""

import json

import pytest
from typer.testing import CliRunner

from invarlock.evidence_pack_contract import canonical_json_bytes
from invarlock.pipeline import cli
from invarlock.pipeline import report as report_module
from invarlock.pipeline.contracts import digest


@pytest.fixture
def project(tmp_path):
    runner = CliRunner()
    root = tmp_path / "project"
    assert runner.invoke(cli.app, ["init", str(root)]).exit_code == 0
    return runner, root


def compare(project, destination, *options):
    runner, root = project
    return runner.invoke(
        cli.app,
        [
            "compare",
            str(root / "pipeline.json"),
            "--output",
            str(destination),
            *options,
        ],
    )


def test_human_compare_preserves_default_json_and_evidence_bytes(project, tmp_path):
    first = compare(project, tmp_path / "json")
    second = compare(
        project, tmp_path / "human", "--output-format", "human", "--explain"
    )
    assert first.exit_code == second.exit_code == 0
    assert set(json.loads(first.stdout)) == {
        "decision",
        "bindings",
        "authentication",
        "output",
        "exit_code",
    }
    assert first.stdout.endswith("\n")
    assert "Policy result: pass" in second.stdout
    assert "Unsigned local" in second.stdout
    assert "Independent verification: not performed" in second.stdout
    assert "40" in second.stdout and "report.html" in second.stdout
    assert (tmp_path / "json/evidence.json").read_bytes() == (
        tmp_path / "human/evidence.json"
    ).read_bytes()


@pytest.mark.parametrize("missing,expected", [(False, 1), (True, 3)])
def test_human_adverse_decisions_explain_counts_and_reasons(
    project, tmp_path, missing, expected
):
    _, root = project
    path = root / "candidate.json"
    value = json.loads(path.read_text())
    for row in value["records"]:
        row["output"] = "wrong"
    if missing:
        value["records"][0]["output"] = None
        value["records"][0]["error"] = "capture incomplete"
    path.write_text(json.dumps(value))
    result = compare(project, tmp_path / "result", "--output-format", "human")
    assert result.exit_code == expected, result.stdout
    assert "Policy result:" in result.stdout
    assert ("missing results" if missing else "regression") in result.stdout
    assert "usable" in result.stdout and "missing" in result.stdout
    assert "Independent verification: not performed" in result.stdout


def test_human_failure_is_literal_and_json_failure_stays_compatible(tmp_path):
    runner = CliRunner()
    path = tmp_path / "absent[red].json"
    args = ["compare", str(path), "--output", str(tmp_path / "result")]
    human = runner.invoke(cli.app, [*args, "--output-format", "human"])
    machine = runner.invoke(cli.app, args)
    assert human.exit_code == machine.exit_code == 2
    assert "Comparison unavailable" in human.stdout
    assert json.loads(machine.stdout)["status"] == "integration_error"
    assert not (tmp_path / "result").exists()


def test_invalid_output_format_is_usage_error_before_publication(project, tmp_path):
    result = compare(project, tmp_path / "result", "--output-format", "yaml")
    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert "human" in result.output and "json" in result.output
    assert not (tmp_path / "result").exists()


def test_human_verify_uses_recipient_inputs_and_rejects_wrong_key(project, tmp_path):
    runner, root = project
    keys = tmp_path / "keys"
    assert runner.invoke(cli.app, ["keygen", str(keys)]).exit_code == 0
    assert (
        compare(
            project, tmp_path / "signed", "--signing-key", str(keys / "private.pem")
        ).exit_code
        == 0
    )
    baseline = json.loads((root / "baseline.json").read_text())
    candidate = json.loads((root / "candidate.json").read_text())
    args = [
        "verify",
        str(tmp_path / "signed/evidence.json"),
        "--public-key",
        str(keys / "public.pem"),
        "--policy",
        str(root / "policy.json"),
        "--expected-baseline",
        digest(baseline),
        "--expected-candidate",
        digest(candidate),
    ]
    machine = runner.invoke(cli.app, args)
    human = runner.invoke(cli.app, [*args, "--output-format", "human", "--explain"])
    assert machine.exit_code == human.exit_code == 0
    assert json.loads(machine.stdout) == {
        "authenticated": True,
        "decision": "pass",
        "exit_code": 0,
    }
    assert "Independent verification: passed" in human.stdout
    assert "recipient" in human.stdout.lower()
    other = tmp_path / "other"
    assert runner.invoke(cli.app, ["keygen", str(other)]).exit_code == 0
    args[args.index("--public-key") + 1] = str(other / "public.pem")
    rejected = runner.invoke(cli.app, [*args, "--output-format", "human"])
    assert rejected.exit_code == 2
    assert "Independent verification: passed" not in rejected.stdout
    assert "signature is invalid" in rejected.stdout


def test_report_regenerates_without_scoring_or_inferred_verification(
    project, tmp_path, monkeypatch
):
    assert compare(project, tmp_path / "original").exit_code == 0
    source = tmp_path / "original/evidence.json"
    original = source.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("report must not score or verify using inferred trust")

    monkeypatch.setattr(cli, "create_evidence", forbidden)
    monkeypatch.setattr(cli, "verify_evidence", forbidden)
    result = project[0].invoke(
        cli.app,
        [
            "report",
            str(source),
            "--output",
            str(tmp_path / "views"),
            "--output-format",
            "human",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Independent verification: not performed" in result.stdout
    assert "Unsigned local" in result.stdout
    assert source.read_bytes() == original
    assert (tmp_path / "views/report.html").is_file()
    assert (tmp_path / "views/summary.md").is_file()
    assert not (tmp_path / "views/evidence.json").exists()
    again = project[0].invoke(
        cli.app, ["report", str(source), "--output", str(tmp_path / "views")]
    )
    assert again.exit_code == 2


@pytest.mark.parametrize("mutation", ["binding", "schema", "oversized"])
def test_report_rejects_invalid_evidence_before_rendering(
    project, tmp_path, monkeypatch, mutation
):
    assert compare(project, tmp_path / "original").exit_code == 0
    source = tmp_path / "original/evidence.json"
    value = json.loads(source.read_text())
    if mutation == "binding":
        value["policy"]["metrics"][0]["maximum_regression"] += 0.01
    elif mutation == "schema":
        value["independently_verified"] = True
    else:
        monkeypatch.setattr(cli, "MAX_EVIDENCE_BYTES", 32)
    source.write_bytes(canonical_json_bytes(value))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached renderer")

    monkeypatch.setattr(report_module, "render_report_html", forbidden)
    result = project[0].invoke(
        cli.app, ["report", str(source), "--output", str(tmp_path / "views")]
    )
    assert result.exit_code == 2, result.stdout
    assert not (tmp_path / "views").exists()


def test_explain_keeps_json_shape_and_signed_evidence_unchanged(project, tmp_path):
    runner, _ = project
    keys = tmp_path / "keys"
    assert runner.invoke(cli.app, ["keygen", str(keys)]).exit_code == 0
    outputs = []
    for name, extra in (
        ("default", []),
        ("explain", ["--explain"]),
        ("human", ["--output-format", "human"]),
    ):
        result = compare(
            project, tmp_path / name, "--signing-key", str(keys / "private.pem"), *extra
        )
        assert result.exit_code == 0, result.stdout
        outputs.append(result.stdout)
    assert set(json.loads(outputs[0])) == set(json.loads(outputs[1]))
    evidence = [
        (tmp_path / name / "evidence.json").read_bytes()
        for name in ("default", "explain", "human")
    ]
    assert evidence[0] == evidence[1] == evidence[2]


def test_report_signature_presence_never_authenticates_recorded_regression(
    project, tmp_path
):
    runner, root = project
    candidate = json.loads((root / "candidate.json").read_text())
    for row in candidate["records"]:
        row["output"] = "wrong"
    (root / "candidate.json").write_text(json.dumps(candidate))
    keys = tmp_path / "keys"
    assert runner.invoke(cli.app, ["keygen", str(keys)]).exit_code == 0
    assert (
        compare(
            project, tmp_path / "signed", "--signing-key", str(keys / "private.pem")
        ).exit_code
        == 1
    )
    source = tmp_path / "signed/evidence.json"
    value = json.loads(source.read_text())
    # Deliberately invalid signature still has valid structure: rendering must
    # not present it as authenticated or pretend to have replayed its arithmetic.
    value["signature"]["value"] = "A" * 86 + "=="
    source.write_bytes(canonical_json_bytes(value))
    result = runner.invoke(
        cli.app, ["report", str(source), "--output", str(tmp_path / "views")]
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout) == {
        "status": "rendered",
        "recorded_decision": "regression",
        "independent_verification": "not_performed",
        "signing": "signature_present_unverified",
        "output": str(tmp_path / "views"),
        "exit_code": 0,
    }
    human = runner.invoke(
        cli.app,
        [
            "report",
            str(source),
            "--output",
            str(tmp_path / "human"),
            "--output-format",
            "human",
            "--explain",
        ],
    )
    assert human.exit_code == 0
    assert "Recorded policy result: regression" in human.stdout
    assert "signature not independently verified" in human.stdout
    assert "not replayed" in human.stdout
    assert "Independent verification: passed" not in human.stdout


def test_human_error_does_not_interpret_brackets(project, tmp_path, monkeypatch):
    def rejected(*args, **kwargs):
        raise ValueError("expected signer [red]required-key[/red]")

    monkeypatch.setattr(cli, "read_json", rejected)
    result = compare(project, tmp_path / "output", "--output-format", "human")
    assert result.exit_code == 2
    assert "[red]required-key[/red]" in result.stdout


@pytest.mark.parametrize("missing,expected", [(False, 1), (True, 3)])
def test_authenticated_adverse_result_keeps_policy_exit(
    project, tmp_path, missing, expected
):
    runner, root = project
    path = root / "candidate.json"
    candidate = json.loads(path.read_text())
    for row in candidate["records"]:
        row["output"] = "wrong"
    if missing:
        candidate["records"][0]["output"] = None
        candidate["records"][0]["error"] = "capture incomplete"
    path.write_text(json.dumps(candidate))
    keys = tmp_path / "keys"
    assert runner.invoke(cli.app, ["keygen", str(keys)]).exit_code == 0
    assert (
        compare(
            project, tmp_path / "signed", "--signing-key", str(keys / "private.pem")
        ).exit_code
        == expected
    )
    baseline = json.loads((root / "baseline.json").read_text())
    result = runner.invoke(
        cli.app,
        [
            "verify",
            str(tmp_path / "signed/evidence.json"),
            "--public-key",
            str(keys / "public.pem"),
            "--policy",
            str(root / "policy.json"),
            "--expected-baseline",
            digest(baseline),
            "--expected-candidate",
            digest(candidate),
            "--output-format",
            "human",
        ],
    )
    assert result.exit_code == expected, result.stdout
    assert "Independent verification: passed" in result.stdout
    assert (
        f"Policy result: {'insufficient_evidence' if missing else 'regression'}"
        in result.stdout
    )
    assert "usable pairs" in result.stdout


@pytest.mark.parametrize("value", [None, [], {}, {"comparison": None}])
def test_report_missing_structure_is_integration_error_before_presentation(
    tmp_path, monkeypatch, value
):
    source = tmp_path / "malformed.json"
    source.write_text(json.dumps(value))

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed evidence reached presentation")

    monkeypatch.setattr(report_module, "render_report_html", forbidden)
    result = CliRunner().invoke(
        cli.app, ["report", str(source), "--output", str(tmp_path / "views")]
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["status"] == "integration_error"
    assert not (tmp_path / "views").exists()


def test_both_cli_publication_paths_build_one_evidence_view(
    project, tmp_path, monkeypatch
):
    original = report_module._view
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(report_module, "_view", counted)
    assert compare(project, tmp_path / "first").exit_code == 0
    assert calls == [1]
    calls.clear()
    result = project[0].invoke(
        cli.app,
        [
            "report",
            str(tmp_path / "first/evidence.json"),
            "--output",
            str(tmp_path / "second"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert calls == [1]
