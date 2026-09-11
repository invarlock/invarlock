"""CLI option conflicts, typed dispatch, and truthful publication outcomes."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from rich.text import Text
from typer.testing import CliRunner

from invarlock import engine
from invarlock.cli.app import app
from tests.cli.test_cli_experience import imported_example as imported_example
from tests.cli.test_trust_profile_cli import _case
from tests.core.test_captured_sdk_omissions import (
    _bytes,
    _digest,
    _inputs,
    _key,
    _profile,
)
from tests.core.test_evaluation_request_contract import _request_payload, _valid_request

RUNNER = CliRunner()


@pytest.mark.parametrize("json_out", [False, True])
@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--init", "new", "--keygen", "keys"], "mutually exclusive"),
        (
            ["--init", "new", "--case-set-output", "cases.json"],
            "--init cannot be combined",
        ),
        (
            ["--keygen", "keys", "--example", "judge"],
            "--keygen accepts no setup options",
        ),
        (
            ["--keygen", "keys", "--case-set-output", "cases.json"],
            "--keygen accepts no setup options",
        ),
        (
            ["--freeze-cases", "cases.json", "--example", "judge"],
            "--freeze-cases cannot use --example",
        ),
        ([], "a request or exactly one setup action is required"),
        (["request.yaml", "--example", "judge"], "require a setup action"),
        (["request.yaml", "--case-set-output", "cases.json"], "require a setup action"),
    ],
)
def test_setup_conflicts_are_closed_errors_without_writes(
    tmp_path, monkeypatch, arguments, message, json_out
):
    monkeypatch.chdir(tmp_path)
    result = RUNNER.invoke(
        app, ["evaluate", *arguments, *(["--json"] if json_out else [])]
    )
    assert result.exit_code == 2, result.output
    if json_out:
        payload = json.loads(result.stdout)
        assert payload == {
            "format_version": "invarlock/evaluation-setup-v1",
            "action": None,
            "ok": False,
            "details": None,
            "errors": [payload["errors"][0]],
        }
        assert message in payload["errors"][0]
    else:
        assert result.stdout.startswith("FAIL ")
        assert message in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_freeze_cases_rejects_nonobject_without_creating_output(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_bytes(b"[]")
    output = tmp_path / "frozen.json"
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            "--freeze-cases",
            str(cases),
            "--case-set-output",
            str(output),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["errors"] == ["planned case set must be an object"]
    assert not output.exists()


@pytest.mark.parametrize("action", ["--init", "--keygen"])
def test_setup_rejects_dangling_destination_symlink(tmp_path, action):
    target = tmp_path / "absent"
    destination = tmp_path / "new"
    destination.symlink_to(target, target_is_directory=True)
    result = RUNNER.invoke(app, ["evaluate", action, str(destination), "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["errors"] == [
        "setup directory must not already exist"
    ]
    assert destination.is_symlink()
    assert not target.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["new"]


@pytest.mark.parametrize("signed", [False, True])
def test_captured_preflight_text_identifies_authentication_without_publication(
    tmp_path, signed
):
    _inputs(tmp_path)
    _key(tmp_path / "key.pem")
    options = ["--signing-key", str(tmp_path / "key.pem")] if signed else ["--unsigned"]
    result = RUNNER.invoke(
        app, ["evaluate", str(tmp_path / "request.json"), "--preflight", *options]
    )
    assert result.exit_code == 0, result.output
    assert "Preflight complete" in result.stdout
    assert "Mode: captured; paired records: 4" in result.stdout
    assert (
        f"Requested authentication: {'signed' if signed else 'unsigned_local'}"
        in result.stdout
    )
    assert (
        "No execution, signing, scoring, or publication was performed" in result.stdout
    )
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    "options", [["--runtime-image", "unused"], ["--allow-installed-scorers"]]
)
def test_captured_evaluate_rejects_explicit_runtime_controls(
    tmp_path, monkeypatch, options
):
    _inputs(tmp_path)
    forbidden = Mock(side_effect=AssertionError("captured must not discover providers"))
    monkeypatch.setattr("invarlock.core.registry.CoreRegistry", forbidden)
    result = RUNNER.invoke(
        app,
        ["evaluate", str(tmp_path / "request.json"), "--unsigned", "--json", *options],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "format_version": "invarlock/evaluation-result-v2",
        "kind": "captured",
        "ok": False,
        "errors": ["runtime options are not valid for captured evaluation"],
    }
    forbidden.assert_not_called()
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--unsigned"],
        ["--max-bootstrap-draws", str(engine.DEFAULT_MAX_BOOTSTRAP_DRAWS)],
    ],
)
def test_native_evaluate_rejects_explicit_captured_controls(tmp_path, options):
    request = _valid_request(tmp_path)
    result = RUNNER.invoke(app, ["evaluate", str(request), "--json", *options])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["format_version"] == "invarlock/evaluation-result-v1"
    assert "applies only to captured evaluation" in payload["errors"][0]
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("discriminator_fails", [False, True])
def test_cli_discriminator_replacement_does_not_cross_runtime_boundary(
    tmp_path, monkeypatch, discriminator_fails
):
    from invarlock.core import evaluation_request

    path = _valid_request(tmp_path)
    native_bytes = path.read_bytes()
    _inputs(tmp_path)
    captured_bytes = (tmp_path / "request.json").read_bytes()
    original = evaluation_request.evaluation_request_mode

    def discriminate(path):
        if discriminator_fails:
            # A replaceable loader may succeed after a failed initial read.
            path.write_bytes(captured_bytes)
            raise engine.EvaluationRequestError("request replaced before loading")
        mode = original(path)
        path.write_bytes(captured_bytes)
        return mode

    assert path.read_bytes() == native_bytes
    monkeypatch.setattr(evaluation_request, "evaluation_request_mode", discriminate)
    options = ["--preflight", "--unsigned"] if discriminator_fails else []
    result = RUNNER.invoke(app, ["evaluate", str(path), "--json", *options])
    payload = json.loads(result.stdout)
    if discriminator_fails:
        assert result.exit_code == 0, result.output
        assert payload["kind"] == "captured"
    else:
        assert result.exit_code == 2
        assert "captured evaluation path" in payload["errors"][0]
    assert not (tmp_path / "artifacts").exists()


def test_cli_discriminator_replacement_does_not_cross_captured_boundary(
    tmp_path, monkeypatch
):
    from invarlock.core import evaluation_request

    native_bytes = _valid_request(tmp_path).read_bytes()
    _inputs(tmp_path)
    path = tmp_path / "request.json"
    original = evaluation_request.evaluation_request_mode

    def discriminate(request_path):
        mode = original(request_path)
        assert mode == "captured"
        request_path.write_bytes(native_bytes)
        return mode

    monkeypatch.setattr(evaluation_request, "evaluation_request_mode", discriminate)
    result = RUNNER.invoke(app, ["evaluate", str(path), "--unsigned", "--json"])
    assert result.exit_code == 2
    assert "did not load as captured evidence" in json.loads(result.stdout)["errors"][0]
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("initial", "replacement"), [("run", "import"), ("import", "run")]
)
def test_cli_discriminator_replacement_does_not_cross_native_execution_modes(
    tmp_path, monkeypatch, initial, replacement
):
    from invarlock.core import evaluation_request

    path = _valid_request(tmp_path, mode="import")
    initial_bytes = yaml.safe_dump(
        _request_payload(mode=initial), sort_keys=False
    ).encode()
    replacement_bytes = yaml.safe_dump(
        _request_payload(mode=replacement), sort_keys=False
    ).encode()
    path.write_bytes(initial_bytes)
    original = evaluation_request.evaluation_request_mode

    def discriminate(request_path):
        mode = original(request_path)
        assert mode == initial
        request_path.write_bytes(replacement_bytes)
        return mode

    monkeypatch.setattr(evaluation_request, "evaluation_request_mode", discriminate)
    result = RUNNER.invoke(app, ["evaluate", str(path), "--json"])
    assert result.exit_code == 2
    assert (
        "execution mode changed while loading" in json.loads(result.stdout)["errors"][0]
    )
    assert not (tmp_path / "artifacts").exists()


def test_failed_discriminator_does_not_reload_a_native_request(
    imported_example, monkeypatch
):
    from invarlock.core import evaluation_request

    root, _ = imported_example
    path = root / "request.yaml"
    original = evaluation_request.load_evaluation_request
    calls = 0

    def load_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        loaded = original(*args, **kwargs)
        path.write_text("replaced after the complete load", encoding="utf-8")
        return loaded

    monkeypatch.setattr(
        evaluation_request,
        "evaluation_request_mode",
        lambda _path: (_ for _ in ()).throw(
            engine.EvaluationRequestError("initial discriminator unavailable")
        ),
    )
    monkeypatch.setattr(evaluation_request, "load_evaluation_request", load_once)
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(path),
            "--preflight",
            "--signing-key",
            str(root / "evidence.pem"),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == 1
    assert json.loads(result.stdout)["ok"] is True


@pytest.mark.parametrize(
    "options",
    [
        ["--runtime-image", "registry.example/worker@sha256:" + "a" * 64],
        ["--runtime-device", "cpu"],
        ["--runtime-memory-mib", "4096"],
    ],
)
def test_import_request_rejects_explicit_run_controls(imported_example, options):
    root, _ = imported_example
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(root / "request.yaml"),
            "--signing-key",
            str(root / "evidence.pem"),
            "--json",
            *options,
        ],
    )
    assert result.exit_code == 2
    assert "applies only to run requests" in json.loads(result.stdout)["errors"][0]
    assert not (root / "artifacts").exists()


@pytest.mark.parametrize("verdict,exit_code", [("pass", 0), ("fail", 7), (None, 2)])
def test_native_policy_gate_preserves_published_json_and_missing_verdict_diagnostic(
    imported_example, monkeypatch, verdict, exit_code
):
    from invarlock import evaluation_transaction

    root, _ = imported_example
    original = evaluation_transaction.evaluate_request_file

    def evaluate(*args, **kwargs):
        from dataclasses import replace

        result = original(*args, **kwargs)
        assert isinstance(result, engine.EvaluationTransactionResult)
        # Older embedding publishers may not supply the optional policy verdict.
        return replace(result, policy_verdict=None)

    if verdict is None:
        monkeypatch.setattr(evaluation_transaction, "evaluate_request_file", evaluate)
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(
                root
                / ("rejected-request.yaml" if verdict == "fail" else "request.yaml")
            ),
            "--signing-key",
            str(root / "evidence.pem"),
            "--fail-on-policy",
            "--json",
        ],
    )
    assert result.exit_code == exit_code, result.output
    payload = json.loads(result.stdout)
    assert payload["format_version"] == "invarlock/evaluation-result-v1"
    assert payload["ok"] is True
    assert (Path(payload["evidence"]) / "manifest.json").is_file()
    report = json.loads(
        (Path(payload["evidence"]) / "reports/evaluation.report.json").read_bytes()
    )
    assert report["verdict"] == (verdict or "pass")
    assert ("policy outcome is unavailable" in result.stderr) is (verdict is None)


@pytest.mark.parametrize("captured", [False, True])
def test_verify_rejects_profile_for_opposite_evidence_kind(tmp_path, captured):
    native_root = tmp_path / "native"
    native_root.mkdir()
    native = _case(native_root)
    captured_root = tmp_path / "captured"
    captured_root.mkdir()
    request, baseline, subject, policy = _inputs(captured_root)
    signer = _key(captured_root / "key.pem")
    pack = engine.evaluate_request_file(
        captured_root / "request.json", signing_key_path=captured_root / "key.pem"
    ).evidence_path
    _profile(captured_root, request, baseline, subject, policy, signer)
    evidence = pack if captured else native["pack"]
    profile = native["profile"] if captured else captured_root / "trust.json"
    receipt = tmp_path / "receipt.json"
    result = RUNNER.invoke(
        app,
        [
            "verify",
            str(evidence),
            "--trust-profile",
            str(profile),
            "--receipt",
            str(receipt),
            "--json",
        ],
    )
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["errors"] == [
        "native trust profile requires native evidence"
        if captured
        else "captured trust profile requires captured evidence"
    ]
    assert (payload.get("kind") == "captured") is captured
    assert not receipt.exists()


@pytest.mark.parametrize("captured", [False, True])
def test_verify_rejects_flags_for_opposite_evidence_kind(tmp_path, captured):
    if captured:
        _inputs(tmp_path)
        evidence = engine.evaluate_request_file(
            tmp_path / "request.json", signing_key_path=None, unsigned=True
        ).evidence_path
        options = ["--expected-schedule", "sha256:" + "a" * 64]
    else:
        evidence = tmp_path / "native"
        evidence.mkdir()
        options = ["--max-bootstrap-draws", str(engine.DEFAULT_MAX_BOOTSTRAP_DRAWS)]
    result = RUNNER.invoke(app, ["verify", str(evidence), *options, "--json"])
    assert result.exit_code == 2
    assert "not valid for" in json.loads(result.stdout)["errors"][0]


def test_verify_malformed_manifest_returns_integrity_exit_code(tmp_path):
    (tmp_path / "manifest.json").write_bytes(b"{broken")
    result = RUNNER.invoke(app, ["verify", str(tmp_path), "--json"])
    assert result.exit_code == 4, result.output
    assert json.loads(result.stdout)["ok"] is False


def test_report_text_describes_partial_outputs_without_clobbering(
    tmp_path, monkeypatch
):
    from invarlock import captured_contracts

    _inputs(tmp_path)
    evidence = engine.evaluate_request_file(
        tmp_path / "request.json", signing_key_path=None, unsigned=True
    ).evidence_path
    html = tmp_path / "report.html"
    markdown = tmp_path / "report.md"
    original = captured_contracts.atomic_write

    def race_after_html(path, data):
        written = original(path, data)
        if path == html:
            markdown.write_bytes(b"owned by another writer")
        return written

    monkeypatch.setattr(captured_contracts, "atomic_write", race_after_html)
    result = RUNNER.invoke(
        app, ["report", str(evidence), "--html", str(html), "--markdown", str(markdown)]
    )
    assert result.exit_code == 2, result.output
    assert "Written html:" in result.stdout
    assert "Failed output: markdown" in result.stdout
    assert html.is_file()
    assert markdown.read_bytes() == b"owned by another writer"


def test_version_is_safe_without_distribution_or_package_version(monkeypatch):
    import invarlock

    module = importlib.import_module("invarlock.cli.app")
    monkeypatch.setattr(
        module, "version", Mock(side_effect=module.PackageNotFoundError)
    )
    monkeypatch.delattr(invarlock, "__version__")
    result = RUNNER.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "InvarLock unknown"


def test_setup_error_is_literal_in_human_output(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_bytes(b"[]")
    result = RUNNER.invoke(app, ["evaluate", "--freeze-cases", str(cases)])
    assert result.exit_code == 2
    assert result.stdout.strip() == "FAIL planned case set must be an object"


@pytest.mark.parametrize("color", [False, True], ids=["plain", "colored"])
def test_preflight_cannot_be_combined_with_policy_gate(tmp_path, monkeypatch, color):
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("FORCE_COLOR", "1")
    if color:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", "1")
    _inputs(tmp_path)
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(tmp_path / "request.json"),
            "--unsigned",
            "--preflight",
            "--fail-on-policy",
        ],
        color=color,
        terminal_width=80,
    )
    assert result.exit_code == 2
    assert (
        "--fail-on-policy cannot be used with --preflight"
        in Text.from_ansi(result.output).plain
    )
    assert not (tmp_path / "artifacts").exists()


def test_captured_policy_gate_publishes_real_failure_before_exiting(tmp_path):
    _, _, _, policy = _inputs(tmp_path)
    policy["metrics"][0]["subject_minimum"] = 2
    (tmp_path / "policy.json").write_bytes(_bytes(policy))
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(tmp_path / "request.json"),
            "--unsigned",
            "--fail-on-policy",
            "--json",
        ],
    )
    assert result.exit_code == 7, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = json.loads(
        (Path(payload["evidence"]) / "reports/evaluation.report.json").read_bytes()
    )
    assert report["decision"] == "regression"


def test_captured_metric_names_are_safe_in_terminal_output(tmp_path):
    _, _, _, policy = _inputs(tmp_path)
    policy["metrics"][0]["name"] = "safe\x9b2JFAKE_PASS"
    (tmp_path / "policy.json").write_bytes(_bytes(policy))
    result = RUNNER.invoke(
        app,
        ["evaluate", str(tmp_path / "request.json"), "--unsigned"],
    )
    assert result.exit_code == 0, result.output
    assert "\x9b" not in result.stdout
    assert "safe\\u009b2JFAKE_PASS" in result.stdout


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("json_out", [False, True])
def test_captured_pairing_errors_escape_controls_without_changing_json_values(
    tmp_path, preflight, json_out
):
    _, baseline, subject, _ = _inputs(tmp_path)
    subject = json.loads(_bytes(subject))
    record_id = "case\x9b2JFAKE_PASS"
    for side, run in (("baseline", baseline), ("subject", subject)):
        run["records"][0]["id"] = record_id
        if side == "subject":
            run["records"][0]["input"] = "different input"
        (tmp_path / f"{side}.json").write_bytes(_bytes(run))
    request_path = tmp_path / "request.json"
    request = json.loads(request_path.read_bytes())
    request["comparison"]["baseline"]["expected_run_digest"] = _digest(baseline)
    request["comparison"]["subject"]["expected_run_digest"] = _digest(subject)
    request_path.write_bytes(_bytes(request))
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(request_path),
            "--unsigned",
            *(["--preflight"] if preflight else []),
            *(["--json"] if json_out else []),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "\x9b" not in result.stdout
    if json_out:
        assert record_id in json.loads(result.stdout)["errors"][0]
    else:
        assert "case\\u009b2JFAKE_PASS" in result.stdout
        assert "input changed between" in " ".join(result.stdout.split())
    assert not (tmp_path / "artifacts").exists()


def test_verify_refuses_symlink_evidence_root(tmp_path):
    evidence = tmp_path / "real"
    evidence.mkdir()
    link = tmp_path / "link"
    link.symlink_to(evidence, target_is_directory=True)
    result = RUNNER.invoke(app, ["verify", str(link), "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["errors"] == ["evidence must be a real directory"]
