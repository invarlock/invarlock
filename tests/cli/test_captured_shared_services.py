"""Common CLI services, trust snapshots, overrides, and private setup publication."""

import importlib
import json
import stat
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from invarlock import engine
from invarlock.cli.app import app
from tests.core.test_captured_sdk_omissions import _bytes, _inputs, _key, _profile

RUNNER = CliRunner()


@pytest.mark.parametrize("width", [80, 120])
def test_help_puts_the_common_journey_before_setup_and_runtime(width, monkeypatch):
    monkeypatch.setenv("COLUMNS", str(width))
    result = RUNNER.invoke(app, ["evaluate", "--help"], terminal_width=width)
    assert result.exit_code == 0, result.output
    text = result.stdout
    assert "Signed handoff:" in text
    assert "Unsigned local use:" in text
    setup = text.index("Evaluation setup")
    assert text.index("Signed handoff:") < setup
    assert text.index("Unsigned local use:") < setup
    assert text.index("Output and workflow") < setup
    assert text.index("Signing and execution authorization") < setup
    assert setup < text.index("Runtime resources (advanced)")


def test_unsigned_preflight_ignores_inherited_key_and_runtime_controls(
    tmp_path, monkeypatch
):
    _inputs(tmp_path)
    monkeypatch.setenv("INVARLOCK_SIGNING_KEY", str(tmp_path / "nonexistent-key.pem"))
    monkeypatch.setenv("INVARLOCK_RUNTIME_IMAGE", "unused-image")
    monkeypatch.setenv("INVARLOCK_ALLOW_INSTALLED_SCORERS", "1")
    forbidden = Mock(side_effect=AssertionError("captured mode must not load runtime"))
    monkeypatch.setattr("invarlock.core.registry.CoreRegistry", forbidden)
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(tmp_path / "request.json"),
            "--preflight",
            "--unsigned",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["kind"] == "captured"
    assert payload["requested_authentication"] == "unsigned_local"
    assert not (tmp_path / "artifacts").exists()
    explicit = RUNNER.invoke(
        app,
        [
            "evaluate",
            str(tmp_path / "request.json"),
            "--preflight",
            "--unsigned",
            "--signing-key",
            str(tmp_path / "nonexistent-key.pem"),
            "--json",
        ],
    )
    assert explicit.exit_code == 2
    assert (
        json.loads(explicit.stdout)["format_version"]
        == "invarlock/evaluation-preflight-v3"
    )
    forbidden.assert_not_called()


def test_cli_overrides_apply_in_caller_directory_and_retain_pins(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    request, _, _, _ = _inputs(root)
    request["comparison"]["baseline"]["path"] = "missing.json"
    request["output"]["evidence"] = "occupied"
    (root / "occupied").mkdir()
    (root / "request.json").write_bytes(_bytes(request))
    monkeypatch.chdir(tmp_path)
    result = RUNNER.invoke(
        app,
        [
            "evaluate",
            "root/request.json",
            "--baseline-run",
            "root/baseline.json",
            "--output",
            "root/override",
            "--unsigned",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (root / "override/manifest.json").is_file()
    rejected = RUNNER.invoke(
        app,
        [
            "evaluate",
            "root/request.json",
            "--baseline-run",
            "root/subject.json",
            "--output",
            "root/other",
            "--unsigned",
            "--json",
        ],
    )
    assert rejected.exit_code == 2
    assert "expected pin" in rejected.stdout
    assert not (root / "other").exists()


def test_cli_captured_profile_passes_frozen_bytes_without_environment_mixing(
    tmp_path, monkeypatch
):
    from invarlock import evidence_verification, trust_inputs

    request, baseline, subject, policy = _inputs(tmp_path)
    signer = _key(tmp_path / "evidence-signer.pem")
    evidence = engine.evaluate_request_file(
        tmp_path / "request.json", signing_key_path=tmp_path / "evidence-signer.pem"
    ).evidence_path
    _profile(tmp_path, request, baseline, subject, policy, signer)
    loader = trust_inputs.load_trust_inputs

    def load(path):
        loaded = loader(path)
        (tmp_path / "policy.json").write_text("not the acquired policy")
        (tmp_path / "verifier.pem").write_text("not the acquired key")
        return loaded

    monkeypatch.setattr(trust_inputs, "load_trust_inputs", load)
    service = Mock(wraps=evidence_verification.verify_evidence)
    monkeypatch.setattr(evidence_verification, "verify_evidence", service)
    for name in (
        "POLICY",
        "EXPECTED_BASELINE_RUN",
        "EXPECTED_BASELINE_ARTIFACT",
        "EXPECTED_REQUEST_DIGEST",
        "VERIFIER_SIGNING_KEY",
    ):
        monkeypatch.setenv("INVARLOCK_" + name, "ignored-environment")
    result = RUNNER.invoke(
        app,
        [
            "verify",
            str(evidence),
            "--trust-profile",
            str(tmp_path / "trust.json"),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ok"] is True
    kwargs = service.call_args.kwargs
    assert isinstance(kwargs["policy_bytes"], bytes)
    assert isinstance(kwargs["verifier_signing_key_bytes"], bytes)
    assert "expected_baseline_artifact" not in kwargs
    assert "scorer_registry" not in kwargs
    conflict = RUNNER.invoke(
        app,
        [
            "verify",
            str(evidence),
            "--trust-profile",
            str(tmp_path / "trust.json"),
            "--policy",
            str(tmp_path / "policy.json"),
            "--json",
        ],
    )
    assert conflict.exit_code == 2
    assert "cannot be mixed" in conflict.stdout
    assert service.call_count == 1


@pytest.mark.parametrize("action", ["--init", "--keygen"])
def test_setup_failure_never_leaves_partial_public_tree(tmp_path, monkeypatch, action):
    module = importlib.import_module("invarlock.cli.app")
    writer = module._write_setup_file
    writes = 0

    def fail_second(path, raw, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected second write failure")
        writer(path, raw, **kwargs)

    monkeypatch.setattr(module, "_write_setup_file", fail_second)
    result = RUNNER.invoke(app, ["evaluate", action, str(tmp_path / "new"), "--json"])
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "new").exists()
    assert list(tmp_path.iterdir()) == []
    assert json.loads(result.stdout)["details"] is None


@pytest.mark.parametrize("action", ["--init", "--keygen"])
def test_setup_directories_are_private_atomic_and_no_clobber(tmp_path, action):
    destination = tmp_path / "new"
    result = RUNNER.invoke(app, ["evaluate", action, str(destination), "--json"])
    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
        for path in destination.rglob("*")
    )
    before = {
        p.relative_to(destination): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    }
    assert (
        RUNNER.invoke(app, ["evaluate", action, str(destination), "--json"]).exit_code
        == 2
    )
    assert before == {
        p.relative_to(destination): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("option", ["--baseline-run", "--subject-run", "--output"])
def test_setup_rejects_source_output_options_before_any_write(tmp_path, option):
    result = RUNNER.invoke(
        app, ["evaluate", "--init", str(tmp_path / "new"), option, "unused", "--json"]
    )
    assert result.exit_code == 2
    assert not (tmp_path / "new").exists()


def test_setup_rejects_symlink_parent_without_writes(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    result = RUNNER.invoke(
        app, ["evaluate", "--keygen", str(tmp_path / "link/new"), "--json"]
    )
    assert result.exit_code == 2
    assert list((tmp_path / "real").iterdir()) == []
