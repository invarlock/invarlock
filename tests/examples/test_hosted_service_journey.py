"""Simulation capture import through signed CLI comparison and recipient replay."""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from invarlock.cli.app import app
from tests.examples.test_hosted_service_capture import (
    SCRIPT,
    collect_fixture,
    protocol,
    response,
)
from tests.examples.test_hosted_service_capture import (
    module as capture_module,
)


def module():
    spec = importlib.util.spec_from_file_location(
        "hosted_journey", SCRIPT.with_name("journey.py")
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def captures(tmp_path, monkeypatch, answer="yes"):
    helper = capture_module()
    import base64

    result = {
        "status": 200,
        "body_base64": base64.b64encode(helper.encoded(response())).decode(),
        "error": None,
    }
    monkeypatch.setattr(helper, "bounded_request", lambda payload: result)
    declaration = protocol(helper, "http://127.0.0.1:1/v1/chat/completions")
    baseline = collect_fixture(tmp_path, helper, declaration)
    result["body_base64"] = base64.b64encode(helper.encoded(response(answer))).decode()
    subject = collect_fixture(tmp_path, helper, declaration, "subject", "-subject")
    return {
        "protocol": baseline[0],
        "protocol_sha256": baseline[1],
        "baseline": baseline[2],
        "baseline_sha256": baseline[3],
        "subject": subject[2],
        "subject_sha256": subject[3],
    }


@pytest.mark.parametrize("answer,decision", [("yes", "pass"), ("no", "regression")])
def test_signed_journey_uses_independently_rebuilt_recipient_inputs(
    tmp_path, monkeypatch, answer, decision
):
    value = module()
    inputs = captures(tmp_path, monkeypatch, answer)
    calls = []
    runner = CliRunner()

    def command(cli, cwd, *arguments, expected=(0,)):
        calls.append((str(cli), str(cwd), arguments))
        old = Path.cwd()
        os.chdir(cwd)
        try:
            result = runner.invoke(app, [str(item) for item in arguments])
        finally:
            os.chdir(old)
        assert result.exit_code in expected, result.output
        return json.loads(result.stdout)

    def prepare_child(argv, **kwargs):
        # Simulates interpreter dispatch only; imports and all CLI operations are real.
        values = json.loads(kwargs["input"])
        anchors = value.prepare(**values)
        return subprocess.CompletedProcess(argv, 0, json.dumps(anchors).encode())

    monkeypatch.setattr(value, "require_wheel", lambda path: Path(path))
    monkeypatch.setattr(value, "command", command)
    monkeypatch.setattr(value.subprocess, "run", prepare_child)
    output = tmp_path / "journey"
    result = value.journey(
        operator_cli=tmp_path / "operator-env/bin/invarlock",
        recipient_cli=tmp_path / "recipient-env/bin/invarlock",
        output=output,
        **inputs,
    )
    assert result["evaluation"]["decision"] == decision
    assert result["verification"]["decision"] == decision
    assert result["verification"]["integrity_ok"] is True
    assert result["source_assurance"] == "captured_inputs"
    assert result["signer_fingerprint"] != result["verifier_fingerprint"]
    assert (output / "recipient/policy.json").read_bytes() == (
        output / "operator/policy.json"
    ).read_bytes()
    assert (output / "recipient/anchors.json").exists()
    assert not (output / "recipient/signer").exists()
    assert (output / "recipient/verification.receipt.json").exists()
    assert all(
        (output / f"recipient/report.{suffix}").read_bytes()
        for suffix in ("html", "md", "xml")
    )
    receipt = json.loads((output / "recipient/verification.receipt.json").read_bytes())
    assert receipt["statement"]["verification_scope"] == "captured_comparison"
    assert len(calls) == 5


def test_temporal_overlap_and_script_drift_rejected(tmp_path, monkeypatch):
    value = module()
    helper = capture_module()
    inputs = captures(tmp_path, monkeypatch)
    data = json.loads(inputs["subject"].read_bytes())
    baseline = json.loads(inputs["baseline"].read_bytes())
    overlap = datetime.fromisoformat(baseline["observation_window"]["ended_at"])
    data["observation_window"]["started_at"] = (
        (overlap - timedelta(microseconds=1)).isoformat().replace("+00:00", "Z")
    )
    changed = tmp_path / "overlap.json"
    inputs["subject_sha256"] = helper.write(changed, data)
    inputs["subject"] = changed
    with pytest.raises(ValueError, match="baseline must end"):
        value.prepare(**inputs, output=tmp_path / "project")
    declaration = json.loads(inputs["protocol"].read_bytes())
    declaration["journey_source_digest"] = helper.digest(b"drift")
    path = tmp_path / "drift.json"
    inputs["protocol_sha256"] = helper.write(path, declaration)
    inputs["protocol"] = path
    with pytest.raises(ValueError, match="journey source"):
        value.prepare(**inputs, output=tmp_path / "project")


@pytest.mark.parametrize(
    "path,editable,accepted",
    [
        ("/env/lib/site-packages/invarlock/__init__.py", False, True),
        ("/repo/src/invarlock/__init__.py", False, False),
        ("/env/lib/site-packages/invarlock/__init__.py", True, False),
    ],
)
def test_wheel_guard(path, editable, accepted, monkeypatch):
    value = module()

    def probe(argv, **kwargs):
        assert argv[1] == "-I"
        return subprocess.CompletedProcess(
            argv,
            0,
            json.dumps(
                {
                    "module": path,
                    "prefix": "/env",
                    "direct_url": json.dumps({"dir_info": {"editable": editable}}),
                }
            ),
        )

    monkeypatch.setattr(value.subprocess, "run", probe)
    if accepted:
        assert value.require_wheel("/env/bin/invarlock") == Path("/env/bin/invarlock")
    else:
        with pytest.raises(ValueError, match="installed wheels"):
            value.require_wheel("/env/bin/invarlock")


def test_command_allowlists_environment_and_sanitizes_errors(tmp_path, monkeypatch):
    value = module()
    monkeypatch.setenv("PRIVATE_PROVIDER_TOKEN", "never-forward")

    def run(argv, **kwargs):
        assert "PRIVATE_PROVIDER_TOKEN" not in kwargs["env"]
        return subprocess.CompletedProcess(argv, 9, "secret output")

    monkeypatch.setattr(value.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="unexpected status 9") as error:
        value.command("invarlock", tmp_path, "verify")
    assert "secret" not in str(error.value)
    monkeypatch.setattr(
        value.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, '{"ok":true}'),
    )
    assert value.command("invarlock", tmp_path, "verify") == {"ok": True}


def test_main_prepare_and_journey_dispatch(tmp_path, monkeypatch, capsys):
    value = module()
    inputs = captures(tmp_path, monkeypatch)
    import io

    monkeypatch.setattr(sys, "argv", ["journey.py", "_prepare"])
    monkeypatch.setattr(
        sys,
        "stdin",
        io.TextIOWrapper(
            io.BytesIO(
                json.dumps(
                    {
                        **{k: str(v) for k, v in inputs.items()},
                        "output": str(tmp_path / "prepared"),
                    }
                ).encode()
            )
        ),
    )
    value.main()
    assert "baseline_run_digest" in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "journey.py",
            *[
                item
                for k, v in {
                    **inputs,
                    "operator_cli": "/operator/bin/invarlock",
                    "recipient_cli": "/recipient/bin/invarlock",
                    "output": tmp_path / "out",
                }.items()
                for item in ("--" + k.replace("_", "-"), str(v))
            ],
        ],
    )
    monkeypatch.setattr(
        value,
        "journey",
        lambda **kwargs: {
            "evaluation": {"decision": "pass"},
            "verification": {"integrity_ok": True},
            "source_assurance": "captured_inputs",
        },
    )
    value.main()
    assert json.loads(capsys.readouterr().out)["decision"] == "pass"
