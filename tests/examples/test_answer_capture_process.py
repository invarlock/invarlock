from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from examples.answer_capture import capture, read
from examples.answer_capture_judge import prepare
from examples.answer_capture_process import ProcessAdapter

ROOT = Path(__file__).resolve().parents[2]


def adapter(tmp_path, body=None, *, timeout=3, output_bytes=4096, environment=None):
    executable = str(Path(sys.executable).resolve())
    script = ROOT / "examples/answer-capture/pipeline.py"
    if body is not None:
        script = tmp_path / "process.py"
        script.write_text(body)
    path = tmp_path / "transport.json"
    path.write_text(
        json.dumps(
            {
                "argv": [executable, "-I", str(script)],
                "assets": [str(script)],
                "environment": environment or [],
            }
        )
    )
    config = read(ROOT / "examples/answer-capture/config.json")
    config["limits"]["deadline_unix_seconds"] = int(time.time()) + 300
    config["limits"]["call_timeout_seconds"] = timeout
    config["limits"]["max_output_bytes_per_call"] = output_bytes
    return ProcessAdapter(path, config["limits"]), config


def request():
    return {
        "side": "baseline",
        "case_id": "case-1",
        "input": "Capital of France?",
        "model": {},
        "max_output_tokens": 32,
    }


def test_pipeline_capture_and_bound_continuation(tmp_path):
    transport, config = adapter(tmp_path)
    cases = read(ROOT / "examples/answer-capture/cases.json")
    directory = tmp_path / "capture"
    result = asyncio.run(
        capture(
            config=config,
            cases=cases,
            adapter_sha256=transport.sha256,
            adapter=transport,
            directory=directory,
        )
    )
    assert result == asyncio.run(
        capture(
            config=config,
            cases=cases,
            adapter_sha256=transport.sha256,
            adapter=transport,
            directory=directory,
        )
    )
    assert read(directory / "manifest.json")["adapter_identity"] == transport.identity
    target = tmp_path / "judge"
    plan = read(ROOT / "examples/judge-measurements/plan.json")
    policy = read(ROOT / "examples/judge-measurements/analysis_policy.json")
    collection = read(ROOT / "examples/judge-measurements/collection.json")
    summary = prepare(directory, plan, policy, {"case-1": "unit-1"}, collection, target)
    bound = read(target / "plan.json")
    assert summary["expected_trials"] == 2
    assert bound["baseline_run_sha256"] == result["baseline"]
    assert bound["subject_run_sha256"] == result["subject"]
    assert bound["case_set_sha256"] == result["case_set"]
    assert (
        read(target / "analysis_policy.json")["minimum_units"]
        == policy["minimum_units"]
    )
    # The prepared request goes through the actual public preflight.
    from typer.testing import CliRunner

    from invarlock.cli.app import app

    checked = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(target / "request.json"),
            "--preflight",
            "--json",
        ],
    )
    assert checked.exit_code == 0, checked.output
    with pytest.raises(FileExistsError):
        prepare(directory, plan, policy, {"case-1": "unit-1"}, collection, target)
    run_path = directory / "subject_run.json"
    run = read(run_path)
    run["records"][0]["output"] = "replacement answer"
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="first result"):
        prepare(
            directory, plan, policy, {"case-1": "unit-1"}, collection, tmp_path / "bad"
        )


@pytest.mark.parametrize(
    "body, message",
    [
        ('print("x" * 100000)', "byte cap"),
        ('import sys; sys.stderr.write("x" * 100000)', "byte cap"),
        ('print(\'{"input_tokens":1,"extra":2}\')', "exactly"),
        ('print(\'{"input_tokens":1,"input_tokens":2}\')', "duplicate"),
        ('import sys; sys.stderr.write("secret-token"); sys.exit(9)', "status 9"),
        ("import time; time.sleep(30)", "timed out"),
    ],
)
def test_bounded_failure(tmp_path, body, message):
    transport, _ = adapter(tmp_path, body, timeout=1, output_bytes=10)
    with pytest.raises(ValueError, match=message) as stopped:
        transport.count_input_tokens(request())
    assert "secret-token" not in str(stopped.value)


def test_clean_environment_and_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTURE_SECRET", "present")
    body = 'import os, json; print(json.dumps({"input_tokens": 2 if "CAPTURE_SECRET" in os.environ else 1}))'
    transport, _ = adapter(tmp_path, body)
    assert transport.count_input_tokens(request()) == 1
    transport, _ = adapter(tmp_path, body, environment=["CAPTURE_SECRET"])
    assert transport.count_input_tokens(request()) == 2
    assert "present" not in json.dumps(transport.identity)


def test_changed_asset_refused(tmp_path):
    transport, _ = adapter(tmp_path, "print('{\"input_tokens\":1}')")
    (tmp_path / "process.py").write_text("print('{\"input_tokens\":2}')")
    with pytest.raises(ValueError, match="changed"):
        transport.count_input_tokens(request())


def test_cancellation_kills_process(tmp_path):
    pidfile = tmp_path / "pid"
    transport, _ = adapter(
        tmp_path,
        f'import os, time; open({str(pidfile)!r}, "w").write(str(os.getpid())); time.sleep(30)',
    )

    async def cancel():
        task = asyncio.create_task(transport.generate(request()))
        for _ in range(100):
            if pidfile.exists():
                break
            await asyncio.sleep(0.01)
        assert pidfile.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
