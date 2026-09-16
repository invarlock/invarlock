"""Exercise the portable HTTP example's literal preparation/capture/export commands."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from invarlock.engine import compare_runs
from tests.examples.test_hosted_service_capture import module, response, server

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/hosted-service"


def test_hosted_example_protocol_and_literal_commands(tmp_path):
    helper = module()
    blocks = re.findall(
        r"```bash\n(.*?)\n```", (EXAMPLE / "README.md").read_text(), re.S
    )
    assert len(blocks) == 5
    subprocess.run(
        ["bash", "-n"],
        input=blocks[4],
        text=True,
        capture_output=True,
        check=True,
    )
    copied = tmp_path / "examples/hosted-service"
    copied.mkdir(parents=True)
    for name in ("capture.py", "journey.py", "example-protocol.json"):
        shutil.copyfile(EXAMPLE / name, copied / name)
    environment = {
        "PATH": str(Path(sys.executable).parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
    }
    subprocess.run(
        ["bash", "-eu", "-c", blocks[0]],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    with server([(200, helper.encoded(response()))]) as (endpoint, calls):
        protocol_path = tmp_path / "campaign/protocol.json"
        protocol = json.loads(protocol_path.read_text())
        for service in protocol["services"].values():
            service["endpoint"] = endpoint
        protocol_path.write_text(json.dumps(protocol))
        commands = "\n".join(blocks[1:4])
        result = subprocess.run(
            ["bash", "-eu", "-c", commands],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(calls) == 8
        assert all(call["authorization"] is None for call in calls)
        assert all("expected" not in call["body"] for call in calls)
    # Server is stopped: preparation/export of both original captures remains offline.
    protocol = helper.load_protocol(
        protocol_path, helper.digest(protocol_path.read_bytes())
    )
    assert protocol["collector_source_digest"] == helper.digest(
        (copied / "capture.py").read_bytes()
    )
    assert protocol["journey_source_digest"] == helper.digest(
        (copied / "journey.py").read_bytes()
    )
    spec = importlib.util.spec_from_file_location(
        "documented_journey", copied / "journey.py"
    )
    journey = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(journey)
    baseline = tmp_path / "campaign/baseline/capture.json"
    subject = tmp_path / "campaign/subject/capture.json"
    prepared = tmp_path / "campaign/prepared"
    journey.prepare(
        protocol_path,
        helper.digest(protocol_path.read_bytes()),
        baseline,
        helper.digest(baseline.read_bytes()),
        subject,
        helper.digest(subject.read_bytes()),
        prepared,
    )
    subject_run = json.loads((tmp_path / "campaign/subject-run.json").read_text())
    assert subject_run == json.loads((prepared / "subject.json").read_text())
    assert subject_run["artifact_digest"] is None
    assert subject_run["service_identity"]["observed_model"] == "simulation-model"
    assert subject_run["service_identity"]["exposed_revision"] is None
    compared = compare_runs(
        baseline=json.loads((prepared / "baseline.json").read_text()),
        subject=subject_run,
        policy=protocol["policy"],
    )
    assert compared["decision"] == "insufficient_evidence"
