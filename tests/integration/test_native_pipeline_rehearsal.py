"""Opt-in real upstream capture using a predeclared local-model protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_native_pipeline_capture():
    if os.environ.get("INVARLOCK_RUN_NATIVE_PIPELINE") != "1":
        pytest.skip(
            "set INVARLOCK_RUN_NATIVE_PIPELINE=1 with the explicit capture inputs"
        )
    output = Path(os.environ["INVARLOCK_NATIVE_OUTPUT"])
    command = [
        os.environ["INVARLOCK_NATIVE_PYTHON"],
        str(ROOT / "examples/pipeline/native_rehearsal.py"),
        "capture",
        "--evaluator",
        os.environ["INVARLOCK_NATIVE_EVALUATOR"],
        "--model",
        os.environ["INVARLOCK_NATIVE_MODEL"],
        "--protocol",
        os.environ["INVARLOCK_NATIVE_PROTOCOL"],
        "--expected-protocol",
        os.environ["INVARLOCK_NATIVE_PROTOCOL_SHA256"],
        "--output",
        str(output),
        "--promptfoo",
        os.environ.get("INVARLOCK_NATIVE_PROMPTFOO", "promptfoo"),
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("INVARLOCK_PIPELINE_SIGNING_KEY", None)
    result = subprocess.run(
        command,
        cwd=output.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    # Retain failed setup/inference diagnostics as well as successful captures.
    (output.parent / f"{output.name}-process.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        )
        + "\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    raw = (output / "capture.json").read_bytes()
    assert (
        result.stdout.strip().splitlines()[-1]
        == "sha256:" + hashlib.sha256(raw).hexdigest()
    )
    manifest = json.loads(raw)
    assert manifest["evaluator"] == os.environ["INVARLOCK_NATIVE_EVALUATOR"]
    assert (
        len([name for name in manifest["files"] if name.endswith("-calls.json")]) == 6
    )
    for name, expected in manifest["files"].items():
        assert (
            "sha256:" + hashlib.sha256((output / name).read_bytes()).hexdigest()
            == expected
        )
