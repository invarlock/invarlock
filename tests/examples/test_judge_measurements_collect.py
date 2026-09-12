from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "examples" / "judge-measurements" / "collect.py"


def test_collection_example_is_inert_without_explicit_execution(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--execute-collection" in result.stderr
    assert not any(tmp_path.iterdir())


def test_collection_example_help_needs_no_optional_sdk():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--execute-collection" in result.stdout


def test_collection_example_checks_output_before_loading_optional_sdk(tmp_path):
    output = tmp_path / "measurements-collected.json"
    output.write_text("already present")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--execute-collection",
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "OPENAI_API_KEY": "unused-test-key"},
    )
    assert result.returncode == 2
    assert "must be a new file" in result.stderr
