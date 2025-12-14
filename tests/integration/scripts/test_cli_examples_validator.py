import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_examples_validator_accepts_valid_commands(tmp_path: Path) -> None:
    md = tmp_path / "sample.md"
    md.write_text(
        "\n".join(
            [
                "```bash",
                "invarlock version",
                "INVARLOCK_ALLOW_NETWORK=0 invarlock doctor",
                "invarlock certify --baseline foo --subject bar --adapter hf_gpt2 --profile ci --tier none",
                "python -m invarlock version",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    res = subprocess.run(
        [sys.executable, "scripts/test_cli_examples.py", "--paths", str(md)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    assert "Validated" in (res.stdout + res.stderr)


def test_cli_examples_validator_rejects_invalid_flag(tmp_path: Path) -> None:
    md = tmp_path / "bad.md"
    md.write_text(
        "\n".join(
            [
                "```bash",
                "invarlock version --bogus",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    res = subprocess.run(
        [sys.executable, "scripts/test_cli_examples.py", "--paths", str(md)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "CLI example validation failed" in (res.stdout + res.stderr)


pytestmark = pytest.mark.integration
