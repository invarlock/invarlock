from __future__ import annotations

import os

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_top_level_help_shows_certify_and_run_descriptions() -> None:
    # Keep imports light for help rendering in tests
    os.environ["INVARLOCK_LIGHT_IMPORT"] = "1"
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.output
    # Ensure commands are listed with descriptions
    assert "certify" in out and "Certify a subject model" in out
    assert "run" in out and "end-to-end run from a YAML" in out
