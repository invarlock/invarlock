from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app as cli

LIGHT_IMPORT = os.getenv("INVARLOCK_LIGHT_IMPORT", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def test_report_generate_help_does_not_require_run(tmp_path: Path) -> None:
    if LIGHT_IMPORT:
        # In light-import mode, help layout may differ across Typer/Rich
        # versions; exercise this more fully in the default import path.
        return
    r = CliRunner().invoke(cli, ["report", "generate", "--help"])
    assert r.exit_code == 0, r.stdout
    # Should show generate help, not the group root help or a missing --run error
    assert "Generate reports from existing run report artifacts" in r.stdout
    assert "--run" in r.stdout


def test_report_html_help_shows_short_flags(tmp_path: Path) -> None:
    if LIGHT_IMPORT:
        # In light-import mode, focus on minimal import safety; detailed help
        # flag layout is exercised under the default import path.
        return
    r = CliRunner().invoke(cli, ["report", "html", "--help"])
    assert r.exit_code == 0
    # Ensure short flags are wired
    assert "--input" in r.stdout
    assert "--output" in r.stdout
