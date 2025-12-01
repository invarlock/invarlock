from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_report_requires_run_flag_when_no_subcommand():
    r = CliRunner().invoke(app, ["report"])  # missing --run
    assert r.exit_code == 2
    assert "--run is required" in r.stdout


def test_report_cert_requires_baseline(tmp_path: Path):
    # Minimal run report path
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {"meta": {"model_id": "m", "adapter": "hf", "seed": 0, "device": "cpu"}}
        )
    )
    r = CliRunner().invoke(
        app, ["report", "--run", str(run), "--format", "cert"]
    )  # no --baseline
    assert r.exit_code == 1
    assert "Certificate format requires --baseline" in r.stdout
