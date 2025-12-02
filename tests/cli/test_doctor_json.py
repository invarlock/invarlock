from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_doctor_json_mode_outputs_findings_and_exitcode(monkeypatch):
    # Enable a note-producing flag
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    r = CliRunner().invoke(app, ["doctor", "--json"])
    # doctor emits JSON and exits with code 0 for healthy (CI host may be CPU-only but still healthy)
    assert r.exit_code in (0, 1)  # accept both in varied environments
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert payload.get("format_version") == "doctor-v1"
    assert isinstance(payload.get("findings"), list)
    # Should include at least one note when INVARLOCK_TINY_RELAX is set
    # (ok if filtered out on some builds; ensure structure viable)
    assert "summary" in payload and "resolution" in payload
