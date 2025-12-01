from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_doctor_json_includes_tiny_relax_note(monkeypatch):
    monkeypatch.setenv("INVARLOCK_TINY_RELAX", "1")
    r = CliRunner().invoke(app, ["doctor", "--json"])
    assert r.exit_code in (0, 1)
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    findings = payload.get("findings", [])
    # Expect a D013 note present
    assert any(
        f.get("code") == "D013" and f.get("severity") == "note" for f in findings
    )
