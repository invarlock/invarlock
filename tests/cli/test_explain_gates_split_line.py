from __future__ import annotations

import json

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_explain_gates_dataset_split_line(monkeypatch, tmp_path):
    rep = tmp_path / "rep.json"
    base = tmp_path / "base.json"
    rep.write_text(
        json.dumps(
            {"provenance": {"dataset_split": "validation", "split_fallback": True}}
        ),
        encoding="utf-8",
    )
    base.write_text(json.dumps({}), encoding="utf-8")

    from invarlock.cli.commands import explain_gates as mod

    def _fake_cert(_report, _baseline):  # type: ignore[no-untyped-def]
        return {
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
            },
            "primary_metric": {
                "kind": "ppl_causal",
                "preview": 10.0,
                "final": 10.0,
                "ratio_vs_baseline": 1.0,
                "display_ci": [1.0, 1.0],
            },
        }

    monkeypatch.setattr(mod, "make_certificate", _fake_cert)
    r = CliRunner().invoke(
        app, ["report", "explain", "--report", str(rep), "--baseline", str(base)]
    )
    assert r.exit_code == 0
    assert "Dataset split: validation" in r.stdout and "(fallback)" in r.stdout
