from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_evaluate_hf_id_normalization_and_preset_fallback(monkeypatch, tmp_path: Path):
    # Patch auto adapter and run/report commands to be no-ops
    import invarlock.cli.commands.evaluate as cert_mod
    import invarlock.cli.commands.run as run_mod

    captured = []

    def _dump_yaml_capture(path: Path, data: dict):
        captured.append((Path(path), data))

    monkeypatch.setattr(cert_mod, "_dump_yaml", _dump_yaml_capture)
    monkeypatch.setattr(cert_mod, "resolve_auto_adapter", lambda src: "hf_causal")

    baseline_rep = tmp_path / "baseline.json"
    baseline_rep.write_text(
        json.dumps(
            {"meta": {"model_id": "m", "adapter": "hf", "seed": 0, "device": "cpu"}}
        )
    )
    edited_rep = tmp_path / "edited.json"
    edited_rep.write_text(
        json.dumps(
            {"meta": {"model_id": "m2", "adapter": "hf", "seed": 0, "device": "cpu"}}
        )
    )

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: str(
            baseline_rep if Path(kwargs["out"]).name.endswith("source") else edited_rep
        ),
    )
    # No-op report emitter
    monkeypatch.setattr(cert_mod, "generate_reports", lambda **kwargs: None)

    # Run with hf: prefix to exercise normalization; preset path is default fallback when missing
    r = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--baseline",
            "hf:org/modelA",
            "--subject",
            "hf:org/modelB",
            "--baseline-adapter",
            "auto",
            "--subject-adapter",
            "auto",
            "--profile",
            "dev",
            "--assurance",
            "off",
            "--out",
            str(tmp_path / "runs"),
            "--report-out",
            str(tmp_path / "cert"),
        ],
    )
    assert r.exit_code == 0, r.stdout
    # First captured config is baseline; ensure hf: was stripped for HF adapter
    assert captured, "_dump_yaml should be called"
    baseline_cfg = captured[0][1]
    assert baseline_cfg["model"]["id"] == "org/modelA"


def test_evaluate_ci_aborts_on_nonfinite_pm(monkeypatch, tmp_path: Path):
    import invarlock.cli.commands.evaluate as cert_mod
    import invarlock.cli.commands.run as run_mod

    baseline_rep = tmp_path / "baseline.json"
    baseline_rep.write_text(
        json.dumps(
            {"meta": {"model_id": "m", "adapter": "hf", "seed": 0, "device": "cpu"}}
        )
    )
    edited_rep = tmp_path / "edited.json"
    # Include a primary_metric with non-finite final (None)
    edited_rep.write_text(
        json.dumps(
            {
                "meta": {"model_id": "m2", "adapter": "hf", "seed": 0, "device": "cpu"},
                "metrics": {"primary_metric": {"kind": "ppl_causal", "final": None}},
                "edit": {"name": "noop"},
            }
        )
    )

    monkeypatch.setattr(
        run_mod,
        "run_command",
        lambda **kwargs: str(
            baseline_rep if Path(kwargs["out"]).name.endswith("source") else edited_rep
        ),
    )
    # No-op report emitter
    monkeypatch.setattr(cert_mod, "generate_reports", lambda **kwargs: None)
    monkeypatch.setattr(cert_mod, "resolve_auto_adapter", lambda src: "hf_causal")

    r = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--baseline",
            "hf:org/modelA",
            "--subject",
            "hf:org/modelB",
            "--baseline-adapter",
            "auto",
            "--subject-adapter",
            "auto",
            "--profile",
            "ci",
            "--out",
            str(tmp_path / "runs"),
            "--report-out",
            str(tmp_path / "cert"),
        ],
    )
    # CI profile with non-finite pm should hard abort (exit 3)
    assert r.exit_code == 3
    assert "Primary metric computation failed" in r.stdout
