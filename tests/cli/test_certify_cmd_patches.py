from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from invarlock.cli.app import app


def test_certify_hf_id_normalization_and_preset_fallback(monkeypatch, tmp_path: Path):
    # Patch auto adapter and run/report commands to be no-ops
    import invarlock.cli.commands.certify as cert_mod
    import invarlock.cli.commands.run as run_mod

    monkeypatch.setattr(run_mod, "run_command", lambda **kwargs: None)

    captured = []

    def _dump_yaml_capture(path: Path, data: dict):  # type: ignore[no-untyped-def]
        captured.append((Path(path), data))

    monkeypatch.setattr(cert_mod, "_dump_yaml", _dump_yaml_capture)
    monkeypatch.setattr(cert_mod, "resolve_auto_adapter", lambda src: "hf_causal")

    # Provide fake latest report paths post-run
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

    def _fake_latest(run_root: Path) -> Path | None:  # type: ignore[override]
        return baseline_rep if run_root.name.endswith("source") else edited_rep

    monkeypatch.setattr(cert_mod, "_latest_run_report", _fake_latest)
    # No-op report emitter
    monkeypatch.setattr(cert_mod, "_report", lambda **kwargs: None)

    # Run with hf: prefix to exercise normalization; preset path is default fallback when missing
    r = CliRunner().invoke(
        app,
        [
            "certify",
            "--source",
            "hf:org/modelA",
            "--edited",
            "hf:org/modelB",
            "--adapter",
            "auto",
            "--profile",
            "dev",
            "--out",
            str(tmp_path / "runs"),
            "--cert-out",
            str(tmp_path / "cert"),
        ],
    )
    assert r.exit_code == 0, r.stdout
    # First captured config is baseline; ensure hf: was stripped for HF adapter
    assert captured, "_dump_yaml should be called"
    baseline_cfg = captured[0][1]
    assert baseline_cfg["model"]["id"] == "org/modelA"


def test_certify_ci_aborts_on_nonfinite_pm(monkeypatch, tmp_path: Path):
    import invarlock.cli.commands.certify as cert_mod
    import invarlock.cli.commands.run as run_mod

    monkeypatch.setattr(run_mod, "run_command", lambda **kwargs: None)
    # Provide fake latest reports
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

    def _fake_latest(run_root: Path) -> Path | None:  # type: ignore[override]
        return baseline_rep if run_root.name.endswith("source") else edited_rep

    monkeypatch.setattr(cert_mod, "_latest_run_report", _fake_latest)
    # No-op report emitter
    monkeypatch.setattr(cert_mod, "_report", lambda **kwargs: None)
    monkeypatch.setattr(cert_mod, "resolve_auto_adapter", lambda src: "hf_causal")

    r = CliRunner().invoke(
        app,
        [
            "certify",
            "--source",
            "hf:org/modelA",
            "--edited",
            "hf:org/modelB",
            "--adapter",
            "auto",
            "--profile",
            "ci",
            "--out",
            str(tmp_path / "runs"),
            "--cert-out",
            str(tmp_path / "cert"),
        ],
    )
    # CI profile with non-finite pm should hard abort (exit 3)
    assert r.exit_code == 3
    assert "Primary metric computation failed" in r.stdout
