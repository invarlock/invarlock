import json
from pathlib import Path

import click
import pytest
import yaml

from invarlock.cli.commands.certify import certify_command


class _StubCLIExit(Exception):
    pass


def _stub_run(out_dir: Path, baseline: Path | None = None):
    # Create a deterministic timestamp directory and write a minimal report.json
    ts_dir = out_dir / "20250101_000000"
    ts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": {"model_id": "stub", "adapter": "hf_gpt2"},
        "edit": {"name": "quant_rtn"},
        "metrics": {"ppl_ratio": 1.0, "ppl_final": 10.0},
        "data": {"preview_n": 1, "final_n": 1},
    }
    (ts_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_certify_orchestrates_runs_and_cert(monkeypatch, tmp_path):
    # Arrange: create HF-like dirs for source/edited
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    calls = {"runs": [], "reports": []}

    def fake_run(**kwargs):
        out = Path(kwargs.get("out"))
        calls["runs"].append(
            {k: kwargs.get(k) for k in ["config", "profile", "out", "baseline"]}
        )
        _stub_run(out)

    def fake_report(**kwargs):
        calls["reports"].append(kwargs)

    # Patch in our fakes
    # Patch the lazily imported run command at its source module
    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import certify as mod

    monkeypatch.setattr(
        run_mod, "run_command", lambda **kwargs: fake_run(**kwargs), raising=False
    )
    # Patch the report entry already imported as _report in certify module
    monkeypatch.setattr(mod, "_report", fake_report, raising=False)

    # Act
    certify_command(
        source=str(src),
        edited=str(edt),
        adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        cert_out=str(tmp_path / "reports"),
    )

    # Assert: two runs and one report
    assert len(calls["runs"]) == 2
    assert len(calls["reports"]) == 1

    # Baseline first, edited second with baseline set
    assert calls["runs"][0]["baseline"] is None
    assert Path(calls["runs"][0]["out"]).name == "source"
    assert Path(calls["runs"][1]["out"]).name == "edited"
    assert calls["runs"][1]["baseline"] is not None

    # Report uses the edited run and baseline
    rep = calls["reports"][0]
    assert rep["format"] == "cert"
    assert Path(rep["output"]).name == "reports"
    assert rep["baseline"] is not None and rep["run"] is not None


def test_certify_reuses_baseline_report_skipping_baseline_run(monkeypatch, tmp_path):
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    baseline_report = tmp_path / "baseline_report.json"
    baseline_report.write_text(
        json.dumps(
            {
                "meta": {"model_id": "stub", "adapter": "hf_gpt2"},
                "context": {"profile": "ci", "auto": {"tier": "balanced"}},
                "edit": {"name": "noop"},
                "evaluation_windows": {
                    "preview": {"window_ids": [1], "input_ids": [[1, 2]]},
                    "final": {"window_ids": [2], "input_ids": [[3, 4]]},
                },
                "metrics": {
                    "primary_metric": {
                        "kind": "ppl_causal",
                        "preview": 1.0,
                        "final": 1.0,
                        "ratio_vs_baseline": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    calls = {"runs": [], "reports": []}

    def fake_run(**kwargs):  # noqa: ANN001
        out = Path(kwargs.get("out"))
        calls["runs"].append(
            {k: kwargs.get(k) for k in ["config", "profile", "out", "baseline"]}
        )
        _stub_run(out)

    def fake_report(**kwargs):  # noqa: ANN001
        calls["reports"].append(kwargs)

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import certify as mod

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(mod, "_report", fake_report, raising=False)

    certify_command(
        source=str(src),
        edited=str(edt),
        baseline_report=str(baseline_report),
        adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        cert_out=str(tmp_path / "reports"),
    )

    assert len(calls["runs"]) == 1
    assert Path(calls["runs"][0]["out"]).name == "edited"
    assert Path(calls["runs"][0]["baseline"]).resolve() == baseline_report.resolve()

    assert len(calls["reports"]) == 1
    rep = calls["reports"][0]
    assert Path(rep["baseline"]).resolve() == baseline_report.resolve()


def test_certify_baseline_report_requires_windows(monkeypatch, tmp_path):
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    baseline_report = tmp_path / "baseline_report.json"
    baseline_report.write_text(
        json.dumps(
            {
                "meta": {"model_id": "stub", "adapter": "hf_gpt2"},
                "context": {"profile": "ci", "auto": {"tier": "balanced"}},
                "edit": {"name": "noop"},
                "evaluation_windows": {"final": {"window_ids": [1]}},
            }
        ),
        encoding="utf-8",
    )

    import invarlock.cli.commands.run as run_mod

    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit):
        certify_command(
            source=str(src),
            edited=str(edt),
            baseline_report=str(baseline_report),
            adapter="auto",
            profile="ci",
            out=str(tmp_path / "runs"),
            cert_out=str(tmp_path / "reports"),
        )


def test_certify_autogen_uses_device_auto(monkeypatch, tmp_path):
    """Auto-generated certify presets should not hard-code CPU device."""
    # Arrange HF-like source/edited dirs so auto adapter resolves to hf_gpt2
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    # Stub run/report so we don't execute real pipelines but capture device
    calls: list[dict] = []

    def fake_run(**kwargs):
        calls.append(
            {k: kwargs.get(k) for k in ("config", "profile", "out", "tier", "device")}
        )
        out = Path(kwargs.get("out"))
        _stub_run(out)

    def fake_report(**_kwargs):
        return None

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import certify as mod

    monkeypatch.setattr(
        run_mod, "run_command", lambda **kwargs: fake_run(**kwargs), raising=False
    )
    monkeypatch.setattr(mod, "_report", fake_report, raising=False)

    # Act
    certify_command(
        source=str(src),
        edited=str(edt),
        adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        cert_out=str(tmp_path / "reports"),
    )

    # Assert: temp baseline config exists and does not pin device=cpu
    baseline_yaml = Path(".certify_tmp") / "baseline_noop.yaml"
    assert baseline_yaml.exists()
    data = yaml.safe_load(baseline_yaml.read_text(encoding="utf-8")) or {}
    model_block = data.get("model") or {}
    # Ensure the preset did not pin device=cpu
    assert model_block.get("device") != "cpu"
    # And the run call saw device=None (auto resolution) by default
    assert calls and calls[0]["device"] is None


def test_certify_quiet_summary_emits_status(monkeypatch, tmp_path, capsys):
    src = tmp_path / "src_model"
    edt = tmp_path / "edt_model"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )
    (edt / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    def fake_run(**kwargs):
        out = Path(kwargs.get("out"))
        _stub_run(out)

    def fake_report(**kwargs):
        output_dir = Path(kwargs.get("output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        cert = {
            "primary_metric": {"ratio_vs_baseline": 1.01},
            "validation": {
                "primary_metric_acceptable": True,
                "preview_final_drift_acceptable": True,
                "invariants_pass": True,
                "spectral_stable": True,
                "rmt_stable": True,
            },
        }
        (output_dir / "evaluation.cert.json").write_text(
            json.dumps(cert), encoding="utf-8"
        )

    import invarlock.cli.commands.run as run_mod
    from invarlock.cli.commands import certify as mod

    monkeypatch.setattr(
        run_mod, "run_command", lambda **kwargs: fake_run(**kwargs), raising=False
    )
    monkeypatch.setattr(mod, "_report", fake_report, raising=False)

    certify_command(
        source=str(src),
        edited=str(edt),
        adapter="auto",
        profile="ci",
        out=str(tmp_path / "runs"),
        cert_out=str(tmp_path / "reports"),
        quiet=True,
    )

    out = capsys.readouterr().out
    assert "INVARLOCK v" in out
    assert "Status: PASS" in out
    assert "Output:" in out
