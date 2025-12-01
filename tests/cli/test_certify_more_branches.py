from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import click
import pytest
import yaml

import invarlock.cli.commands.run as run_mod
from invarlock.cli.commands import certify as mod


def _stub_run_dir(out_dir: Path, name: str = "report.json") -> Path:
    ts = out_dir / "20250101_000000"
    ts.mkdir(parents=True, exist_ok=True)
    report_path = ts / name
    report_path.write_text(
        json.dumps({"meta": {}, "metrics": {}, "data": {}}), encoding="utf-8"
    )
    return report_path


def test_latest_run_report_variants(tmp_path: Path):
    # Non-existent root → None
    assert mod._latest_run_report(tmp_path / "missing") is None
    # Exists but empty → None
    root = tmp_path / "runs" / "source"
    root.mkdir(parents=True, exist_ok=True)
    assert mod._latest_run_report(root) is None
    # Has a directory but no standard report.json; picks first *.json
    ts = root / "20250101_000000"
    ts.mkdir()
    alt = ts / f"{ts.name}.json"
    alt.write_text("{}", encoding="utf-8")
    assert mod._latest_run_report(root) == alt
    alt.unlink()
    fallback = ts / "fallback.json"
    fallback.write_text("{}", encoding="utf-8")
    assert mod._latest_run_report(root) == fallback


def test_normalize_model_id_handles_bad_adapter():
    class BadAdapter:
        def __str__(self):  # pragma: no cover - invoked indirectly
            raise RuntimeError("boom")

    result = mod._normalize_model_id("hf:demo/model", BadAdapter())
    assert result == "hf:demo/model"


def test_load_yaml_rejects_non_mapping(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("- this is a list, not a mapping", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._load_yaml(p)


def test_certify_missing_preset_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    # Patch to prevent actual run invocation
    monkeypatch.setattr(run_mod, "run_command", lambda **k: None, raising=False)
    with pytest.raises(click.exceptions.Exit):
        mod.certify_command(
            source=str(src),
            edited=str(edt),
            preset=str(tmp_path / "no_such_preset.yaml"),
            out=str(tmp_path / "runs"),
        )


def test_certify_uses_inline_preset_when_repo_preset_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()
    runs = Path("runs")
    calls = {"runs": 0, "reports": 0}

    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")
    edited_report = tmp_path / "edited.json"
    edited_report.write_text("{}", encoding="utf-8")

    def fake_latest(run_root: Path):
        return baseline_report if Path(run_root).name == "source" else edited_report

    def fake_run(**_kwargs):
        calls["runs"] += 1

    def fake_report(**_kwargs):
        calls["reports"] += 1

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(mod, "_latest_run_report", fake_latest)
    monkeypatch.setattr(mod, "_report", fake_report, raising=False)

    mod.certify_command(
        source=str(src),
        edited=str(edt),
        adapter="hf_gpt2",
        out=str(runs),
        cert_out=str(Path("certs")),
        profile="dev",
    )

    cfg_path = Path(".certify_tmp/baseline_noop.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["dataset"]["provider"] == "wikitext2"
    assert calls["runs"] == 2 and calls["reports"] == 1


def test_certify_edit_config_successfully_merges_subject(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()
    preset = Path("preset.yaml")
    preset.write_text("dataset: { provider: demo }\n", encoding="utf-8")
    edit_cfg = Path("edit_config.yaml")
    edit_cfg.write_text(
        "model:\n  id: \"<MODEL_ID>\"\n  adapter: ''\nedit:\n  name: quant_rtn\n  plan: {}\n",
        encoding="utf-8",
    )

    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")
    edited_report = tmp_path / "edited.json"
    edited_report.write_text("{}", encoding="utf-8")

    def fake_latest(run_root: Path):
        return baseline_report if Path(run_root).name == "source" else edited_report

    calls = {"runs": 0}

    def fake_run(**kwargs):
        calls["runs"] += 1
        if calls["runs"] == 1:
            assert kwargs.get("baseline") is None
        else:
            assert kwargs.get("baseline") == str(baseline_report)

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    monkeypatch.setattr(mod, "_latest_run_report", fake_latest)
    monkeypatch.setattr(mod, "_report", lambda **_: None, raising=False)

    mod.certify_command(
        source=str(src),
        edited=str(edt),
        adapter="hf_gpt2",
        preset=str(preset),
        edit_config=str(edit_cfg),
        out=str(Path("runs")),
        cert_out=str(Path("certs")),
        profile="dev",
    )

    merged = yaml.safe_load(
        Path(".certify_tmp/edited_merged.yaml").read_text(encoding="utf-8")
    )
    assert merged["model"]["id"] == str(edt)
    assert merged["model"]["adapter"] == "hf_gpt2"
    assert calls["runs"] == 2


def test_certify_edit_config_invalid_yaml_exits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()
    edit_cfg = Path("edit_config.yaml")
    edit_cfg.write_text("- not a mapping", encoding="utf-8")
    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(
        mod,
        "_latest_run_report",
        lambda run_root: baseline_report
        if Path(run_root).name == "source"
        else baseline_report,
    )

    with pytest.raises(click.exceptions.Exit):
        mod.certify_command(
            source=str(src),
            edited=str(edt),
            adapter="hf_gpt2",
            edit_config=str(edit_cfg),
            out=str(Path("runs")),
        )


def test_certify_ci_profile_invalid_json_exits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()

    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")
    bad_report = tmp_path / "edited.json"
    bad_report.write_text("{not-json", encoding="utf-8")

    def fake_latest(run_root: Path):
        return baseline_report if Path(run_root).name == "source" else bad_report

    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "_latest_run_report", fake_latest)
    monkeypatch.setattr(mod, "_report", lambda **_: None, raising=False)

    with pytest.raises(click.exceptions.Exit):
        mod.certify_command(
            source=str(src),
            edited=str(edt),
            adapter="hf_gpt2",
            out=str(Path("runs")),
            profile="ci",
        )


def test_certify_ci_nonfinite_primary_metric_exits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    src = Path("src")
    edt = Path("edt")
    src.mkdir()
    edt.mkdir()

    baseline_report = tmp_path / "baseline.json"
    baseline_report.write_text("{}", encoding="utf-8")
    edited_report = tmp_path / "edited.json"
    edited_report.write_text(
        json.dumps(
            {
                "meta": {"device": "cpu", "adapter": "hf_gpt2"},
                "edit": {"name": "quant_rtn"},
                "metrics": {"primary_metric": {"final": {"bad": "value"}}},
            }
        ),
        encoding="utf-8",
    )

    def fake_latest(run_root: Path):
        return baseline_report if Path(run_root).name == "source" else edited_report

    monkeypatch.setattr(run_mod, "run_command", lambda **_: None, raising=False)
    monkeypatch.setattr(mod, "_latest_run_report", fake_latest)
    monkeypatch.setattr(mod, "_report", lambda **_: None, raising=False)
    monkeypatch.setattr(
        mod, "_resolve_exit_code", lambda err, profile: 9, raising=False
    )

    with pytest.raises(click.exceptions.Exit) as exc:
        mod.certify_command(
            source=str(src),
            edited=str(edt),
            adapter="hf_gpt2",
            out=str(Path("runs")),
            profile="ci",
        )

    assert exc.value.exit_code == 9


def test_certify_missing_baseline_report_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    # Fake run does not create any reports
    monkeypatch.setattr(run_mod, "run_command", lambda **k: None, raising=False)
    with pytest.raises(click.exceptions.Exit):
        mod.certify_command(
            source=str(src),
            edited=str(edt),
            adapter="hf_gpt2",
            out=str(tmp_path / "runs"),
        )


def test_certify_missing_edited_report_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    runs = tmp_path / "runs"

    # Baseline run produces a report, edited run produces nothing
    def fake_run(**kwargs):
        out = Path(kwargs["out"])
        if out.name == "source":
            _stub_run_dir(out)

    monkeypatch.setattr(run_mod, "run_command", fake_run, raising=False)
    with pytest.raises(click.exceptions.Exit):
        mod.certify_command(
            source=str(src),
            edited=str(edt),
            adapter="hf_gpt2",
            out=str(runs),
        )


def test_certify_edit_config_missing_exits(monkeypatch, tmp_path: Path):
    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    # Baseline and edited runs are stubbed out
    monkeypatch.setattr(run_mod, "run_command", lambda **k: None, raising=False)
    # Make sure there is at least some baseline report to bypass first exit
    _stub_run_dir(Path(tmp_path / "runs" / "source"))
    with pytest.raises(click.exceptions.Exit):
        mod.certify_command(
            source=str(src),
            edited=str(edt),
            adapter="hf_gpt2",
            edit_config=str(tmp_path / "missing_edit.yaml"),
            out=str(tmp_path / "runs"),
        )


def test_certify_happy_path_with_preset_and_auto_adapter(monkeypatch, tmp_path: Path):
    # Create a minimal fake preset
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        "model: { id: x }\nedit: { name: structured, plan: {} }\n", encoding="utf-8"
    )

    src = tmp_path / "src"
    edt = tmp_path / "edt"
    src.mkdir()
    edt.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}),
        encoding="utf-8",
    )

    runs = tmp_path / "runs"
    certs = tmp_path / "certs"
    calls = {"runs": 0, "reports": 0}

    def run_stub(**kwargs):
        calls["runs"] += 1
        out = Path(kwargs["out"])
        _stub_run_dir(out)

    def report_stub(**kwargs):
        calls["reports"] += 1

    with ExitStack() as stack:
        stack.enter_context(patch.object(run_mod, "run_command", run_stub))
        stack.enter_context(patch.object(mod, "_report", report_stub))
        mod.certify_command(
            source=str(src),
            edited=str(edt),
            adapter="auto",
            preset=str(preset),
            out=str(runs),
            cert_out=str(certs),
        )

    assert calls["runs"] == 2 and calls["reports"] == 1
